"""
MIMIC-IV hybrid DB build — unified RDB/GDB schema.

Goals
-----
1. Diagnoses attach to the first Day (index 0) via HAS_DIAG (not directly to Admission).
2. In-stay events only: lab/medi/vital/op whose event date falls within admit~discharge.
3. vital long->wide pivot: label -> {heartRate, respirationRate, systolicBp, diastolicBp,
   temperature, oxygenSaturation}.
4. Column rename: subject_id->patientId, hadm_id->admissionId, charttime->event date, etc.
5. lab test name (LABEL) preserved as testName.

Topology
--------
(Patient {patientId})-[:HAS_ADMISSION]->(Admission {admissionKey})
(Patient)-[:NEXT]->(Admission)                 # branching
(Admission)-[:HAS_ADMIN_RECORD]->(AdminRecord)
(Admission)-[:NEXT]->(Day)-[:NEXT]->(Day)      # one Day per in-stay date, linear chain
(Day{index=0})-[:HAS_DIAG]->(Diagnosis)        # diagnoses on the first day
(Day)-[:HAS_LAB]->(Lab) / HAS_MEDI / HAS_VITAL / HAS_OPERATION / HAS_PHYSICAL  # event-day attach

RDB (db) — uppercase operational tables
  Patient, Admission, Diagnosis, Lab, Medi, Operation, Vital, Physical

Run:  python build_mimic_db.py  [--no-rdb] [--no-gdb]
"""
from __future__ import annotations
import sys
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sqlalchemy import create_engine

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import config

SAMPLE = str(config.DATA_DIR)
NEO4J = (config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD)
MYSQL = config.MYSQL_URI

# ---------------- vital / physical pivot maps ----------------
VITAL_MAP = {
    "Heart Rate": "heartRate",
    "Respiratory Rate": "respirationRate",
    "O2 saturation pulseoxymetry": "oxygenSaturation",
    "Non Invasive Blood Pressure systolic": "systolicBp",
    "Non Invasive Blood Pressure diastolic": "diastolicBp",
    "Arterial Blood Pressure systolic": "systolicBp",
    "Arterial Blood Pressure diastolic": "diastolicBp",
    "Temperature Celsius": "temperature",
    "Temperature Fahrenheit": "_tempF",      # converted below
}
PHYS_MAP = {
    "Height (cm)": "height", "Height": "height",
    "Admission Weight (Kg)": "weight", "Daily Weight": "weight",
    "Admission Weight (lbs.)": "_weightLb",  # converted below
}


def _date8(s: pd.Series) -> pd.Series:
    """Any timestamp/datestr → YYYYMMDD (8 chars). MIMIC charttime e.g. '2147-12-02 16:50:00'."""
    return s.astype(str).str.replace(r"\D", "", regex=True).str[:8]


def load():
    d = {}
    for f in ["patient", "admin", "diag", "lab", "medi", "op", "physical", "vital"]:
        d[f] = pd.read_csv(f"{SAMPLE}/{f}_sample.csv")
        d[f]["subject_id"] = d[f]["subject_id"].astype(str).str.strip()
        if "hadm_id" in d[f].columns:
            d[f]["hadm_id"] = d[f]["hadm_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    return d


def build_frames(d):
    """Rename to site-A style + pivot vital/physical + admissionKey. Returns dict of clean DFs."""
    out = {}

    # ---- Patient ----
    p = d["patient"].rename(columns={"subject_id": "patientId"})
    out["Patient"] = p

    # ---- Admission ----
    a = d["admin"].rename(columns={
        "subject_id": "patientId", "hadm_id": "admissionId",
        "admittime": "admissionDate", "dischtime": "dischargeDate",
        "admission_type": "visitType", "admission_location": "admissionPath",
        "los": "daysOfStay",
    })
    a["admissionKey"] = a["patientId"] + "_" + a["admissionId"]
    a["admissionDate"] = _date8(a["admissionDate"])
    a["dischargeDate"] = _date8(a["dischargeDate"])
    a = a[(a["admissionDate"].str.len() == 8) & (a["dischargeDate"].str.len() == 8)].copy()
    out["Admission"] = a

    # admit/discharge lookup for in-stay filtering
    span = a[["patientId", "admissionId", "admissionDate", "dischargeDate"]].copy()
    span["adm_i"] = span["admissionDate"].astype(int)
    span["dis_i"] = span["dischargeDate"].astype(int)
    span_map = {(r.patientId, r.admissionId): (r.adm_i, r.dis_i) for r in span.itertuples()}

    def in_stay(df, date_col):
        """Keep rows whose event date is within its admission's admit~discharge."""
        df = df.copy()
        df["_d"] = _date8(df[date_col])
        df = df[df["_d"].str.len() == 8].copy()
        def ok(r):
            sp = span_map.get((r.patientId, r.admissionId))
            if not sp:
                return False
            return sp[0] <= int(r._d) <= sp[1]
        keep = df.apply(ok, axis=1)
        df = df[keep].drop(columns=["_d"])
        return df

    # ---- Diagnosis (seq_num 1-5) ----
    dg = d["diag"].rename(columns={
        "subject_id": "patientId", "hadm_id": "admissionId",
        "icd_code": "diagnosisCode", "DICD": "diagnosisName",
    })
    dg = dg[dg["seq_num"].isin([1, 2, 3, 4, 5])].copy()
    # diagnosis has no own date → will attach to first day; keep admissionId
    out["Diagnosis"] = dg

    # ---- Lab ----
    lb = d["lab"].rename(columns={
        "subject_id": "patientId", "hadm_id": "admissionId",
        "itemid": "testCode", "LABEL": "testName", "charttime": "reportDate",
        "valuenum": "resultNumericValue", "value": "resultValue",
        "valueuom": "unit", "ref_range_lower": "lowerLimit", "ref_range_upper": "upperLimit",
    })
    lb = in_stay(lb, "reportDate")
    lb["reportDate"] = _date8(lb["reportDate"])
    out["Lab"] = lb

    # ---- Medi ----
    md = d["medi"].rename(columns={
        "subject_id": "patientId", "hadm_id": "admissionId",
        "medication": "genericName", "charttime": "prescriptionDate",
    })
    md = in_stay(md, "prescriptionDate")
    md["prescriptionDate"] = _date8(md["prescriptionDate"])
    out["Medi"] = md

    # ---- Operation ----
    op = d["op"].rename(columns={
        "subject_id": "patientId", "hadm_id": "admissionId",
        "icd_code": "operationCode", "OICD": "operationName", "chartdate": "operationDate",
    })
    op = in_stay(op, "operationDate")
    op["operationDate"] = _date8(op["operationDate"])
    out["Operation"] = op

    # ---- Vital (long → wide) ----
    v = d["vital"].copy()
    v["mapped"] = v["label"].map(VITAL_MAP)
    v = v[v["mapped"].notna()].copy()
    # Fahrenheit → Celsius
    fmask = v["mapped"] == "_tempF"
    v.loc[fmask, "valuenum"] = (pd.to_numeric(v.loc[fmask, "valuenum"], errors="coerce") - 32) * 5.0 / 9.0
    v.loc[fmask, "mapped"] = "temperature"
    v = v.rename(columns={"subject_id": "patientId", "hadm_id": "admissionId", "charttime": "recordDate"})
    v["valuenum"] = pd.to_numeric(v["valuenum"], errors="coerce")
    vw = v.pivot_table(index=["patientId", "admissionId", "recordDate"],
                       columns="mapped", values="valuenum", aggfunc="mean").reset_index()
    vw.columns.name = None
    for c in ["systolicBp", "diastolicBp", "heartRate", "respirationRate", "temperature", "oxygenSaturation"]:
        if c not in vw.columns:
            vw[c] = np.nan
    vw = in_stay(vw, "recordDate")
    vw["recordDate"] = _date8(vw["recordDate"])
    out["Vital"] = vw

    # ---- Physical (long → wide) ----
    ph = d["physical"].copy()
    ph["mapped"] = ph["label"].map(PHYS_MAP)
    ph = ph[ph["mapped"].notna()].copy()
    lbmask = ph["mapped"] == "_weightLb"
    ph.loc[lbmask, "valuenum"] = pd.to_numeric(ph.loc[lbmask, "valuenum"], errors="coerce") * 0.453592
    ph.loc[lbmask, "mapped"] = "weight"
    ph = ph.rename(columns={"subject_id": "patientId", "hadm_id": "admissionId", "charttime": "measurementDate"})
    ph["valuenum"] = pd.to_numeric(ph["valuenum"], errors="coerce")
    phw = ph.pivot_table(index=["patientId", "admissionId", "measurementDate"],
                         columns="mapped", values="valuenum", aggfunc="mean").reset_index()
    phw.columns.name = None
    for c in ["height", "weight"]:
        if c not in phw.columns:
            phw[c] = np.nan
    phw = in_stay(phw, "measurementDate")
    phw["measurementDate"] = _date8(phw["measurementDate"])
    out["Physical"] = phw

    return out


# ======================= RDB =======================
def load_rdb(frames):
    eng = create_engine(MYSQL)
    print("\n[RDB] writing uppercase operational tables to mysql_mimic4 ...")
    for name, df in frames.items():
        df.to_sql(name=name, con=eng, if_exists="replace", index=False)
        print(f"  {name}: {len(df)} rows")
    eng.dispose()


# ======================= GDB =======================
def gdb_session():
    uri, u, pw = NEO4J
    return GraphDatabase.driver(uri, auth=(u, pw))


def load_gdb(frames):
    drv = gdb_session()
    P, A, DG = frames["Patient"], frames["Admission"], frames["Diagnosis"]
    with drv.session() as s:
        print("\n[GDB] wiping MIMIC graph (7691) ...")
        s.run("MATCH (n) DETACH DELETE n")
        s.run("CREATE CONSTRAINT mimic_admkey IF NOT EXISTS FOR (a:Admission) REQUIRE a.admissionKey IS UNIQUE")
        s.run("CREATE CONSTRAINT mimic_daykey IF NOT EXISTS FOR (d:Day) REQUIRE d.dayKey IS UNIQUE")

        # Patient
        prows = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in P.to_dict("records")]
        s.run("UNWIND $rows AS r MERGE (p:Patient {patientId: toString(r.patientId)}) SET p += r", rows=prows)
        print(f"  Patient nodes: {P['patientId'].nunique()}")

        # Admission + AdminRecord + HAS_ADMISSION
        arows = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in A.to_dict("records")]
        s.run("""
        UNWIND $rows AS r
        MATCH (p:Patient {patientId: toString(r.patientId)})
        MERGE (a:Admission {admissionKey: toString(r.admissionKey)})
          ON CREATE SET a.admitDate = date({year:toInteger(substring(r.admissionDate,0,4)),
                                             month:toInteger(substring(r.admissionDate,4,2)),
                                             day:toInteger(substring(r.admissionDate,6,2))}),
                        a.dischargeDate = date({year:toInteger(substring(r.dischargeDate,0,4)),
                                                month:toInteger(substring(r.dischargeDate,4,2)),
                                                day:toInteger(substring(r.dischargeDate,6,2))}),
                        a.patientId = toString(r.patientId),
                        a.admissionId = toString(r.admissionId)
        MERGE (p)-[:HAS_ADMISSION]->(a)
        MERGE (p)-[:NEXT]->(a)
        MERGE (ar:AdminRecord {admissionKey: toString(r.admissionKey)}) SET ar += r
        MERGE (a)-[:HAS_ADMIN_RECORD]->(ar)
        """, rows=arows)
        print(f"  Admission nodes: {A['admissionKey'].nunique()}")

        # Days within each admission + linear NEXT chain
        s.run("""
        MATCH (a:Admission)
        WITH a, duration.inDays(a.admitDate, a.dischargeDate).days AS nd
        WITH a, CASE WHEN nd >= 0 THEN range(0, nd) ELSE [0] END AS offs
        UNWIND offs AS i
        WITH a, i, (a.admitDate + duration({days:i})) AS dd
        MERGE (day:Day {dayKey: a.admissionKey + '_' + toString(dd)})
          ON CREATE SET day.date = dd, day.index = i,
                        day.patientId = a.patientId, day.admissionKey = a.admissionKey
        """)
        # Admission → first Day
        s.run("""
        MATCH (a:Admission)
        MATCH (d:Day {admissionKey: a.admissionKey})
        WITH a, d ORDER BY d.index
        WITH a, head(collect(d)) AS firstDay
        WHERE firstDay IS NOT NULL
        MERGE (a)-[:NEXT]->(firstDay)
        """)
        # Day → Day linear chain within an admission
        s.run("""
        MATCH (a:Admission)
        MATCH (d:Day {admissionKey: a.admissionKey})
        WITH a, d ORDER BY d.index
        WITH a, collect(d) AS days
        UNWIND range(0, size(days)-2) AS i
        WITH days[i] AS d1, days[i+1] AS d2
        MERGE (d1)-[:NEXT]->(d2)
        """)
        nday = s.run("MATCH (d:Day) RETURN count(d) AS n").single()["n"]
        print(f"  Day nodes: {nday}")

        # Diagnosis → first day (index 0)
        dgrows = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in DG.to_dict("records")]
        for idx, r in enumerate(dgrows):
            r["_rid"] = f"dg_{idx}"          # unique per row (RDB has dup seq_num rows)
        s.run("""
        UNWIND $rows AS r
        MATCH (d:Day {admissionKey: toString(r.patientId)+'_'+toString(r.admissionId), index:0})
        MERGE (x:Diagnosis {rowId: r._rid})
        SET x += r
        MERGE (d)-[:HAS_DIAG]->(x)
        """, rows=dgrows)
        ndg = s.run("MATCH (:Day)-[:HAS_DIAG]->(x:Diagnosis) RETURN count(DISTINCT x) AS n").single()["n"]
        print(f"  Diagnosis (first-day): {ndg}")

        # Event tables → day by (patientId + date) within admission
        def attach(df, label, rel, date_col):
            rows = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in df.to_dict("records")]
            # rowId for uniqueness
            for idx, r in enumerate(rows):
                r["_rid"] = f"{label}_{idx}"
            s.run(f"""
            UNWIND $rows AS r
            WITH r, date({{year:toInteger(substring(r.{date_col},0,4)),
                          month:toInteger(substring(r.{date_col},4,2)),
                          day:toInteger(substring(r.{date_col},6,2))}}) AS dt
            MATCH (d:Day {{admissionKey: toString(r.patientId)+'_'+toString(r.admissionId)}})
            WHERE d.date = dt
            MERGE (n:{label} {{rowId: r._rid}}) SET n += r
            MERGE (d)-[:{rel}]->(n)
            """, rows=rows)
            cnt = s.run(f"MATCH (:Day)-[:{rel}]->(x:{label}) RETURN count(DISTINCT x) AS n").single()["n"]
            print(f"  {label}: {cnt} attached (of {len(df)})")

        attach(frames["Lab"], "Lab", "HAS_LAB", "reportDate")
        attach(frames["Medi"], "Medi", "HAS_MEDI", "prescriptionDate")
        attach(frames["Vital"], "Vital", "HAS_VITAL", "recordDate")
        attach(frames["Operation"], "Operation", "HAS_OPERATION", "operationDate")
        attach(frames["Physical"], "Physical", "HAS_PHYSICAL", "measurementDate")
    drv.close()


def main():
    args = sys.argv[1:]
    d = load()
    print("loaded sample CSVs:", {k: len(v) for k, v in d.items()})
    frames = build_frames(d)
    print("\nprepared frames:")
    for k, v in frames.items():
        print(f"  {k}: {len(v)} rows, cols={list(v.columns)[:8]}...")
    if "--no-rdb" not in args:
        load_rdb(frames)
    if "--no-gdb" not in args:
        load_gdb(frames)
    print("\n✅ MIMIC v6 DB build done.")


if __name__ == "__main__":
    main()
