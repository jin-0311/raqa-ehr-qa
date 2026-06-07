"""Benchmark generator (MIMIC-IV release).

Original-style generation (sample a real patient as the anchor; the LLM freely chooses the
clinical concept) with the v7 details: 4 buckets (focus x difficulty) over C1-C5, a SQL and a
Cypher that must compute the SAME answer, execution validation (keep only non-empty meaningful
payloads -> query_type in {BOTH,RDB,GDB}), placeholder fill from the real payload, an inline LLM
judge, and per-prefix dedup. (No v7 entity-keyword pre-fetch; the concept is the LLM's choice.)

  python gen_benchmark.py               # full
  V7_TARGET=5 python gen_benchmark.py   # smoke (5 per bucket)
"""
from __future__ import annotations
import json, os, random, re, sys, time
from pathlib import Path
import pandas as pd
import pymysql
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from openai import OpenAI

client = OpenAI(api_key=config.OPENAI_API_KEY)
OUT_CSV = config.DATA_DIR / "mimic_benchmark.csv"

# ---- DB (from env via config) ----
def _mysql():
    kw = config.pymysql_kwargs(); kw["cursorclass"] = pymysql.cursors.DictCursor; return pymysql.connect(**kw)
_neo = {"d": None}
def _driver():
    if _neo["d"] is None:
        _neo["d"] = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    return _neo["d"]

# ---- compact unified MIMIC schema (RDB uppercase tables + GDB topology) ----
SCHEMA = """\
RDB (MySQL): Patient(patientId,gender,anchor_age), Admission(patientId,admissionId,admissionKey,
admissionDate,dischargeDate,daysOfStay,race,insurance), Diagnosis(patientId,admissionId,seq_num[1-5],
diagnosisCode,diagnosisName), Lab(patientId,admissionId,testCode,testName,reportDate,resultNumericValue,unit,flag),
Medi(patientId,admissionId,genericName,prescriptionDate), Operation(patientId,admissionId,operationCode,operationName),
Vital(patientId,admissionId,recordDate,heartRate,systolicBp,diastolicBp,respirationRate,oxygenSaturation) [WIDE].
IDs are STRINGS; match names via LIKE '%kw%'. Table names are EXACT CamelCase (case-sensitive). No Day table in RDB.
GDB (Neo4j): (Patient)-[:HAS_ADMISSION]->(Admission)-[:NEXT]->(Day{index:0,1,..})-[:NEXT]->(Day);
(Day{index:0})-[:HAS_DIAG]->(Diagnosis); (Day)-[:HAS_LAB|HAS_MEDI|HAS_VITAL|HAS_OPERATION]->(event).
Reach ALL days via MATCH (d:Day{admissionKey:a.admissionKey}) (NOT (a)-[:NEXT]->(:Day), the first day only).
Timeline = ORDER BY d.index. Numeric props are strings -> toFloat(); date props are Date (no datetime() rewrap).
Name match: toLower(x.testName) CONTAINS toLower('kw')."""

# suggested clinical concepts per entity (examples only; the LLM may pick others)
ENTITY_EG = {
    "lab": "Glucose, Hemoglobin, Potassium, Creatinine, Platelet Count, Lactate, INR(PT)",
    "medi": "Heparin, Insulin, Acetaminophen, Furosemide, Vancomycin, Warfarin",
    "vital": "heart rate, systolic blood pressure, respiratory rate, oxygen saturation",
    "diag": "hypertension, atrial fibrillation, pneumonia, acute kidney failure, sepsis, heart failure",
    "op": "Hemodialysis, central venous catheter, mechanical ventilation",
}
TARGETS = {b: int(os.environ.get("V7_TARGET", 20)) for b in ("pa_easy","pa_complex","pop_easy","pop_complex")}
BUCKET_CATS = {
    "pa_easy": ["patient_C1_lookup","patient_C2_agg"],
    "pa_complex": ["patient_C3_timeline","patient_C4_multihop","patient_C5_hybrid"],
    "pop_easy": ["pop_C1_lookup","pop_C2_agg"],
    "pop_complex": ["pop_C3_timeline","pop_C4_multihop","pop_C5_hybrid"],
}
CAT_SPECS = {
    "patient_C1_lookup":   ("easy", ["lab","medi","vital","op","diag"], "Single-entity lookup/count for one patient."),
    "patient_C2_agg":      ("easy", ["lab","medi","vital","op"], "Filtered aggregate of an entity for one patient."),
    "patient_C3_timeline": ("complex", ["lab","vital"], "Temporal trend of a lab/vital over the patient's days (ORDER BY d.index)."),
    "patient_C4_multihop": ("complex", ["lab","vital","medi","op"], "Relate a first-day diagnosis to a downstream entity for one patient."),
    "patient_C5_hybrid":   ("complex", ["lab","vital","medi","op","diag"], "Combine two entity tables for one patient via a clinical link."),
    "pop_C1_lookup":   ("easy", ["lab","medi","vital","op","diag"], "Cohort lookup/count filtered by a demographic."),
    "pop_C2_agg":      ("easy", ["lab","medi","vital","op"], "Cohort-wide aggregate with a patient-level filter."),
    "pop_C3_timeline": ("complex", ["lab","vital"], "Cohort temporal trend over days/years for a subgroup."),
    "pop_C4_multihop": ("complex", ["lab","vital","medi","op"], "Cohort: patients in a subgroup with diagnosis X and entity-Y pattern."),
    "pop_C5_hybrid":   ("complex", ["lab","vital","diag"], "Cohort: combine two entity tables for a demographic subgroup."),
}

# ---- DB helpers ----
def exec_sql(sql):
    try:
        with _mysql() as c, c.cursor() as cur:
            cur.execute(sql); return True, cur.fetchall(), ""
    except Exception as e:
        return False, None, str(e)[:200]

def exec_cypher(cy):
    if not cy.strip(): return False, None, "empty"
    try:
        with _driver().session() as s: return True, [dict(r) for r in s.run(cy)], ""
    except Exception as e:
        return False, None, str(e)[:200]

def sample_patients(limit=120):
    """Original-style anchor: sample real patient IDs with enough coverage (no entity pre-fetch)."""
    sql = ("SELECT p.patientId,(SELECT COUNT(*) FROM Lab WHERE patientId=p.patientId) n_lab,"
           "(SELECT COUNT(*) FROM Diagnosis WHERE patientId=p.patientId) n_diag FROM Patient p "
           f"HAVING n_lab>=10 AND n_diag>=2 ORDER BY n_lab+n_diag DESC LIMIT {limit}")
    ok, rows, _ = exec_sql(sql); return [str(r["patientId"]) for r in rows] if ok else []

def is_useful(p):
    if not isinstance(p, list) or not p: return False
    f = p[0]
    if not isinstance(f, dict): return True
    for k, v in f.items():
        if "id" in k.lower() or "key" in k.lower() or isinstance(v, bool): continue
        if isinstance(v, (int, float)) and v not in (0, None): return True
        if isinstance(v, str) and v.strip() and v.lower() not in ("0","none","null","nan","0.0"): return True
    return False

def _scalars(p):
    out = []
    if isinstance(p, list) and p and isinstance(p[0], dict):
        for v in p[0].values():
            try: out.append(round(float(v), 3))
            except (TypeError, ValueError): pass
    return out

def scalar_match(a, b, tol=0.05):
    sa, sb = _scalars(a), _scalars(b)
    if not sa or not sb: return False
    return all(abs(x-y) <= tol*max(1.0, abs(x), abs(y)) for x, y in zip(sa, sb))

def fill_placeholders(tpl, payload, keys):
    if not tpl or not payload: return tpl
    first = payload[0] if isinstance(payload, list) and payload else {}
    if not isinstance(first, dict): return tpl
    r = tpl
    for ph, key in zip(re.findall(r"\{\{?([a-zA-Z_]+)\}?\}", tpl), keys or []):
        r = re.sub(rf"\{{\{{?{re.escape(ph)}\}}?\}}", str(first.get(key, "?")), r, count=1)
    for ph in re.findall(r"\{\{?([a-zA-Z_]+)\}?\}", r):
        for k in first:
            if ph.lower() in k.lower():
                r = re.sub(rf"\{{\{{?{re.escape(ph)}\}}?\}}", str(first[k]), r, count=1); break
    return r

# ---- LLM (terse prompts) ----
SYS_GEN = "EHR-QA hybrid benchmark generator. Strict JSON. SQL and Cypher must agree. Maximize clinical relevance and diversity; never reuse a template."

def build_prompt(cat, entity, anchor, focus):
    diff, _, shape = CAT_SPECS[cat]
    who = (f"Single patient patientId='{anchor}' (STRING)." if focus == "patient"
           else "A specific cohort by a demographic filter (anchor_age/gender/race/insurance); aggregate over patients; no patientId.")
    return (f"Generate ONE {focus}/{diff} EHR-QA item, category {cat}. {shape}\n"
            f"Target entity {entity} (e.g. {ENTITY_EG.get(entity,'')}) — pick a concept yourself; the question MUST involve it. {who}\n"
            f"Ask something a clinician would; vary the shape; avoid stereotyped 'top-5/total count' phrasings.\n"
            f"Schema (use EXACT names):\n{SCHEMA}\n"
            "Return JSON: {natural_question, sql_query, cypher_query (same answer), "
            "natural_answer_template (with {placeholder}), answer_value_keys[], clinical_rationale}.")

def call_llm(prompt, sys=SYS_GEN, temp=0.7, retries=2):
    for i in range(retries+1):
        try:
            r = client.chat.completions.create(model=config.LLM_MODEL, temperature=temp, max_tokens=1600,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}])
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            if i == retries: print("  [LLM_ERR]", e)
            else: time.sleep(2)
    return None

def judge_one(row):
    """Inline LLM judge (terse) — accept clinically meaningful, correct items only."""
    p = (f"Rate this EHR-QA item. Q: {row['natural_question']}\nSQL: {row['sql_query'][:400]}\n"
         f"Answer: {row['natural_answer']}\nReturn JSON {{overall_decision: accept|revise|reject, comment}}.")
    d = call_llm(p, sys="Strict EHR-QA reviewer. JSON only.", temp=0.0) or {}
    return d.get("overall_decision", "reject")

def per_cat_targets(bucket, n):
    cats = BUCKET_CATS[bucket]; base, rem = divmod(n, len(cats))
    return {c: base + (1 if i < rem else 0) for i, c in enumerate(cats)}

def prefix_dedup(rows, w=10, cap=2):
    seen, out = {}, []
    for r in rows:
        k = (r["category"], " ".join(r["natural_question"].lower().split()[:w]))
        if seen.get(k, 0) < cap:
            seen[k] = seen.get(k, 0) + 1; out.append(r)
    return out

# ---- orchestrator ----
def generate():
    pool = sample_patients()
    print(f"patient pool: {len(pool)}")
    if not pool: print("[abort] no patients (build the DB first)"); return
    rng = random.Random(int(os.environ.get("V7_SEED", "20260605")))
    accepted, qid = [], 1
    for bucket, btarget in TARGETS.items():
        for cat, want in per_cat_targets(bucket, btarget).items():
            diff, ents, _ = CAT_SPECS[cat]; is_p = cat.startswith("patient"); got = att = 0
            while got < want and att < want*10:
                att += 1
                entity = ents[(got+att) % len(ents)]
                anchor = rng.choice(pool) if is_p else None
                d = call_llm(build_prompt(cat, entity, anchor, "patient" if is_p else "cohort"))
                if not d: continue
                sql, cyq = str(d.get("sql_query","")).strip(), str(d.get("cypher_query","")).strip()
                ok_s, sp, _ = exec_sql(sql) if sql else (False, None, "")
                ok_c, cp, _ = exec_cypher(cyq)
                sgood, cgood = ok_s and is_useful(sp), ok_c and is_useful(cp)
                if not (sgood or cgood): continue           # execution validation (keep only working items)
                if sgood and cgood: qt, gm, canon = "BOTH", scalar_match(sp, cp), sp
                elif sgood:         qt, gm, canon = "RDB", None, sp
                else:               qt, gm, canon = "GDB", None, cp
                ans = fill_placeholders(str(d.get("natural_answer_template","")), canon, d.get("answer_value_keys", []))
                if re.search(r"\{\{?[a-zA-Z0-9_]+\}?\}", ans): continue
                row = {"question_id": qid, "category": cat, "difficulty": diff, "query_type": qt,
                       "focus_type": "patient_centric" if is_p else "population_centric", "bucket": bucket,
                       "anchor_patient_id": anchor or "", "gdb_match": gm,
                       "natural_question": str(d.get("natural_question","")).strip(),
                       "query": ("Step 1) RDB: "+sql+"  Step 2) GDB: "+cyq) if qt=="BOTH" else (sql if qt=="RDB" else "GDB: "+cyq),
                       "sql_query": sql, "cypher_query": cyq, "natural_answer": ans,
                       "clinical_rationale": str(d.get("clinical_rationale","")).strip()}
                if judge_one(row) != "accept": continue
                accepted.append(row); got += 1; qid += 1
                print(f"  [{len(accepted)}] {bucket}/{cat} {qt} got={got}/{want}")
    final = prefix_dedup(accepted)
    pd.DataFrame(final).to_csv(OUT_CSV, index=False)
    print(f"\nFINAL {len(final)} rows -> {OUT_CSV}  (run judge.py for an extra accept filter)")

if __name__ == "__main__":
    generate()
    if _neo["d"]: _neo["d"].close()
