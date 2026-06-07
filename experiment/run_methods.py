"""
6-method ReAct experiment for ICDM 2026 (fully-learned RAQA).

Methods
-------
1. RDB-fixed (1-query)
2. GDB-fixed (1-query)
3. RDB+GDB split-merge (2-query, LLM-merged)
4. Hybrid + r_h     (v1, hand-tuned risk, ω cost)
5. Hybrid + r_θ     (learned risk, ω cost)
6. Hybrid + r_θ + c_θ  (fully learned RAQA — r_θ inject + c_θ override expected_cost)

Datasets
--------
(MIMIC-only release)
MIMIC: final_datasets/mimic_final_100.csv  (80 questions, 4 buckets)

Run
---
  python run_methods.py mimic
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# Agent code is bundled under agents/. react_agent loads agent_base from the same
# folder via here.with_name(); the schema is obtained by DB introspection (schema-agnostic).
AGENTS = Path(__file__).resolve().parent.parent / "agents"
MIMIC_AGENT = AGENTS / "agent.py"   # single merged agent (base + ReAct)

MIMIC_CSV = Path(os.environ.get("RAQA_MIMIC_CSV", str(config.DATA_DIR / "mimic_benchmark.csv")))
R_THETA_PATH = config.BASE / "models/r_theta_lightgbm_isotonic.joblib"
C_THETA_PATH = config.BASE / "models/c_theta_lightgbm.joblib"

OUT_DIR = Path(os.environ.get("RAQA_OUT_DIR", str(config.ARTIFACT_DIR / "results")))
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"  MIMIC_CSV = {MIMIC_CSV}")
print(f"  OUT_DIR   = {OUT_DIR}")


# --- env ---
def load_openai_key():
    if config.OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

def set_mimic_env():
    os.environ["MYSQL_URI"] = config.MYSQL_URI
    os.environ["NEO4J_URI"] = config.NEO4J_URI
    os.environ["NEO4J_USER"] = config.NEO4J_USER
    os.environ["NEO4J_PASSWORD"] = config.NEO4J_PASSWORD

# --- r_θ / c_θ ---
print(f"Loading r_θ: {R_THETA_PATH}")
R_THETA = joblib.load(R_THETA_PATH)
print(f"Loading c_θ: {C_THETA_PATH}")
C_THETA = joblib.load(C_THETA_PATH)

QFK = ["q_len","has_optional","has_distinct","has_count","has_match","has_join",
       "has_group_by","has_order_by","has_with","is_multi_step","step_count","optional_count","q_has_next"]

def qfeats(q):
    if not isinstance(q, str) or not q: return {k:0 for k in QFK}
    s = q.lower(); sc = max(len(re.findall(r"step\s*\d+", s)), 1)
    return {"q_len":min(len(q),5000),"has_optional":int("optional" in s),"has_distinct":int("distinct" in s),
            "has_count":int("count(" in s or "count (" in s),"has_match":int("match(" in s or "match (" in s),
            "has_join":int(" join " in s),"has_group_by":int("group by" in s),"has_order_by":int("order by" in s),
            "has_with":int(" with " in s or s.startswith("with ")),"is_multi_step":int(sc>=2),
            "step_count":sc,"optional_count":s.count("optional"),"q_has_next":int(":next" in s or " next " in s)}


# --- dynamic import ---
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


def build_evaluators(LegacyCAQA):
    """Returns (RThetaEval, RThetaCThetaEval) — two CAQAEvaluator subclasses."""

    class _BaseCtx:
        _ctx: Dict[str, str] = {"category":"unknown","difficulty":"unknown","query_type":"unknown"}
        @classmethod
        def set_ctx(cls, c, d, qt):
            cls._ctx = {"category":str(c),"difficulty":str(d),"query_type":str(qt)}

    def feat_row(self, plan):
        q = str(plan.get("query") or "")
        f = qfeats(q)
        f["db_type"] = str(plan.get("db_type","UNKNOWN"))
        f["category"] = self._ctx["category"]
        f["difficulty"] = self._ctx["difficulty"]
        f["query_type"] = self._ctx["query_type"]
        raw = plan.get("predicted_error_risk")
        f["planner_risk"] = float(raw) if isinstance(raw,(int,float)) else 0.5
        return f

    class RThetaEval(LegacyCAQA, _BaseCtx):
        _feat_row = feat_row
        def calculate_caqa_scores(self, plans_json, actual_cardinalities=None,
                                  predicted_risks=None, risk_lambda=1.0,
                                  normalize_expected_cost=True, user_query=""):
            if predicted_risks is None:
                plans = plans_json["plans"] if isinstance(plans_json, dict) else json.loads(plans_json)["plans"]
                X = pd.DataFrame([self._feat_row(p) for p in plans])
                try:
                    probs = R_THETA.predict_proba(X)[:, 1]
                    predicted_risks = {str(p.get("plan_id","")): float(prob) for p, prob in zip(plans, probs)}
                except Exception as exc:
                    print(f"  [r_θ ERR] {exc}"); predicted_risks = None
            return super().calculate_caqa_scores(plans_json=plans_json,
                actual_cardinalities=actual_cardinalities, predicted_risks=predicted_risks,
                risk_lambda=risk_lambda, normalize_expected_cost=normalize_expected_cost,
                user_query=user_query)

    class RThetaCThetaEval(RThetaEval):
        """r_θ + c_θ: also override expected_cost using c_θ predicted latency."""
        def calculate_caqa_scores(self, plans_json, actual_cardinalities=None,
                                  predicted_risks=None, risk_lambda=1.0,
                                  normalize_expected_cost=True, user_query=""):
            # First: parent (RThetaEval) computes everything including r_θ risk + ω·N cost
            scored = super().calculate_caqa_scores(plans_json=plans_json,
                actual_cardinalities=actual_cardinalities, predicted_risks=predicted_risks,
                risk_lambda=risk_lambda, normalize_expected_cost=normalize_expected_cost,
                user_query=user_query)
            # Then: override expected_cost using c_θ
            plans = scored["plans"]
            X = pd.DataFrame([self._feat_row(p) for p in plans])
            try:
                log_lats = C_THETA.predict(X)
                pred_lats = np.exp(log_lats)
            except Exception as exc:
                print(f"  [c_θ ERR] {exc}"); return scored
            # Min-max normalize within candidate set
            if normalize_expected_cost and len(pred_lats) > 0:
                lo, hi = float(pred_lats.min()), float(pred_lats.max())
                if hi - lo > 1e-9:
                    norm = [(c - lo) / (hi - lo) for c in pred_lats]
                else:
                    norm = [0.0] * len(pred_lats)
            else:
                norm = list(pred_lats)
            for plan, raw_lat, n_cost in zip(plans, pred_lats, norm):
                plan["expected_cost"] = float(raw_lat)
                plan["expected_cost_normalized"] = float(n_cost)
                plan["caqa_score"] = float(raw_lat)
                risk = float(plan.get("predicted_error_risk", 0.0))
                pi = float(plan.get("semantic_penalty", 0.0))
                plan["risk_aware_caqa_score"] = float(n_cost) + float(risk_lambda) * risk + pi
                plan["selection_objective"] = "c_theta_normalized + lambda*r_theta + semantic_penalty"
            scored["caqa_config"]["selection_objective"] = plans[0]["selection_objective"] if plans else ""
            scored["caqa_config"]["cost_source"] = "c_theta_lightgbm"
            return scored

    return RThetaEval, RThetaCThetaEval


class PatchCtx:
    """Swap CAQAEvaluator on the single merged agent module during a run."""
    def __init__(self, agent_mod, new_cls):
        self.agent_mod, self.new_cls = agent_mod, new_cls
        self.old = None
    def __enter__(self):
        self.old = getattr(self.agent_mod, "CAQAEvaluator", None)
        if self.old is not None:
            self.agent_mod.CAQAEvaluator = self.new_cls
        return self
    def __exit__(self, *a):
        if self.old is not None:
            self.agent_mod.CAQAEvaluator = self.old


class Noop:
    def __enter__(self): return self
    def __exit__(self, *a): pass


def run_dataset(ds, csv_path, agent_py, env_set):
    env_set(); load_openai_key()
    print(f"\n{'='*70}\n>>> {ds}: {csv_path}\n{'='*70}")
    agent = _load(f"agent_{ds}", agent_py)
    LegacyCAQA = agent.CAQAEvaluator
    ReActHybridAgent = getattr(agent, "ReActHybridAgent", None) or agent.BaseHybridAgent
    RThetaEval, RThetaCThetaEval = build_evaluators(LegacyCAQA)

    from neo4j import GraphDatabase
    from langchain_community.utilities import SQLDatabase
    from openai import OpenAI
    rdb = SQLDatabase.from_uri(os.environ["MYSQL_URI"])
    neo4j_driver = GraphDatabase.driver(os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    SYS = getattr(getattr(agent,"OpenAILLMBase",None),"SYSTEM_PROMPT","") or "Helpful assistant."

    class LLM:
        def __init__(self, model="gpt-4.1-mini", temperature=0.0, max_tokens=2200):
            self.client = client; self.model = model; self.temperature = temperature; self.max_tokens = max_tokens
            self.total_prompt_tokens=self.total_completion_tokens=self.total_tokens=self.total_calls=0
            self.total_llm_latency_sec=0.0
        def invoke(self, p, json_mode=False):
            kw = {"model":self.model,"messages":[{"role":"system","content":SYS},{"role":"user","content":p}],
                  "temperature":self.temperature,"max_tokens":self.max_tokens}
            if json_mode: kw["response_format"]={"type":"json_object"}
            t0=time.time(); r=self.client.chat.completions.create(**kw)
            self.total_llm_latency_sec += time.time()-t0
            c=(r.choices[0].message.content or "").strip()
            u=getattr(r,"usage",None)
            if u:
                self.total_prompt_tokens+=int(getattr(u,"prompt_tokens",0) or 0)
                self.total_completion_tokens+=int(getattr(u,"completion_tokens",0) or 0)
                self.total_tokens+=int(getattr(u,"total_tokens",0) or 0)
            self.total_calls+=1; return c
    llm = LLM()

    def judge(q, gold, ma):
        prompt = (f"Judge whether MODEL_ANSWER is semantically equivalent to GOLD_ANSWER. "
                 f"Numeric ±10% OK. Different units/patients/aggregations = wrong.\n\n"
                 f"QUESTION:\n{q}\n\nGOLD:\n{gold}\n\nMODEL:\n{ma}\n\n"
                 'Return JSON: {"correct":0 or 1,"reason":"<short>"}')
        try:
            raw = llm.invoke(prompt, json_mode=True)
            d = json.loads(re.search(r"\{[\s\S]*\}", raw).group(0))
            return int(d.get("correct", 0))
        except Exception:
            return 0

    def direct_query(question, db_type):
        try:
            sa = ReActHybridAgent(rdb=rdb, neo4j_driver=neo4j_driver, llm=llm, enable_recovery=False, recovery_policy="never", debug=False)
            mk = sa.meta_kg_generator.generate_meta_kg()
            pk = sa._build_schema_context(question, mk, table_names=None)
            sc = str(pk.get("schema_context",""))
            task = "Generate exactly one executable SQL query." if db_type=="RDB" else "Generate exactly one executable Cypher query."
            kk = "sql_query" if db_type=="RDB" else "cypher_query"
            p = f"You are a database query generator.\n{task}\n\nUSER_QUESTION:\n{question}\n\nSCHEMA_CONTEXT:\n{sc}\n\nRules: only schema names; simple; Return JSON only: {{\"{kk}\":\"...\"}}"
            raw = llm.invoke(p, json_mode=True); d = json.loads(re.search(r"\{[\s\S]*\}", raw).group(0))
            query = str(d.get(kk,"")).strip()
            if not query: raise ValueError("empty query")
            if db_type == "RDB":
                pl_raw = rdb.run(query)
                try: import ast as _a; pl = _a.literal_eval(pl_raw)
                except: pl = pl_raw
            else:
                with neo4j_driver.session() as s: pl = s.run(query).data()
            answer = llm.invoke(f"Write one concise factual answer in English.\nQUESTION: {question}\nEXECUTED_RESULT: {str(pl)[:2000]}", json_mode=False)
            return {"success":1,"query":query,"payload":pl,"answer":answer,"db_type":db_type,"risk":float("nan")}
        except Exception as exc:
            return {"success":0,"query":"","payload":None,"answer":"","db_type":db_type,"risk":float("nan"),"error":str(exc)[:300]}

    def split_merge(question):
        rdb_r = direct_query(question, "RDB")
        gdb_r = direct_query(question, "GDB")
        if rdb_r["success"] == 0 and gdb_r["success"] == 0:
            return {"success":0,"query":"","payload":None,"answer":"","db_type":"split-merge","risk":float("nan")}
        if rdb_r["success"] == 1 and gdb_r["success"] == 0:
            return {**rdb_r, "db_type":"split-merge", "query":f"[RDB]{rdb_r['query']}"}
        if gdb_r["success"] == 1 and rdb_r["success"] == 0:
            return {**gdb_r, "db_type":"split-merge", "query":f"[GDB]{gdb_r['query']}"}
        # Both succeeded — merge via LLM
        merge_p = (f"Two query results for the same question. Merge into one final factual answer.\n"
                  f"QUESTION: {question}\n\nRDB_RESULT: {str(rdb_r['payload'])[:1500]}\n"
                  f"GDB_RESULT: {str(gdb_r['payload'])[:1500]}\n\nWrite ONE consolidated answer in English.")
        try:
            ma = llm.invoke(merge_p, json_mode=False)
            return {"success":1,"query":f"[RDB]{rdb_r['query']} || [GDB]{gdb_r['query']}",
                    "payload":{"rdb":rdb_r["payload"],"gdb":gdb_r["payload"]},
                    "answer":ma,"db_type":"split-merge","risk":float("nan")}
        except Exception as exc:
            return {**rdb_r, "db_type":"split-merge","error":str(exc)[:200]}

    def hybrid_run(question, cat, diff, qt, evaluator_class=None):
        if evaluator_class is not None:
            evaluator_class.set_ctx(cat, diff, qt)
        ctx = PatchCtx(agent, evaluator_class) if evaluator_class else Noop()
        with ctx:
            agent = ReActHybridAgent(rdb=rdb, neo4j_driver=neo4j_driver, llm=llm,
                                     enable_recovery=True, recovery_policy="auto", debug=False)
            try:
                res = agent.run(question)
            except Exception as exc:
                return {"success":0,"answer":"","db_type":"UNKNOWN","risk":float("nan"),"query":"","error":str(exc)[:300]}
            pl = res.get("execution_result_payload")
            ne = (pl is not None and not (isinstance(pl,(list,dict,str)) and len(pl)==0))
            ma = str(res.get("generated_natural_answer","") or "")
            sel = res.get("selected_plan") or {}
            risk_raw = sel.get("selected_plan_predicted_error_risk")
            return {"success":int(bool(ne and ma)), "answer":ma,
                    "db_type":str(sel.get("selected_plan_db_type","UNKNOWN")),
                    "risk":float(risk_raw) if isinstance(risk_raw,(int,float)) else float("nan"),
                    "query":str(sel.get("selected_plan_query","") or ""), "payload":pl}

    benchmark = pd.read_csv(csv_path)
    print(f"Loaded {len(benchmark)} questions")
    methods = [
        ("RDB-fixed-1query",     lambda q,c,d,qt: direct_query(q,"RDB")),
        ("GDB-fixed-1query",     lambda q,c,d,qt: direct_query(q,"GDB")),
        ("RDB+GDB-split-merge",  lambda q,c,d,qt: split_merge(q)),
        ("Hybrid-RAQA-r_h",      lambda q,c,d,qt: hybrid_run(q,c,d,qt, None)),
        ("Hybrid-RAQA-r_theta",  lambda q,c,d,qt: hybrid_run(q,c,d,qt, RThetaEval)),
        ("Hybrid-RAQA-r_theta-c_theta", lambda q,c,d,qt: hybrid_run(q,c,d,qt, RThetaCThetaEval)),
    ]
    logs = []
    for mname, runner in methods:
        print(f"\n----- {mname} -----")
        for i, row in benchmark.iterrows():
            qid = int(row["question_id"]); q = str(row["natural_question"])
            gold = str(row["natural_answer"]); cat = str(row.get("category","")); diff = str(row.get("difficulty",""))
            qt = str(row.get("query_type",""))
            t0 = time.time()
            r = runner(q, cat, diff, qt)
            el = time.time() - t0
            ok = int(r.get("success",0))
            ma = str(r.get("answer","") or "")
            correct = judge(q, gold, ma) if ok and ma else 0
            rbw = int(ok==1 and correct==0)
            logs.append({"dataset":ds,"question_id":qid,"method":mname,"category":cat,"difficulty":diff,
                "execution_success":ok,"answer_correct":correct,"run_but_wrong":rbw,
                "selected_plan_db_type":r.get("db_type",""),
                "selected_plan_predicted_risk":r.get("risk",float("nan")),
                "generated_query":str(r.get("query",""))[:300],"model_answer":ma[:200],
                "latency_sec":el,"error":r.get("error","")})
            print(f"  [{i+1:3d}/{len(benchmark)}] qid={qid} exec={ok} correct={correct} rbw={rbw} db={r.get('db_type','')} ({el:.1f}s)")

    df = pd.DataFrame(logs)
    out = OUT_DIR / f"{ds}_6methods_logs.csv"
    df.to_csv(out, index=False)
    summary = df.groupby("method").agg(
        n=("question_id","count"), success=("execution_success","mean"),
        accuracy=("answer_correct","mean"), rbw_rate=("run_but_wrong","mean"),
        avg_latency=("latency_sec","mean")).round(4)
    summary["rbw_er"] = (df.groupby("method").apply(
        lambda g: g["run_but_wrong"].sum()/max(g["execution_success"].sum(),1)).round(4))
    summary.to_csv(OUT_DIR / f"{ds}_6methods_summary.csv")
    print(f"\n===== {ds} Summary =====\n{summary}\nLogs: {out}")


def main():
    run_dataset("mimic", MIMIC_CSV, MIMIC_AGENT, set_mimic_env)


if __name__ == "__main__":
    main()
