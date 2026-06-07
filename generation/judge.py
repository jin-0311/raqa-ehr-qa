"""
LLM judge for EHR-QA benchmark items.

For each row, ask gpt-4.1-mini to evaluate:
  - clinical_meaningfulness (1-5): would a physician actually ask this?
  - query_correctness (1-5): does the SQL correctly answer the question?
  - answer_factuality (1-5): does the natural_answer match what the SQL returned?
  - diversity (1-5): is this question distinct from common stereotypes?
  - overall_decision: accept / revise / reject
  - llm_comment: 1-2 sentence justification

Input CSV must have columns: natural_question, sql_query, natural_answer, category, query_result.
Output adds: clinical_meaningfulness, query_correctness, answer_factuality, diversity,
             overall_decision, llm_comment.

Usage
-----
  python judge.py <input.csv> <output.csv>
  python judge.py input.csv judged.csv
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import config
if config.OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

SYS = ("You are a clinical informatics reviewer evaluating EHR-QA benchmark items. "
       "BIAS TOWARD ACCEPT — these items are for a research benchmark, not a textbook. "
       "Accept any item that is clinically plausible, has a working SQL, and a natural answer "
       "that matches the SQL result. Mark REVISE only for items with a material quality issue "
       "that would mislead a reader. REJECT only for genuinely broken items (unanswerable, "
       "wrong SQL that does not match the question, fabricated answer).")

STEREOTYPES = ("top 5 diagnoses", "top 3 medications", "total count of medications", "how many in total")


def make_prompt(row):
    return f"""Evaluate this EHR-QA benchmark item.

Category: {row.get('category')}
Question: {row.get('natural_question')}
SQL: {str(row.get('sql_query',''))[:1000]}
SQL_RESULT (first rows): {str(row.get('query_result',''))[:1500]}
Natural_answer: {row.get('natural_answer')}

Score each on 1-5 (5 = best) and give an overall decision:
- clinical_meaningfulness: would a real physician ask this for patient care?
- query_correctness: does the SQL truly answer the question?
- answer_factuality: does natural_answer match the SQL_RESULT?
- diversity: is this question distinct from these stereotyped patterns: {list(STEREOTYPES)}?

Decision rule (BIAS TOWARD ACCEPT for borderline cases):
- ACCEPT if the question is clinically plausible, the SQL appears to answer it,
  and the natural_answer matches the SQL_RESULT — even if the question could be
  more elegant or specific. "Common but useful" is fine to accept.
- REVISE only for material quality issues that would mislead a reader.
- REJECT only for genuinely broken items: unanswerable, clearly wrong SQL,
  fabricated answer that does NOT match SQL_RESULT.

Return JSON only:
{{"clinical_meaningfulness": <1-5>, "query_correctness": <1-5>,
  "answer_factuality": <1-5>, "diversity": <1-5>,
  "overall_decision": "accept"|"revise"|"reject",
  "llm_comment": "<1-2 sentence justification>"}}"""


def judge_one(row, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            r = client.chat.completions.create(
                model="gpt-4.1-mini", temperature=0.1, max_tokens=400,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": make_prompt(row)}])
            return json.loads(r.choices[0].message.content)
        except Exception as exc:
            if attempt == max_retries:
                return {"clinical_meaningfulness": None, "query_correctness": None,
                        "answer_factuality": None, "diversity": None,
                        "overall_decision": "error", "llm_comment": str(exc)[:200]}
            time.sleep(2)


def main():
    if len(sys.argv) < 3:
        print("Usage: python judge.py <input.csv> <output.csv>")
        sys.exit(1)
    inp = Path(sys.argv[1]); outp = Path(sys.argv[2])
    df = pd.read_csv(inp)
    print(f"Loaded {len(df)} rows from {inp.name}")

    out_rows = []
    t0 = time.time()
    # incremental save every 25 rows
    for i, row in df.iterrows():
        v = judge_one(row)
        merged = {**row.to_dict(), **v}
        out_rows.append(merged)
        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            pd.DataFrame(out_rows).to_csv(outp, index=False)
            dec = pd.Series([r.get("overall_decision","?") for r in out_rows]).value_counts().to_dict()
            print(f"  [{i+1:4d}/{len(df)}]  {(time.time()-t0)/60:.1f}min  decisions={dec}")

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(outp, index=False)
    print(f"\nSaved → {outp}")
    print("\nDecisions:")
    print(df_out["overall_decision"].value_counts().to_string())
    if "category" in df_out.columns:
        print("\nAccept rate by category:")
        df_out["_accept"] = (df_out["overall_decision"] == "accept").astype(int)
        print(df_out.groupby("category")["_accept"].agg(["mean","count"]).round(3).to_string())
    # by bucket if present
    for buc_col in ["bucket"]:
        if buc_col in df_out.columns:
            print(f"\nAccept rate by {buc_col}:")
            print(df_out.groupby(buc_col)["_accept"].agg(["mean","count","sum"]).round(3).to_string())


if __name__ == "__main__":
    main()
