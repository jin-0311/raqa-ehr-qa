# Mitigating Run-but-Wrong: Learned Risk-Aware Query Alignment for Relational and Graph EHR QA

Reproducibility code for our ICDM 2026 (Applied Track) paper. The package
reproduces the **MIMIC-IV** pipeline end-to-end: hybrid RDB/GDB database construction,
benchmark generation, the learned RAQA components (`r_θ` risk + `c_θ` cost), and the
six-method ReAct experiment.

**Method in one line.** RAQA selects among executable candidate query plans by trading off
estimated cost, predicted error risk, and question–plan alignment; instead of hand-tuning the
cost and risk terms, we *learn* both from past execution logs (`r_θ`: calibrated LightGBM risk;
`c_θ`: LightGBM log-latency cost), reducing Run-but-Wrong (RbW) error.

> **MIMIC-IV-only release.** All code, pretrained models, and data artifacts here reproduce the **MIMIC-IV** pipeline only. The AMC (Asan Medical Center) cohort used in the paper cannot be exported per institutional policy, so it is not part of this repository.
>
> **Data.** MIMIC-IV is credentialed (PhysioNet); raw data is **not** included. Provide your own
> MIMIC-IV sample CSVs under `DATA_DIR`. All hosts/credentials are read from environment
> variables via `config.py` — nothing is hard-coded.

## Layout

```
code_availability/
├── demo_quickstart.ipynb   # ONE-RUN end-to-end demo (synthetic data, no servers needed)
├── config.py              # env-based config (DB URIs, OpenAI key, paths)
├── .env.example           # copy to .env and fill in (or export the vars)
├── requirements.txt
├── db/
│   └── build_mimic_db.py  # MIMIC sample CSVs -> hybrid RDB (MySQL) + GDB (Neo4j)
├── generation/
│   ├── gen_benchmark.py   # 4-bucket (focus × difficulty, C1–C5) generator; SQL+Cypher,
│   │                      #   execution-validated, inline LLM judge, prefix dedup
│   └── judge.py           # standalone LLM-as-judge accept filter
├── models/                # PRETRAINED learned components (joblib)
│   ├── r_theta_lightgbm_isotonic.joblib   # r_θ : calibrated risk predictor
│   └── c_theta_lightgbm.joblib            # c_θ : log-latency cost predictor
├── agents/
│   └── agent.py           # single-file hybrid RDB/GDB ReAct agent (base + RAQA + ReAct)
└── experiment/
    └── run_methods.py     # 6 methods: RDB / GDB / split-merge / Hybrid r_h / r_θ / r_θ+c_θ
```

## Quickstart (no servers, no API key)

Open **`demo_quickstart.ipynb`** and *Run All*: it runs the whole RAQA process end-to-end on a
tiny **synthetic** sample in one go — generates data, builds a lightweight SQLite DB, poses two
slightly complex benchmark questions, and shows the **learned RAQA** (`r_θ`/`c_θ`) picking the
correct plan while the over-joined alternative would have produced a Run-but-Wrong error. No
MySQL/Neo4j/OpenAI required. (For a faithful run on real MIMIC-IV, use the full pipeline below.)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then edit .env  (or export the variables directly)
```

Environment variables (see `.env.example`):

| Variable | Meaning |
|---|---|
| `MIMIC_MYSQL_URI` | `mysql+pymysql://USER:PASS@HOST:3306/mimic` |
| `MIMIC_NEO4J_URI` | `bolt://HOST:7687` |
| `NEO4J_USER`, `NEO4J_PASSWORD` | Neo4j credentials |
| `OPENAI_API_KEY` | OpenAI key (generation + LLM judge + agent backbone) |
| `DATA_DIR` | folder with the MIMIC-IV sample CSVs |

`DATA_DIR` should contain the MIMIC-IV sample tables
(`patient`, `admin`, `diag`, `lab`, `medi`, `op`, `vital`) and the ICD dictionary
`d_icd_diagnoses.csv`.

## Pipeline

```bash
# 1) build the hybrid DB (RDB uppercase tables + GDB unified topology)
python db/build_mimic_db.py

# 2) generate the audited 4-bucket benchmark (-> DATA_DIR/mimic_benchmark.csv)
python generation/gen_benchmark.py          # V7_TARGET=5 python generation/gen_benchmark.py  for a smoke run
python generation/judge.py                  # optional extra accept filter

# 3) run the six-method ReAct experiment (uses the pretrained r_θ / c_θ in models/)
python experiment/run_methods.py mimic
```

Results (`*_logs.csv`, `*_summary.csv`) are written under `ARTIFACT_DIR` (default `./artifacts`).

The learned components (`r_θ`, `c_θ`) are shipped **pretrained** under `models/`. They were
trained on a separate institutional cohort (the cross-institution setting in the paper); that
data is not releasable, so training code is omitted and the released models are applied directly
for reproduction. Generator and agent prompts are condensed for readability.

## Unified hybrid graph topology

```
(Patient)-[:HAS_ADMISSION]->(Admission)-[:NEXT]->(Day)-[:NEXT]->(Day)
(Day{index:0})-[:HAS_DIAG]->(Diagnosis)                      # diagnoses on the first day
(Day)-[:HAS_LAB|HAS_MEDI|HAS_VITAL|HAS_OPERATION]->(event)   # events on their day
```

Reach all days of an admission via `MATCH (d:Day {admissionKey: a.admissionKey})`
(`(a)-[:NEXT]->(:Day)` is the first day only). Diagnoses/operations carry both code and
clinical-name columns, so questions can be asked by code or by clinical name.

## Data availability

- **MIMIC-IV benchmark triples**: shareable under PhysioNet credentialed-access terms.
- **AMC (Asan Medical Center) raw EHR**: not releasable due to institutional policy; a
  schema-compatible synthetic example can be used for re-implementation.

## License

Released under the MIT License (see `LICENSE`).

## Expected results

Main six-method results on the audited benchmark, produced by
`python experiment/run_methods.py mimic` (Accuracy / RbW-ER, %):

| Method | AMC Acc↑ | AMC RbW-ER↓ | MIMIC Acc↑ | MIMIC RbW-ER↓ |
|---|---:|---:|---:|---:|
| RDB-fixed | 59.2 | 34.5 | 47.5 | 32.1 |
| GDB-fixed | 14.4 | 82.9 | 25.0 | 69.2 |
| RDB+GDB split-merge | 39.6 | 57.5 | 38.8 | 56.9 |
| Hybrid + r_h (hand-tuned) | 55.2 | 35.8 | 60.0 | 39.2 |
| Hybrid + r_θ | 53.6 | 35.9 | 56.2 | 40.0 |
| **Hybrid + r_θ + c_θ (ours)** | **59.6** | **30.4** | **63.8** | **35.4** |

Risk calibration (`r_θ` vs. hand-tuned `r_h`): AMC AUC 0.47→0.98, ECE 0.41→0.07;
MIMIC AUC 0.72→0.79, ECE 0.33→0.14.

(Exact values depend on the OpenAI model snapshot and your MIMIC-IV sample.)
