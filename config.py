"""
Central configuration for the RAQA code release (MIMIC-IV).

All credentials / hosts / paths are read from environment variables so that no
secrets are committed. Set them before running, e.g.:

    export MIMIC_MYSQL_URI="mysql+pymysql://USER:PASS@HOST:3306/mimic"
    export MIMIC_NEO4J_URI="bolt://HOST:7687"
    export NEO4J_USER="neo4j"
    export NEO4J_PASSWORD="..."
    export OPENAI_API_KEY="sk-..."
    export DATA_DIR="/path/to/mimic_sample"   # MIMIC sample CSVs + ICD dicts

MIMIC-IV is credentialed (PhysioNet) — raw data is NOT included. Place the
required sample CSVs under DATA_DIR (see README).
"""
import os
import pathlib

# --- relational + graph DB (MIMIC-IV) ---
MYSQL_URI = os.getenv("MIMIC_MYSQL_URI", "mysql+pymysql://USER:PASSWORD@HOST:3306/mimic")
NEO4J_URI = os.getenv("MIMIC_NEO4J_URI", "bolt://HOST:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "PASSWORD")

# --- OpenAI (benchmark generation + LLM judge) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- data / artifact paths ---
BASE = pathlib.Path(__file__).resolve().parent
DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", str(BASE / "data")))
ARTIFACT_DIR = pathlib.Path(os.getenv("ARTIFACT_DIR", str(BASE / "artifacts")))
MODEL_DIR = BASE / "models"   # pretrained r_theta / c_theta
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def pymysql_kwargs():
    """Parse MYSQL_URI into pymysql.connect kwargs (host/port/user/password/db)."""
    import re
    m = re.match(r"mysql\+pymysql://([^:]+):([^@]+)@([^:/]+):?(\d+)?/(\w+)", MYSQL_URI)
    if not m:
        raise ValueError("Set MIMIC_MYSQL_URI as mysql+pymysql://USER:PASS@HOST:PORT/DB")
    user, pw, host, port, db = m.groups()
    return dict(host=host, port=int(port or 3306), user=user, password=pw,
                database=db, charset="utf8mb4")
