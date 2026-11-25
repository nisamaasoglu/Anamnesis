"""Single source of truth for every tunable value.

Nothing else in the package hard-codes a path, a threshold or a model name.
Every value can be overridden with an environment variable, so the same code
runs unchanged in a container, in CI and on a laptop.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Identity -------------------------------------------------------------
# Meridian General Hospital is FICTIONAL. Every patient, appointment, bed and
# lab result in this project is synthetic and generated locally from SEED.
HOSPITAL_NAME = "Meridian General Hospital"
HOSPITAL_SHORT = "Meridian General"

# --- Paths ----------------------------------------------------------------
KNOWLEDGE_DIR = Path(os.getenv("ANAMNESIS_KNOWLEDGE_DIR", BASE_DIR / "knowledge_base"))
DATA_DIR = Path(os.getenv("ANAMNESIS_DATA_DIR", BASE_DIR / "data"))
DB_PATH = Path(os.getenv("ANAMNESIS_DB_PATH", DATA_DIR / "meridian.sqlite3"))
INDEX_PATH = Path(os.getenv("ANAMNESIS_INDEX_PATH", DATA_DIR / "kb_index.npz"))

# --- Providers ------------------------------------------------------------
# "offline" is the default so the project runs with no API key and no network.
EMBEDDING_PROVIDER = os.getenv("ANAMNESIS_EMBEDDING_PROVIDER", "offline")
LLM_PROVIDER = os.getenv("ANAMNESIS_LLM_PROVIDER", "offline")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("ANAMNESIS_OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("ANAMNESIS_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

OLLAMA_HOST = os.getenv("ANAMNESIS_OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("ANAMNESIS_OLLAMA_CHAT_MODEL", "llama3.1")

# --- Retrieval ------------------------------------------------------------
EMBEDDING_DIM = int(os.getenv("ANAMNESIS_EMBEDDING_DIM", "768"))
CHUNK_WORDS = int(os.getenv("ANAMNESIS_CHUNK_WORDS", "110"))
CHUNK_OVERLAP_WORDS = int(os.getenv("ANAMNESIS_CHUNK_OVERLAP_WORDS", "30"))
TOP_K = int(os.getenv("ANAMNESIS_TOP_K", "4"))
MIN_SCORE = float(os.getenv("ANAMNESIS_MIN_SCORE", "0.08"))
# Minimum field-weighted BM25 score for a passage to count as evidence offline.
BM25_MIN_SCORE = float(os.getenv("ANAMNESIS_BM25_MIN_SCORE", "2.0"))

# --- Agent ----------------------------------------------------------------
MAX_TOOL_STEPS = int(os.getenv("ANAMNESIS_MAX_TOOL_STEPS", "4"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("ANAMNESIS_REQUEST_TIMEOUT", "30"))

# --- Synthetic data -------------------------------------------------------
SEED = int(os.getenv("ANAMNESIS_SEED", "20260723"))
SYNTHETIC_PATIENTS = int(os.getenv("ANAMNESIS_SYNTHETIC_PATIENTS", "60"))

# --- Server ---------------------------------------------------------------
HOST = os.getenv("ANAMNESIS_HOST", "127.0.0.1")


def resolve_port(env: Mapping[str, str]) -> int:
    """ANAMNESIS_PORT wins; otherwise the hosting platform's PORT; otherwise 8000.

    Container hosts (Vercel, Render, Cloud Run) route traffic to $PORT, so a
    server that ignores it is unreachable even though it starts cleanly.
    """
    return int(env.get("ANAMNESIS_PORT") or env.get("PORT") or "8000")


PORT = resolve_port(os.environ)

SYNTHETIC_DATA_NOTICE = (
    f"{HOSPITAL_NAME} is a fictional institution. All patient, appointment, bed and "
    "laboratory records in this system are synthetic and generated locally. "
    "No real patient data is present, and none should ever be loaded into it."
)
