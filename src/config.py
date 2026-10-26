"""Central configuration: paths, model names, and the two API clients.

Both providers expose OpenAI-compatible endpoints, so the plain OpenAI SDK is used for each.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "data" / "raw"
MANIFEST = RAW_DIR / "manifest.json"
CHROMA_DIR = ROOT / "chroma_db"
LOG_DIR = ROOT / "logs"

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-flash-lite-latest")

# Baseline RAG settings (varied in Phase 8 experiments)
CHUNK_TOKENS = 200
CHUNK_OVERLAP = 30
TOP_K = 4
COLLECTION = "banking_baseline"

PROMPT_VERSION = "baseline-v1"


def groq_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url=os.getenv("GROQ_BASE_URL"))


def google_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url=os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    )
