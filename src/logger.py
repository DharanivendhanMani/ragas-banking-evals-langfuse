"""Append-only JSONL log of every RAG interaction (question, contexts, answer, trace_id, ...).

PII is redacted before anything is written. The trace_id links each row to its Langfuse trace.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.pii import redact_obj

DEFAULT_LOG = config.LOG_DIR / "interactions.jsonl"


def log_interaction(result: dict, path: Path = DEFAULT_LOG) -> None:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **redact_obj(result)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_log(path: Path = DEFAULT_LOG) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
