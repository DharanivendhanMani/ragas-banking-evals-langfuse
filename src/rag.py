"""RAG pipeline: embed the question, retrieve top-k chunks from Chroma, generate with Groq.

Every call is one Langfuse trace:

    answer-question (span, root: input = question, output = answer)
    ├── retrieve-context (retriever)
    │   └── embed-query (embedding)
    └── generate-response (generation: model, usage, prompt version, reasoning)

answer() returns the contexts, answer, trace_id and run settings (the shape the logger and Ragas consume).
"""
import re
import threading
import time
from dataclasses import asdict, dataclass

import chromadb
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from langfuse import propagate_attributes

from src import config
from src.logger import log_interaction
from src.prompts import PROMPT_NAME, PROMPTS
from src.tracing import init_langfuse

langfuse = init_langfuse()


@dataclass(frozen=True)
class RAGConfig:
    """RAG settings. The default is the baseline."""
    name: str = "baseline"
    collection: str = config.COLLECTION
    k: int = config.TOP_K
    model: str = config.GROQ_MODEL
    prompt_label: str = "baseline-v1"


BASELINE = RAGConfig()


_chroma_lock = threading.Lock()
_chroma = None
_collections: dict = {}


def _collection(name: str):
    """One shared Chroma client and one handle per collection. Creating a PersistentClient concurrently from
    several threads fails ("Could not connect to tenant"), so creation is serialised with a lock."""
    global _chroma
    with _chroma_lock:
        if _chroma is None:
            _chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        if name not in _collections:
            _collections[name] = _chroma.get_collection(name)
        return _collections[name]


def retrieve(question: str, cfg: RAGConfig) -> list[dict]:
    with langfuse.start_as_current_observation(
        as_type="retriever", name="retrieve-context", input=question,
        metadata={"collection": cfg.collection, "k": str(cfg.k)},
    ) as span:
        with langfuse.start_as_current_observation(
            as_type="embedding", name="embed-query", model=config.EMBEDDING_MODEL, input=question,
        ) as emb:
            resp = config.google_client().embeddings.create(model=config.EMBEDDING_MODEL, input=[question])
            vec = resp.data[0].embedding
            emb.update(output={"dimensions": len(vec)},
                       usage_details={"input": resp.usage.prompt_tokens} if resp.usage else None)
        res = _collection(cfg.collection).query(query_embeddings=[vec], n_results=cfg.k)
        hits = [
            {"id": cid, "text": doc, "metadata": meta, "distance": round(dist, 4)}
            for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
        span.update(output=[
            {"chunk_id": h["id"], "version": h["metadata"]["version"], "status": h["metadata"]["status"],
             "distance": h["distance"], "text": h["text"]} for h in hits
        ])
    return hits


def build_prompt(question: str, hits: list[dict], cfg: RAGConfig) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        header = f"[{i}] {m['title']} (version {m['version']}, effective {m['effective_date'] or 'n/a'}, status: {m['status']})"
        blocks.append(f"{header}\n{h['text']}")
    return "Context:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


def generate(question: str, hits: list[dict], cfg: RAGConfig) -> tuple[str, str]:
    """Returns (answer, reasoning). Recorded as a manual generation so model, usage and reasoning are captured."""
    prompt_client = langfuse.get_prompt(PROMPT_NAME, label=cfg.prompt_label, type="text", fallback=PROMPTS[cfg.prompt_label])
    messages = [
        {"role": "system", "content": prompt_client.prompt},
        {"role": "user", "content": build_prompt(question, hits, cfg)},
    ]
    params = {"temperature": 0, "max_completion_tokens": 1500, "reasoning_effort": "low"}
    with langfuse.start_as_current_observation(
        as_type="generation", name="generate-response", model=cfg.model, input=messages,
        model_parameters=params, prompt=prompt_client,
    ) as gen:
        for attempt in range(8):
            try:
                resp = config.groq_client().chat.completions.create(model=cfg.model, messages=messages, **params)
                break
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
                if attempt == 7:
                    raise
                # rate limit / transient error; Groq's 429 says "try again in 5.7s", so wait at least that long
                hinted = re.search(r"try again in ([\d.]+)s", str(e))
                time.sleep(max(2 ** min(attempt, 4) * 2, float(hinted.group(1)) + 1 if hinted else 0))
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        reasoning = (msg.model_extra or {}).get("reasoning") or ""
        gen.update(
            output={"content": text, "reasoning": reasoning},
            usage_details={"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens},
        )
    return text, reasoning


def answer(question: str, cfg: RAGConfig = BASELINE, *, run_id: str = "adhoc", session_id: str | None = None,
           env: str = "dev", log: bool = True, log_path=None) -> dict:
    """Answer one question. Returns the full record (also appended to the interaction log)."""
    t0 = time.time()
    with langfuse.start_as_current_observation(as_type="span", name="answer-question", input=question) as root:
        with config_attributes(cfg, run_id, session_id, env):
            hits = retrieve(question, cfg)
            text, _ = generate(question, hits, cfg)
        root.update(output=text)
        trace_id = root.trace_id

    result = {
        "run_id": run_id,
        "config": cfg.name,
        "question": question,
        "answer": text,
        "contexts": [h["text"] for h in hits],
        "chunk_ids": [h["id"] for h in hits],
        "distances": [h["distance"] for h in hits],
        "model": cfg.model,
        "prompt_version": cfg.prompt_label,
        "latency_s": round(time.time() - t0, 2),
        "trace_id": trace_id,
        "settings": asdict(cfg),
    }
    if log:
        log_interaction(result, log_path) if log_path else log_interaction(result)
    return result


def config_attributes(cfg: RAGConfig, run_id: str, session_id: str | None, env: str):
    """Trace-level attributes: set once, propagated to every child observation."""
    return propagate_attributes(
        trace_name="answer-question",
        session_id=session_id or run_id,
        environment=env,
        version=cfg.prompt_label,
        tags=["rag", cfg.name],
        metadata={"run_id": run_id, "config": cfg.name, "k": str(cfg.k), "collection": cfg.collection,
                  "model": cfg.model, "prompt_label": cfg.prompt_label},
    )


if __name__ == "__main__":
    import json
    import sys

    q = " ".join(sys.argv[1:]) or "What is the overdraft fee?"
    out = answer(q)
    print(json.dumps({k: v for k, v in out.items() if k not in ("contexts", "settings")}, indent=2))
    langfuse.flush()
