"""Run the RAG over the golden set, score it with Ragas + custom checks, and report.

    python -m eval.run_eval --run-id baseline-dev --split dev
    python -m eval.run_eval --run-id baseline-dev --split dev --reuse        # re-score saved answers only
    python -m eval.run_eval --run-id smoke --limit 5 --no-langfuse-scores

Stages
  1. generate : answer every golden question with the chosen RAGConfig (traced in Langfuse, logged to logs/runs/<run_id>.jsonl)
  2. score    : Ragas faithfulness / answer relevancy / context precision / context recall, a custom answer-correctness
                check, hit-rate (deterministic) and a PII-leak check
  3. report   : reports/<run_id>_scores.csv, reports/<run_id>_summary.json (means, by category),
                and the scores pushed to the Langfuse traces
"""
import argparse
import asyncio
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pandas as pd

from eval.judge import Judge, with_retry
from src import config
from src.logger import log_interaction
from src.pii import redact
from src.rag import BASELINE, RAGConfig, answer, langfuse

GOLDEN = config.ROOT / "data" / "golden" / "golden_v1.jsonl"
RUNS_DIR = config.LOG_DIR / "runs"
REPORTS = config.ROOT / "reports"
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
ALL_METRICS = RAGAS_METRICS + ["answer_correctness", "hit_rate", "pii_leak"]


def load_golden(split: str, limit: int | None, categories: list[str] | None = None) -> list[dict]:
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    if split != "all":
        rows = [r for r in rows if r["split"] == split]
    if categories:
        rows = [r for r in rows if r["category"] in categories]
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------- stage 1: generate
def generate_all(rows: list[dict], cfg: RAGConfig, run_id: str, workers: int, reuse: bool) -> dict[str, dict]:
    path = RUNS_DIR / f"{run_id}.jsonl"
    done: dict[str, dict] = {}
    if path.exists():
        if reuse:  # resume: keep answers already saved for this run id
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                done[rec["golden_id"]] = rec
        else:
            path.unlink()  # fresh run
    todo = [r for r in rows if r["id"] not in done]

    def one(row):
        res = answer(row["question"], cfg, run_id=run_id, env="eval", log=False)
        rec = {"golden_id": row["id"], **res}
        log_interaction(rec, path)  # saved immediately, so an interrupted run can be resumed with --reuse
        return row["id"], rec

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for gid, rec in pool.map(one, todo):
                done[gid] = rec
    return done


# ---------------------------------------------------------------- stage 2: score
def retrieval_metrics(row: dict, rec: dict) -> dict:
    """hit_rate (no LLM): was a source document for the question retrieved?"""
    retrieved_docs = {cid.split("::")[0] for cid in rec["chunk_ids"]}
    return {"hit_rate": (1.0 if retrieved_docs & set(row["source_docs"]) else 0.0) if row["source_docs"] else None}


async def score_row(judge: Judge, row: dict, rec: dict, sem: asyncio.Semaphore) -> dict:
    q, ref, resp, ctx = row["question"], row["reference"], rec["answer"], rec["contexts"]
    scores: dict = {"golden_id": row["id"], "trace_id": rec["trace_id"], "category": row["category"],
                    "difficulty": row["difficulty"], "split": row["split"], "answer": resp}
    scores.update(retrieval_metrics(row, rec))
    scores["pii_leak"] = 1.0 if redact(resp) != resp else 0.0
    errors = {}

    async def run(name, coro_fn, *args, **kwargs):
        async with sem:
            try:
                result = await with_retry(coro_fn, *args, **kwargs)
                scores[name] = float(result.value) if hasattr(result, "value") else float(result)
            except Exception as e:  # keep scoring the other metrics
                scores[name] = float("nan")
                errors[name] = f"{type(e).__name__}: {str(e)[:200]}"

    tasks = [
        run("faithfulness", judge.faithfulness.ascore, user_input=q, response=resp, retrieved_contexts=ctx),
        run("answer_relevancy", judge.answer_relevancy.ascore, user_input=q, response=resp),
        run("context_precision", judge.context_precision.ascore, user_input=q, reference=ref, retrieved_contexts=ctx),
        run("context_recall", judge.context_recall.ascore, user_input=q, retrieved_contexts=ctx, reference=ref),
    ]

    async def correctness():
        verdict = await judge.correctness(q, ref, resp)
        scores["correctness_reason"] = verdict.reason
        return 1.0 if verdict.correct else 0.0
    tasks.append(run("answer_correctness", correctness))
    await asyncio.gather(*tasks)
    scores["errors"] = json.dumps(errors) if errors else ""
    return scores


async def score_all(judge: Judge, rows: list[dict], recs: dict[str, dict], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def tracked(row):
        nonlocal done
        scores = await score_row(judge, row, recs[row["id"]], sem)
        done += 1
        failed = f"  ({len(json.loads(scores['errors']))} metric errors)" if scores["errors"] else ""
        print(f"  scored {done}/{len(rows)}  {row['id']}{failed}", flush=True)
        return scores

    return await asyncio.gather(*(tracked(r) for r in rows))


# ---------------------------------------------------------------- stage 3: report
def summarise(df: pd.DataFrame) -> dict:
    def block(frame):
        out = {}
        for m in ALL_METRICS:
            if m in frame and frame[m].notna().any():
                out[m] = {"mean": round(float(frame[m].mean()), 4), "n": int(frame[m].notna().sum()),
                          "failed": int(frame[m].isna().sum())}
        return out

    return {"overall": block(df), "by_category": {c: block(g) for c, g in df.groupby("category")}}


def push_scores(df: pd.DataFrame) -> None:
    for _, r in df.iterrows():
        for m in ALL_METRICS:
            v = r.get(m)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            langfuse.create_score(trace_id=r["trace_id"], name=m, value=float(v), data_type="NUMERIC",
                                  comment=f"{r['golden_id']} ({r['category']})", environment="eval")
    langfuse.flush()


def print_summary(summary: dict) -> None:
    print("\n=== overall (mean, n) ===")
    for m, s in summary["overall"].items():
        print(f"{m:20s} {s['mean']:.3f}  n={s['n']}" + (f"  ({s['failed']} judge calls failed)" if s["failed"] else ""))
    print("\n=== by category (means) ===")
    cats = pd.DataFrame({c: {m: s["mean"] for m, s in b.items()} for c, b in summary["by_category"].items()}).round(2)
    print(cats.to_string())


def evaluate_run(run_id: str, cfg: RAGConfig, split: str = "dev", limit: int | None = None, reuse: bool = False,
                 workers: int = 4, concurrency: int = 6, judge_model: str = config.JUDGE_MODEL,
                 push: bool = True, categories: list[str] | None = None) -> pd.DataFrame:
    """Importable entry point. Returns the per-question score frame."""
    rows = load_golden(split, limit, categories)
    print(f"[{run_id}] {len(rows)} questions | config={cfg.name} | judge={judge_model}")
    recs = generate_all(rows, cfg, run_id, workers, reuse)
    judge = Judge(judge_model)
    scored = asyncio.run(score_all(judge, rows, recs, concurrency))
    df = pd.DataFrame(scored)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS / f"{run_id}_scores.csv", index=False)
    summary = summarise(df)
    summary["run"] = {"run_id": run_id, "config": cfg.__dict__, "split": split, "judge_model": judge_model, "n": len(df)}
    (REPORTS / f"{run_id}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if push:
        push_scores(df)
    langfuse.flush()
    print_summary(summary)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--reuse", action="store_true", help="re-score answers already saved for this run id")
    ap.add_argument("--k", type=int, default=BASELINE.k)
    ap.add_argument("--collection", default=BASELINE.collection)
    ap.add_argument("--prompt-label", default=BASELINE.prompt_label)
    ap.add_argument("--config-name", default="baseline")
    ap.add_argument("--judge-model", default=config.JUDGE_MODEL)
    ap.add_argument("--workers", type=int, default=4, help="parallel RAG calls (Groq rate limits)")
    ap.add_argument("--concurrency", type=int, default=6, help="parallel judge calls")
    ap.add_argument("--no-langfuse-scores", action="store_true")
    a = ap.parse_args()
    cfg = replace(BASELINE, name=a.config_name, k=a.k, collection=a.collection, prompt_label=a.prompt_label)
    evaluate_run(a.run_id, cfg, a.split, a.limit, a.reuse, a.workers, a.concurrency, a.judge_model, not a.no_langfuse_scores)


if __name__ == "__main__":
    main()
