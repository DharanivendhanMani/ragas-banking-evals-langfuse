"""Phase 8: run controlled experiments, one change at a time, and compare them with the baseline.

    python -m eval.experiments --split dev                       # all experiments on the dev set
    python -m eval.experiments --split dev --only E1_k6,E3_conflict_prompt
    python -m eval.experiments --split test --only E0_baseline,E3_conflict_prompt   # confirm the winner on held-out

Each experiment changes exactly one variable relative to E0_baseline and states its hypothesis up front.
Results go to reports/experiments_<split>.md: mean per metric, the paired per-question difference vs the baseline
with a 95% bootstrap CI (an interval that excludes 0 is evidence of a real change), latency, and token usage.
Fill in the 'Verdict' column yourself after reading the worst answers of each run.
"""
import argparse
import json
from dataclasses import replace

import pandas as pd

from eval.run_eval import REPORTS, RUNS_DIR, evaluate_run
from eval.stats import paired_diff_ci
from src import config
from src.ingest import build_index, collection_exists
from src.rag import BASELINE

# id -> (config, (chunk_tokens, overlap) if it needs its own index, hypothesis)
EXPERIMENTS = {
    "E0_baseline": (BASELINE, None, "Reference: 200-token chunks, k=4, dense retrieval, baseline-v1 prompt, version headers shown."),
    "E1_k6": (replace(BASELINE, name="k6", k=6), None,
              "k=6 raises context recall (multi-chunk and table questions) but lowers context precision and may lower faithfulness."),
    "E2_chunk400": (replace(BASELINE, name="chunk400", collection="banking_c400"), (400, 60),
                    "Larger chunks keep table rows and multi-part answers together: recall and table/multi-chunk scores up, precision down."),
    "E3_conflict_prompt": (replace(BASELINE, name="conflict_prompt", prompt_label="conflict-aware-v2"), None,
                           "An explicit prompt rule about versions, contradictions and vague questions improves version_conflict, "
                           "ambiguous and unanswerable behaviour without hurting the rest."),
    "E4_hybrid": (replace(BASELINE, name="hybrid", retrieval="hybrid"), None,
                  "BM25 + dense fusion helps exact-term queries (fee names, phone numbers, section keywords): hit_rate and recall up."),
    "E5_no_version_headers": (replace(BASELINE, name="no_version_headers", show_version_metadata=False), None,
                              "Without version/status headers the model cannot tell current from superseded ATM policy: more stale answers "
                              "(answer_correctness down on version_conflict)."),
    "E6_gpt_oss_20b": (replace(BASELINE, name="gpt_oss_20b", model="openai/gpt-oss-20b"), None,
                       "The 20b generator is cheaper and faster but less reliable on conflicts, tables and refusals: "
                       "faithfulness and expected_behavior drop versus the 120b baseline."),
}
KEY_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness",
               "expected_behavior", "hit_rate", "evidence_recall"]


def run_stats(run_id: str) -> dict:
    """Median end-to-end latency for a run, from its saved log (token usage and cost live in Langfuse)."""
    recs = [json.loads(l) for l in (RUNS_DIR / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    lat = sorted(r["latency_s"] for r in recs)
    return {"median_latency_s": lat[len(lat) // 2] if lat else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--only", help="comma-separated experiment ids (E0_baseline is always run first)")
    ap.add_argument("--reuse", action="store_true", help="re-score saved answers instead of regenerating")
    ap.add_argument("--judge-model", default=config.JUDGE_MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()

    chosen = [e.strip() for e in a.only.split(",")] if a.only else list(EXPERIMENTS)
    if "E0_baseline" not in chosen:
        chosen.insert(0, "E0_baseline")
    frames, rows = {}, []
    for exp_id in chosen:
        cfg, index, hypothesis = EXPERIMENTS[exp_id]
        if index and not collection_exists(cfg.collection):
            print(f"building index '{cfg.collection}' (chunk={index[0]}, overlap={index[1]}) ...")
            build_index(index[0], index[1], cfg.collection)
        run_id = f"{exp_id}-{a.split}"
        frames[exp_id] = evaluate_run(run_id, cfg, a.split, reuse=a.reuse, workers=a.workers,
                                      concurrency=a.concurrency, judge_model=a.judge_model)
        base = frames["E0_baseline"]
        row = {"experiment": exp_id, "hypothesis": hypothesis, **run_stats(run_id)}
        for m in KEY_METRICS:
            if m in frames[exp_id] and frames[exp_id][m].notna().any():
                row[m] = round(float(frames[exp_id][m].mean()), 3)
                if exp_id != "E0_baseline":
                    d, lo, hi = paired_diff_ci(base.set_index("golden_id")[m].reindex(frames[exp_id]["golden_id"]).tolist(),
                                               frames[exp_id][m].tolist())
                    row[f"{m} Δ vs base [95% CI]"] = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        row["Verdict"] = ""
        rows.append(row)

    df = pd.DataFrame(rows)
    out = REPORTS / f"experiments_{a.split}.md"
    out.write_text(
        f"# Experiments on the `{a.split}` split (judge: {a.judge_model})\n\n"
        "Δ is the mean per-question difference vs E0_baseline (paired); a CI that excludes 0 suggests a real change. "
        "The golden set is small, so confirm any winner on the held-out `test` split and read the worst answers before deciding.\n\n"
        + df.to_markdown(index=False) + "\n", encoding="utf-8")
    print(df.drop(columns=["hypothesis", "Verdict"]).to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
