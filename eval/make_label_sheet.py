"""Phase 7, step 1: draw a stratified sample of scored answers and export a BLIND labeling sheet.

    python -m eval.make_label_sheet --run-id baseline-all --n 40

The sample is stratified by category and spread across the faithfulness range, so the check covers both
good and bad answers. The sheet deliberately contains NO Ragas scores, so the labeler cannot be anchored by them.
Fill in the three label columns, save as data/labels/human_v1.csv, then run eval.judge_check.
"""
import argparse
import json

import numpy as np
import pandas as pd

from src import config

LABELS = config.ROOT / "data" / "labels"
RUBRIC = """# Labeling rubric (blind: do not look at the Ragas scores before labeling)

Label each row from the question, the retrieved context, and the chatbot response only.

**faithful**: is every claim in the response supported by the retrieved context shown?
- yes: every claim is supported. A refusal or "I don't have that information" is faithful.
- partial: some claims are supported, some are not (or are contradicted).
- no: the main claim is unsupported or contradicted by the context.

**relevant**: does the response address the customer's question?
- yes: directly answers it (or correctly declines / asks a clarifying question when that is appropriate).
- partial: partly answers it, or includes a lot of unrelated material.
- no: does not address it.

**context_sufficient**: does the retrieved context contain the information needed to answer (compare with the reference answer)?
- yes / no. For questions that have no answer in the corpus, label yes if the context is not misleading, no if it is.

Judge the response, not your own knowledge of banking. Add a short note when you are unsure.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    scores = pd.read_csv(config.ROOT / "reports" / f"{a.run_id}_scores.csv")
    recs = {}
    for line in (config.LOG_DIR / "runs" / f"{a.run_id}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        recs[r["golden_id"]] = r
    golden = {}
    for line in (config.ROOT / "data" / "golden" / "golden_v1.jsonl").read_text(encoding="utf-8").splitlines():
        g = json.loads(line)
        golden[g["id"]] = g

    # proportional allocation per category (at least 1), then evenly spaced picks along the faithfulness ranking
    sizes = scores.groupby("category").size()
    quota = (sizes / sizes.sum() * a.n).round().clip(lower=1).astype(int)
    picked = []
    for cat, group in scores.groupby("category"):
        group = group.sort_values("faithfulness", na_position="first")
        idx = np.linspace(0, len(group) - 1, min(quota[cat], len(group))).round().astype(int)
        picked.extend(group.iloc[sorted(set(idx))]["golden_id"].tolist())
    rng = np.random.default_rng(a.seed)
    rng.shuffle(picked)   # random order so position does not reveal the score

    rows = []
    for gid in picked:
        rec, g = recs[gid], golden[gid]
        rows.append({
            "golden_id": gid, "category": g["category"], "question": g["question"], "reference": g["reference"],
            "response": rec["answer"],
            "retrieved_context": "\n\n----\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(rec["contexts"])),
            "faithful": "", "relevant": "", "context_sufficient": "", "notes": "",
        })
    LABELS.mkdir(parents=True, exist_ok=True)
    out = LABELS / "human_v1_template.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    (LABELS / "RUBRIC.md").write_text(RUBRIC, encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out}\nrubric: {LABELS / 'RUBRIC.md'}")
    print("Next: fill faithful / relevant / context_sufficient, save as data/labels/human_v1.csv, run `python -m eval.judge_check --run-id", a.run_id + "`")


if __name__ == "__main__":
    main()
