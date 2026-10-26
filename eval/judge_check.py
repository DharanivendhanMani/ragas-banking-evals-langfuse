"""Phase 7, step 2: how well do the Ragas scores agree with the human labels?

    python -m eval.judge_check --run-id baseline-all
    python -m eval.judge_check --run-id baseline-all --labels-b data/labels/human_v1_b.csv   # adds inter-labeler agreement

Pairs (Ragas metric <-> human label):
    faithfulness      <-> faithful            (yes=1, partial=0.5, no=0)
    answer_relevancy  <-> relevant            (yes=1, partial=0.5, no=0)
    context_recall    <-> context_sufficient  (yes=1, no=0)

Reports Spearman correlation (does the metric rank answers like a human?) and, at a threshold, Cohen's kappa and a
confusion matrix (does it make the same good/bad call?). Writes reports/judge_agreement.md.
"""
import argparse

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src import config

PAIRS = [("faithfulness", "faithful"), ("answer_relevancy", "relevant"), ("context_recall", "context_sufficient")]
MAP = {"yes": 1.0, "partial": 0.5, "no": 0.0}
THRESHOLD = 0.8   # judge score at/above this = "good"; human 'yes' (1.0) = "good"


def load_labels(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("faithful", "relevant", "context_sufficient"):
        df[col] = df[col].astype(str).str.strip().str.lower().map(MAP)
    return df


def interpret(kappa: float) -> str:
    if kappa != kappa:
        return "n/a"
    return "poor" if kappa < 0.2 else "fair" if kappa < 0.4 else "moderate" if kappa < 0.6 else "substantial" if kappa < 0.8 else "strong"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--labels", default=str(config.ROOT / "data" / "labels" / "human_v1.csv"))
    ap.add_argument("--labels-b", help="optional second labeler's file for inter-labeler agreement")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    a = ap.parse_args()

    scores = pd.read_csv(config.ROOT / "reports" / f"{a.run_id}_scores.csv")
    human = load_labels(a.labels)
    merged = human.merge(scores, on="golden_id", suffixes=("_h", ""))
    lines = [f"# Judge agreement: run `{a.run_id}`", "",
             f"Labeled rows merged with scores: **{len(merged)}** (threshold for 'good' = {a.threshold}). "
             "Small sample: treat as indicative, not precise.", "",
             "| Ragas metric | Human label | n | Spearman | Cohen's kappa | Agreement reading |", "|---|---|---|---|---|---|"]
    details = []
    for metric, label in PAIRS:
        d = merged[[metric, label]].dropna()
        if len(d) < 5:
            lines.append(f"| {metric} | {label} | {len(d)} | n/a | n/a | too few labeled rows |")
            continue
        rho, p = spearmanr(d[metric], d[label])
        judge_good = (d[metric] >= a.threshold).astype(int)
        human_good = (d[label] >= 1.0).astype(int)
        kappa = cohen_kappa_score(human_good, judge_good)
        cm = confusion_matrix(human_good, judge_good, labels=[0, 1])
        lines.append(f"| {metric} | {label} | {len(d)} | {rho:.2f} (p={p:.3f}) | {kappa:.2f} | {interpret(kappa)} |")
        details.append(f"**{metric}**: rows = human (bad, good), columns = judge (bad, good)\n\n"
                       f"|  | judge bad | judge good |\n|---|---|---|\n| human bad | {cm[0][0]} | {cm[0][1]} |\n| human good | {cm[1][0]} | {cm[1][1]} |")
    lines += ["", "## Confusion matrices", ""] + [d + "\n" for d in details]

    if a.labels_b:
        b = load_labels(a.labels_b)
        both = human.merge(b, on="golden_id", suffixes=("_a", "_b"))
        lines += ["## Inter-labeler agreement (human A vs human B)", "", "| label | n | Cohen's kappa |", "|---|---|---|"]
        for _, label in PAIRS:
            d = both[[f"{label}_a", f"{label}_b"]].dropna()
            if len(d) >= 5:
                lines.append(f"| {label} | {len(d)} | {cohen_kappa_score((d[f'{label}_a'] >= 1).astype(int), (d[f'{label}_b'] >= 1).astype(int)):.2f} |")
        lines.append("")

    low = merged[(merged["faithfulness"] - merged["faithful"]).abs() >= 0.5][["golden_id", "category", "faithful", "faithfulness", "notes"]]
    lines += ["## Largest faithfulness disagreements (|judge - human| >= 0.5)", "",
              low.to_markdown(index=False) if len(low) else "None.", "",
              "## How to read this", "",
              "- Kappa < 0.4 on a metric: do not rely on it for decisions; revise the judge model, prompt, or rubric and re-check.",
              "- Spearman high but kappa low: the metric ranks answers sensibly but the threshold is miscalibrated; pick a different cut-off.",
              "- Always read the disagreement rows: they show whether the judge or the human is wrong."]
    out = config.ROOT / "reports" / "judge_agreement.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
