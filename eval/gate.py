"""CI quality gate: fail (exit 1) if a run's metrics fall below the thresholds in eval/thresholds.json.

    python -m eval.gate --run-id baseline-dev

A threshold applies to the overall mean; "pii_leak" is a maximum (must stay at or below it), the others are minimums.
"""
import argparse
import json
import sys

from src import config

THRESHOLDS = config.ROOT / "eval" / "thresholds.json"
MAXIMUMS = {"pii_leak", "stale_context"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    summary = json.loads((config.ROOT / "reports" / f"{a.run_id}_summary.json").read_text())["overall"]
    thresholds = json.loads(THRESHOLDS.read_text())
    failed = []
    for metric, limit in thresholds.items():
        if metric not in summary:
            failed.append(f"{metric}: not in the run summary")
            continue
        value = summary[metric]["mean"]
        ok = value <= limit if metric in MAXIMUMS else value >= limit
        print(f"{'PASS' if ok else 'FAIL'}  {metric:20s} {value:.3f}  ({'<=' if metric in MAXIMUMS else '>='} {limit})")
        if not ok:
            failed.append(f"{metric}: {value:.3f} vs limit {limit}")
    if failed:
        print("\nGate FAILED:\n  " + "\n  ".join(failed))
        sys.exit(1)
    print("\nGate passed.")


if __name__ == "__main__":
    main()
