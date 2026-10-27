"""Check the LLM judge against your own labels on a small sample.

    python -m eval.label_check make --run-id mix20 --n 10      # writes data/labels/human_template.csv (no judge scores shown)
    # fill in `correct` and `faithful` with yes / no, save as data/labels/human.csv
    python -m eval.label_check compare --run-id mix20          # how often does the judge agree with you?

correct   = does the response give the gold answer's key facts (or, for refusal / clarify / flag questions, do what the gold says)?
faithful  = is every claim in the response supported by the retrieved context shown? (a refusal counts as faithful)
"""
import argparse
import json

import pandas as pd

from src import config

LABELS = config.ROOT / "data" / "labels"
FAITHFUL_AT = 0.8   # Ragas faithfulness at or above this counts as "faithful"


def make(run_id: str, n: int) -> None:
    golden = {g["id"]: g for g in map(json.loads, (config.ROOT / "data/golden/golden_v1.jsonl").read_text(encoding="utf-8").splitlines())}
    recs = {r["golden_id"]: r for r in map(json.loads, (config.LOG_DIR / "runs" / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines())}
    # one per category first (so rare categories are covered), then fill up with the rest, in a fixed order
    ids, seen = [], set()
    for gid, g in golden.items():
        if g["category"] not in seen:
            seen.add(g["category"]); ids.append(gid)
    ids += [gid for gid in golden if gid not in ids]
    rows = [{"golden_id": gid, "category": golden[gid]["category"], "question": golden[gid]["question"],
             "reference": golden[gid]["reference"], "response": recs[gid]["answer"],
             "retrieved_context": "\n\n----\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(recs[gid]["contexts"])),
             "correct": "", "faithful": "", "notes": ""} for gid in ids[:n]]
    LABELS.mkdir(parents=True, exist_ok=True)
    out = LABELS / "human_template.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows to {out}\nFill `correct` and `faithful` (yes/no), save as {LABELS / 'human.csv'}")


def compare(run_id: str) -> None:
    human = pd.read_csv(LABELS / "human.csv")
    scores = pd.read_csv(config.ROOT / "reports" / f"{run_id}_scores.csv")
    m = human.merge(scores, on="golden_id", suffixes=("_h", ""))
    m["h_correct"] = m["correct"].astype(str).str.strip().str.lower().eq("yes")
    m["h_faithful"] = m["faithful"].astype(str).str.strip().str.lower().eq("yes")
    m["j_correct"] = m["answer_correctness"] >= 0.5
    m["j_faithful"] = m["faithfulness"] >= FAITHFUL_AT
    for name, h, j in [("correct", "h_correct", "j_correct"), ("faithful", "h_faithful", "j_faithful")]:
        d = m.dropna(subset=["answer_correctness" if name == "correct" else "faithfulness"])
        print(f"{name}: judge agrees with you on {(d[h] == d[j]).sum()} of {len(d)} rows")
        for _, r in d[d[h] != d[j]].iterrows():
            print(f"   disagree {r['golden_id']} ({r['category']}): you={'yes' if r[h] else 'no'}, judge={'yes' if r[j] else 'no'}"
                  f"  note: {r.get('notes_h', '') if 'notes_h' in r else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "compare"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    make(a.run_id, a.n) if a.cmd == "make" else compare(a.run_id)
