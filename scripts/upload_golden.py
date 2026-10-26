"""Upload data/golden/golden_v1.jsonl to Langfuse as the dataset 'banking-rag-golden-v1'.

Item ids are deterministic (golden-<id>), so re-running updates items instead of duplicating them.
Schema: input = {question}; expected_output = {answer: reference}; metadata holds category, split, sources, evidence.
"""
import json

from src import config
from src.tracing import init_langfuse

DATASET = "banking-rag-golden-v1"

if __name__ == "__main__":
    lf = init_langfuse()
    rows = [json.loads(l) for l in (config.ROOT / "data" / "golden" / "golden_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    lf.create_dataset(
        name=DATASET,
        description="100 synthetic Northbridge Bank questions: 40 single-fact, 15 multi-chunk, 12 table/numeric, "
                    "12 version/conflict, 10 unanswerable, 7 adversarial, 4 ambiguous. 70 dev / 30 test.",
        metadata={"version": "v1", "corpus": "synthetic Northbridge Bank", "frozen": "true"},
    )
    for r in rows:
        lf.create_dataset_item(
            dataset_name=DATASET,
            id=f"golden-{r['id']}",
            input={"question": r["question"]},
            expected_output={"answer": r["reference"]},
            metadata={k: r[k] for k in ("category", "difficulty", "split", "source_docs", "distractor_docs", "evidence")},
        )
    lf.flush()
    print(f"uploaded {len(rows)} items to dataset '{DATASET}'")
