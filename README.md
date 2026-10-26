# Banking RAG: evaluated with Ragas, traced with Langfuse

A learning project. A plain-Python RAG chatbot answers questions about a **synthetic** bank ("Northbridge Bank"),
every request is traced in Langfuse, and quality is measured against a 100-question golden set with Ragas.
Nothing here uses real customer data, and nothing in it is production-ready for a regulated bank.

## Architecture

```
data/raw (14 synthetic docs, 5 deliberately messy)        manifest.json = metadata (version, date, status)
        │  src/ingest.py   chunk (200 tok / 30 overlap) → Gemini embeddings → Chroma
        ▼
question → src/rag.py: embed → retrieve top-k → Groq (gpt-oss-120b) → answer
        │      └─ Langfuse trace: answer-question ▸ retrieve-context ▸ embed-query, generate-response
        ▼
logs/interactions.jsonl  (PII-redacted question / contexts / answer / trace_id)
        ▼
eval/run_eval.py: Ragas (Gemini judge) + custom checks → reports/*.csv|json → scores pushed to the Langfuse traces
```

| Role | Provider / model |
|---|---|
| Chatbot generation | Groq `openai/gpt-oss-120b` (OpenAI-compatible API) |
| Embeddings | Google `gemini-embedding-001` (OpenAI-compatible API) |
| Ragas judge | Google `gemini-flash-lite-latest` (a different model family from the generator, which limits self-preference bias) |
| Vector store | Chroma (local, `chroma_db/`) |
| Tracing, prompts, scores, dataset | Langfuse Cloud (US region) |

## Layout

```
src/        config, pii redaction, tracing (Langfuse + masking), ingest, rag, logger, prompts
eval/       judge, run_eval, make_label_sheet, judge_check, experiments, gate, stats, thresholds.json
scripts/    build_pdfs, build_golden, upload_golden, sync_prompts, thin_slice
data/raw/   corpus + MESSINESS_LOG.md (answer key for the planted flaws; not indexed)
data/golden golden_v1.jsonl (100 rows)         data/labels  human labels (you create these)
reports/    generated results                  logs/        interaction logs, per-run answers
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in the keys; check LANGFUSE_BASE_URL matches your Langfuse region
```

## What each phase produced

| Phase | Deliverable | Where |
|---|---|---|
| 1 Corpus | 8 clean + 6 messy files (table-heavy PDF, conflicting policy versions, internal contradiction, layout-noise PDF, near-duplicate FAQ) | `data/raw/`, `MESSINESS_LOG.md` |
| 2 RAG baseline | ingest + retrieve + generate | `src/ingest.py`, `src/rag.py` |
| 3 Tracing | one trace per request, nested retriever/embedding/generation, prompt linked to Prompt Management, reasoning captured, PII masked at export | `src/rag.py`, `src/tracing.py`, `scripts/sync_prompts.py` |
| 4 Logging | JSONL per interaction, redacted, linked by `trace_id` | `src/logger.py` |
| 5 Golden set | 100 questions, evidence quotes machine-checked against the corpus, 70 dev / 30 test | `scripts/build_golden.py`, `data/golden/` |
| 6 Evaluation | Ragas faithfulness, answer relevancy, context precision/recall, factual correctness + expected-behaviour, PII-leak, hit-rate/MRR, bootstrap CIs | `eval/run_eval.py` |
| 7 Judge check | blind labeling sheet + agreement analysis (Spearman, Cohen's kappa) | `eval/make_label_sheet.py`, `eval/judge_check.py` |
| 8 Experiments | 5 single-change experiments with hypotheses and paired comparisons | `eval/experiments.py` |
| 9 Packaging | this README, CI gate, manual GitHub workflow | `eval/gate.py`, `.github/workflows/eval.yml` |

## Validation checklist (run in this order)

Phases 5 to 9 were written but **not yet executed**. Expect to fix small issues; the likeliest trouble spots are listed below.

```bash
python -m src.ingest                                  # builds banking_baseline (48 chunks); already done once
python -m scripts.sync_prompts                        # prompt versions to Langfuse (already done once)
python -m scripts.build_golden                        # rebuilds golden_v1.jsonl and re-checks every evidence quote
python -m scripts.upload_golden                       # dataset 'banking-rag-golden-v1' in Langfuse

python -m eval.run_eval --run-id smoke --limit 5 --no-langfuse-scores    # 1. smoke test the Ragas + Gemini setup
python -m eval.run_eval --run-id baseline-all --split all                # 2. baseline on all 100 (needed for labeling)

python -m eval.make_label_sheet --run-id baseline-all --n 40             # 3. blind sheet → fill it in by hand
#    save the filled sheet as data/labels/human_v1.csv  (rubric: data/labels/RUBRIC.md)
python -m eval.judge_check --run-id baseline-all                         # 4. do the Ragas scores agree with you?

python -m eval.experiments --split dev                                   # 5. E0..E5 on dev; table in reports/experiments_dev.md
python -m eval.experiments --split test --only E0_baseline,E3_conflict_prompt   # 6. confirm the winner on held-out

python -m eval.gate --run-id baseline-all                                # 7. optional quality gate
```

### Likely trouble spots

- **Judge + Gemini**: Ragas runs through `llm_factory(provider="openai")` with the Gemini OpenAI-compatible endpoint (JSON mode).
  If structured output fails, check `errors` in `reports/<run>_scores.csv`; the first fix to try is a different `JUDGE_MODEL`
  (for example `gemini-2.5-flash`).
- **Rate limits**: lower `--workers` (Groq) and `--concurrency` (Gemini); failed metric calls are retried and otherwise recorded as NaN.
- **Run cost/time**: 100 rows × ~6 judge calls (several LLM calls each). Use `--limit` and `--split dev` while iterating.
- **Gemini embeddings**: `gemini-embedding-001` through the OpenAI-compatible endpoint; the index must be rebuilt if you change the model.
- **Langfuse cost column is empty** for the Groq model because it is not in Langfuse's pricing table. Add a custom model price
  in Langfuse (Settings → Models) if you want cost.

## Design decisions worth knowing

- **Plain Python, no RAG framework**: every step the metrics measure is visible in the trace.
- **Manual generation observation**: Groq returns the model's reasoning separately; recording the generation by hand keeps the reasoning,
  token usage and linked prompt version in the trace.
- **Dev/test split (70/30)**: iterate on dev; use test only to confirm the final choice. 30 questions is thin, so do not choose between close variants on it.
- **Statistics**: with 100 questions, a single metric has a 95% CI of roughly ±0.05 to ±0.08. Experiments report paired per-question differences with CIs.
  Per-category numbers (for example 7 adversarial questions) are anecdotes: read the rows.
- **Custom expected-behaviour check** (LLM judge) for unanswerable, adversarial, ambiguous and "flag the inconsistency" questions, where
  faithfulness and relevancy are not the right measures. **PII-leak** is a deterministic regex check on the response.
- **Gold answers were written by the project author**, with machine-checked evidence quotes, but they are still unreviewed by a second person.

## Known limitations

- Synthetic corpus, simplified regulation (Reg E / Reg CC paraphrased), single language, single turn, no authentication or account lookup.
- The LLM judge is unproven until Phase 7 is done with real human labels.
- No multi-turn conversations, guardrail layer, reranker or production monitoring.

## Results

*(Fill in after running the checklist: baseline numbers with CIs, judge-agreement summary, experiment table, top failure modes, and surprises.)*
