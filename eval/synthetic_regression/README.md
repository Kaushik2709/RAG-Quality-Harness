# Synthetic Regression Evaluation

This directory documents the **synthetic software-policy regression suite** used for deterministic CI quality gates.

## Purpose

Test **implementation behavior**, not semantic generalization. These are agent-authored controlled regression cases, not human-verified ground truth or real user questions.

## Contents

- `eval/fixtures/queries.jsonl` — 28 synthetic queries over 16 fixture documents (12 fictional policy documents + 4 distractors).
- `eval/fixtures/diagnostic_queries.jsonl` — 10-category diagnostic suite (direct, near_neighbor, comparison, keyword, multi_document, multi_hop, ambiguous, no_answer, long_document, conflicting) including one deliberate no-answer query.
- `eval/fixtures/corpus/` — 16 `.txt` policy documents plus distractors.

## Provenance

Labels are **inherited / synthetic**, not manually verified ground truth. The name "golden" is historical; source text supports many answers but this is not independent human validation.

All 16 documents are indexed in each configuration; none are synthesized during the run. Document-level K=5 evaluation; repeated chunks of the same document collapse before scoring.

## Metrics

Recall@K, Precision@K, MRR@K, nDCG@K, MAP@K, hit_rate@K, retrieval_empty rate, plus lexical answer metrics (exact match, contains-reference, token F1) where reference answers exist.

## Regression gates

Offline CI requires, for each configuration:
- Recall@5 >= 0.90
- MRR >= 0.85
- nDCG@5 >= 0.85

These are explicit initial acceptance floors for this controlled fixture, not statistical significance claims.

Baseline comparisons are opt-in, fail closed, and reject mismatched dataset/corpus/configuration/dependency versions.

## Reproduction

Use Python 3.14 with `requirements-offline.txt`:

```bash
python -m pytest -q
python -m src.evaluation.regression_runner \
  --dataset eval/fixtures/queries.jsonl --corpus eval/fixtures/corpus \
  --mode dense --min-recall 0.9
python -m src.evaluation.regression_runner \
  --dataset eval/fixtures/queries.jsonl --corpus eval/fixtures/corpus \
  --mode bm25 --min-recall 0.9
python -m src.evaluation.regression_runner \
  --dataset eval/fixtures/queries.jsonl --corpus eval/fixtures/corpus \
  --mode hybrid --min-recall 0.9
python -m src.evaluation.regression_runner \
  --dataset eval/fixtures/diagnostic_queries.jsonl --corpus eval/fixtures/corpus \
  --mode hybrid --min-recall 0.9
```

Optional baseline workflow:

```bash
python -m src.evaluation.regression_runner --accept-baseline --baseline eval/baseline_metrics.json
python -m src.evaluation.regression_runner --baseline eval/baseline_metrics.json
```

## Limitations

- Single-process, local wall-clock timings; not production capacity claims.
- Offline hash embeddings are a deterministic lexical fixture, not a semantic benchmark.
- Real pretrained embeddings / cross-encoder quality not measured offline.
- No human-verified held-out set.
