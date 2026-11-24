# ADR 0009: Retrieval and Generation Evaluation Separated

## Status
Accepted

## Date
2026-11-24

## Context
A single "quality" score conflates retrieval correctness with generation faithfulness. These are distinct failure modes requiring distinct diagnostics.

## Decision
Evaluation platform (`src/evaluation/`) separates concerns:
- **Retrieval metrics** (`retrieval_metrics.py`): Recall@K, Precision@K, MRR@K, nDCG@K, MAP@K, hit_rate@K, retrieval_empty rate. Document-level, duplicate chunks collapsed.
- **Answer metrics** (`answer_metrics.py`): exact match, contains-reference, token F1 (where reference answers exist).
- **LLM-as-a-Judge** (`judge.py`, `judge_prompts.py`): faithfulness, relevancy, context_precision. Judge metadata recorded: `judge_model`, `prompt_version`, `timestamp`, `usage_source` (provider / estimated / unavailable). Invalid JSON and provider errors rejected.
- **Statistical tests** (`statistical_tests.py`): percentile bootstrap, paired bootstrap, Cohen's d for configuration comparisons.
- **Datasets** (`datasets.py`): 10-category taxonomy (direct, near_neighbor, comparison, keyword, multi_document, multi_hop, ambiguous, no_answer, long_document, conflicting). Split support (`dev`/`test`/`unspecified`). Manifest records dataset/corpus SHA-256, category counts, label provenance.
- **Regression runner schema v3** (`regression_runner.py`): manifest, confidence intervals, per-category metrics, unanswerable stats, lexical answer metrics, judge provenance, baseline comparison details. CLI: `--split`, `--bootstrap-samples`, `--no-bootstrap`.

## Consequences
- ✅ Retrieval regressions caught independently of generation changes.
- ✅ Generation regressions (faithfulness, citation precision) measured separately.
- ✅ Judge calibration report (MAE, RMSE, bias, Pearson) against human-reviewed sample.
- ✅ Baselines immutable: created only with `--accept-baseline`; compared only with explicit `--baseline`; fail closed on mismatch.
- ✅ Synthetic (28 queries, 16 docs) and human-annotated datasets clearly separated.
- ⚠️ Human labels remain unverified; no reviewed held-out benchmark exists.
- ⚠️ Real cloud embeddings / cross-encoder / LLM judge not measured offline.

## Related
- ADR 0003: Dense and Sparse Retrieval Evaluated Separately
- ADR 0008: Citation Validation and Grounded Generation