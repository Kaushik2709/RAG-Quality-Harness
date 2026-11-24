# ADR 0004: Reciprocal Rank Fusion for Hybrid Retrieval

## Status
Accepted

## Date
2026-11-24

## Context
Hybrid retrieval needs to combine ranked lists from dense and sparse retrievers. Options include score normalization + weighted sum, Reciprocal Rank Fusion (RRF), and learned fusion. The choice must be deterministic, parameterizable, and auditable.

## Decision
Use weighted Reciprocal Rank Fusion (RRF) as implemented in `src/retrieval/fusion.py`:
- Formula: `score = w_dense / (k + r_dense) + w_sparse / (k + r_sparse)`
- Default weights: 0.6 dense, 0.4 BM25; RRF constant k=60.
- Deduplication by chunk ID (stable per-document chunk IDs from ingestion).
- Candidate list bounded to `candidate_k` (default 10) before fusion.
- Evaluation K=5 applied after fusion and deduplication.

## Consequences
- ✅ Deterministic, no model training required, works with any ranking sources.
- ✅ Weights and constant are configurable and recorded in evaluation reports.
- ✅ No score calibration needed (ranks only).
- ⚠️ RRF assumes comparable rank distributions; dense and BM25 rank scales may differ.
- ⚠️ Weight tuning is manual; no automated optimization implemented.
- ⚠️ Hybrid did not outperform BM25 in nDCG on the synthetic fixture (see EVALUATION_REPORT.md).

## Related
- ADR 0001: BM25 Indexing Separated from Query Path
- ADR 0003: Dense and Sparse Retrieval Evaluated Separately