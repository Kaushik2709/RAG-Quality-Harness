# ADR 0003: Dense and Sparse Retrieval Evaluated Separately

## Status
Accepted

## Date
2026-11-24

## Context
Hybrid retrieval combines dense and sparse signals. To understand the contribution of each component, they must be evaluated independently before fusion. Mixing them in a single "hybrid" metric obscures whether dense or sparse is driving quality.

## Decision
Evaluation runs three distinct configurations via `src/evaluation/regression_runner`:
- `--mode dense`: cosine similarity over embeddings only.
- `--mode bm25`: BM25Okapi sparse retrieval only.
- `--mode hybrid`: weighted RRF (0.6 dense, 0.4 BM25, constant 60) over both candidate sets.
- All three use the exact same corpus/query files, isolated in-memory store, deterministic SHA-256 lexical hash embeddings (offline), candidate K=10, evaluation K=5.
- Reranking (`--mode hybrid-rerank`) is a fourth configuration requiring real cross-encoder.

## Consequences
- ✅ Clear attribution of quality to dense vs. sparse vs. fusion.
- ✅ BM25 fallback (lexical overlap when all BM25 scores ≤ 0) is tested independently.
- ✅ Regression gates apply to each configuration separately (Recall@5 ≥ 0.90, MRR ≥ 0.85, nDCG@5 ≥ 0.85).
- ✅ Per-query ranked document IDs and misses recorded for each mode.
- ⚠️ Offline hash embeddings are a deterministic lexical fixture, not a semantic benchmark.
- ⚠️ Real pretrained embeddings / cross-encoder quality not measured offline.

## Related
- ADR 0004: Reciprocal Rank Fusion for Hybrid Retrieval
- ADR 0005: Reranker Lifecycle is Application-Managed