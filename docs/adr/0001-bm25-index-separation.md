# ADR 0001: BM25 Indexing Separated from Query Path

## Status
Accepted

## Date
2026-11-24

## Context
The original implementation rebuilt the BM25 index on every query by scrolling the entire Qdrant corpus. This is not acceptable for a scalable request path — it adds O(N) latency per request and prevents caching.

## Decision
Build a persistent, versioned BM25 index independent of query execution:
- `src/retrieval/index_manager.py` manages immutable `IndexSnapshot` objects.
- `src/retrieval/retrieval_runtime.py` provides `get_retrieval_documents()` returning active documents for index builds.
- `BM25Retriever` in `src/retrieval/retriever.py` is constructed once per index version and reused.
- Explicit `/index/refresh` endpoint rebuilds the index from the active document set.
- Ingestion stages: build new index → validate → atomically swap → retire old snapshot on success.

## Consequences
- ✅ Repeated queries reuse the same active BM25 index (no full corpus scroll).
- ✅ Concurrent queries remain safe during index replacement (immutable snapshots).
- ✅ Failed index rebuild preserves the previous active snapshot.
- ✅ Index version and corpus fingerprint recorded for reproducibility.
- ⚠️ In-memory store uses volatile registries; Qdrant persistence not yet exercised.
- ⚠️ Single-node in-process index; distributed search backend requires interface abstraction.

## Related
- ADR 0002: Versioned Retrieval Snapshots
- ADR 0004: Reciprocal Rank Fusion for Hybrid Retrieval