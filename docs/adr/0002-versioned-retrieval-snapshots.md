# ADR 0002: Versioned Retrieval Snapshots

## Status
Accepted

## Date
2026-11-24

## Context
The retrieval index must be rebuilt when documents are added, updated, or deleted. Swapping the index atomically without blocking queries is required. The previous active index must remain available if the new build fails.

## Decision
Use immutable, versioned `IndexSnapshot` objects managed by `IndexManager`:
- Each snapshot contains: version, corpus fingerprint, chunk count, built_at, build_ms, list of `Document` objects (immutable copies).
- `IndexManager.refresh_from_store()` builds a new snapshot from the active document set.
- On success, `IndexManager` atomically swaps `_snapshot` reference under a lock.
- On failure, `_invalid_reason` is set and the old snapshot remains active.
- `IndexManager.status()` exposes: status (ready/invalidated/not_built), version, chunk_count, builds, failures, last_error, invalid_reason.
- Ingestion service calls `refresh_from_store()` after activating a new document version.
- Deletion invalidates the index; refresh-on-failure is handled gracefully.

## Consequences
- ✅ No query ever sees a partially built index.
- ✅ Failed rebuilds are observable via `/ready` and `/runtime/status`.
- ✅ Index version increments monotonically; corpus fingerprint enables drift detection.
- ✅ Empty index (all documents deleted) is valid — status `ready`, chunk_count=0.
- ⚠️ In-memory store uses volatile registries; Qdrant not yet validated for atomic snapshots.
- ⚠️ Concurrent production ingestion is not an atomic BM25/dense snapshot (documented limitation).

## Related
- ADR 0001: BM25 Indexing Separated from Query Path
- ADR 0006: Safe Document Version Activation