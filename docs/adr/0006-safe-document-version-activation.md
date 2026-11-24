# ADR 0006: Safe Document Version Activation

## Status
Accepted

## Date
2026-11-24

## Context
Document updates must not corrupt the active searchable index. A failed update must leave the previous version searchable.

## Decision
Staging/activation workflow in `src/ingestion/ingestion_service.py`:
1. **Stage**: Register new document version (status `staging`), write chunks + vectors to a staging namespace.
2. **Verify**: Validate chunk count, vector integrity, no corruption.
3. **Retire old**: Mark previous active version `retired` (soft delete).
4. **Activate new**: Set new version status `active`, increment global index version.
5. **Registry**: Persist to `INGESTION_REGISTRY_PATH` (JSON).
6. **Rebuild snapshot**: Call `IndexManager.refresh_from_store()` to build new retrieval snapshot from active documents only.
7. **Delete retired**: Remove retired chunks from vector store.
8. **Rollback on failure**: If any step fails after activation, revert status, re-activate old version, rebuild snapshot from old active set, log `invalid_reason`.

## Consequences
- ✅ Failed updates never damage the active version.
- ✅ Idempotent ingestion: repeated upload of identical content (by SHA-256) returns `duplicate=true` without creating new chunks.
- ✅ Stable chunk IDs: `doc_id::chunk_index::content_sha256_prefix` — content changes produce new IDs.
- ✅ Deletion removes all chunks, vectors, sparse entries, registry rows, and invalidates index.
- ⚠️ Legacy Qdrant documents lacking unique chunk IDs require reingestion (documented limitation).
- ⚠️ In-memory store uses volatile registries; Qdrant persistence not yet exercised.

## Related
- ADR 0002: Versioned Retrieval Snapshots
- ADR 0007: Document Identity and Chunk Identity