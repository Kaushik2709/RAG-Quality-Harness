# ADR 0007: Document Identity and Chunk Identity

## Status
Accepted

## Date
2026-11-24

## Context
Document and chunk identifiers must be stable, unique, and survive reingestion. Filenames are not globally unique. Chunking changes must invalidate dependent caches.

## Decision
- **Document ID**: caller-supplied or auto-generated UUID; never filename alone. Used as primary key in ingestion registry.
- **Document version**: monotonically increasing integer per document ID. Stored in `version_state` registry.
- **Chunk ID**: `{doc_id}::c{chunk_index}::{content_sha256_prefix}` (e.g., `orion-retention.txt::c0::ea3e98fa4c149524`).
  - Content SHA-256 prefix (16 chars) ensures content changes produce new chunk IDs.
  - Chunk index ensures ordering stability within a document version.
- **Corpus fingerprint**: SHA-256 of concatenated active chunk IDs + content hashes. Recorded in `IndexSnapshot` and evaluation reports.
- **Evaluation**: document-level metrics collapse duplicate chunks by `doc_id` before scoring (first K unique documents).
- **API**: `POST /ingest` and `POST /ingest/file` accept optional `doc_id`; return `doc_id`, `version`, `chunks_created`, `duplicate`, `content_sha256`, `index_version`.

## Consequences
- ✅ Repeated ingestion of identical content is idempotent (`duplicate=true`).
- ✅ Content changes create new chunk IDs; old chunks retired on version activation.
- ✅ Evaluation metrics use stable IDs; reordering other documents does not change chunk IDs.
- ✅ Corpus fingerprint enables drift detection across runs.
- ⚠️ Legacy Qdrant collections with missing/non-unique chunk IDs need reingestion.
- ⚠️ Duplicate basenames in old collections require explicit document IDs on ingestion.

## Related
- ADR 0002: Versioned Retrieval Snapshots
- ADR 0006: Safe Document Version Activation