# ADR 0014: Transactional Document Lifecycle and Crash Recovery

## Status
Accepted

## Date
2026-11-24

## Context
Iteration 1 lacked transactional atomicity during document ingestion. If a failure or process crash occurred between embedding, staging, indexing, or activation, orphaned vector embeddings remained in the vector store and document records became inconsistent. Furthermore, document deletion performed a full collection scroll on the vector store, causing O(N) latency.

## Decision
Introduce a SQL-backed document registry and a 9-state transactional state machine:
- Persistent schema via migration `001_document_lifecycle.sql` supporting PostgreSQL and SQLite (WAL mode, foreign keys enforced).
- 9-state machine: `RECEIVED -> VALIDATING -> EXTRACTING -> CHUNKING -> EMBEDDING -> STAGING -> VERIFYING -> ACTIVATING -> ACTIVE`, with explicit failure states `ROLLBACK_REQUIRED`, `FAILED`, and `CANCELLED`.
- Staged vectors are written with status `staged`. Point existence is verified against the vector store (`chunks_present`). If activation fails, staging vectors are rolled back and the previous active version is preserved.
- Document chunks track full provenance: `tenant_id`, `doc_id`, `version`, `page`, `section_path`, `point_id`.
- Document deletion queries registry chunk IDs directly, eliminating full vector store scans and executing in O(k) chunk deletions.
- Automated crash recovery (`recover_and_cleanup()`): scans for stuck ingestion jobs and orphaned staging vectors after configurable timeouts and marks them failed with resource reclamation.

## Consequences
- ✅ Zero orphaned staging vectors on ingestion crashes or exceptions.
- ✅ Atomic, rollback-safe document version transitions.
- ✅ O(k) deletion performance replacing full-collection scans.
- ✅ Granular document and chunk lineage with PDF page provenance.
- ⚠️ Requires SQLite WAL or PostgreSQL persistent database connectivity.

## Related
- ADR 0006: Safe Document Version Activation
- ADR 0007: Document Identity and Chunk Identity
