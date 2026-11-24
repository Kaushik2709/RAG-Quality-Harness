# ADR 0012: Tenant-Isolated Sparse Retrieval and Lazy BM25 Initialization

## Status
Accepted

## Date
2026-11-24

## Context
In Iteration 1, the BM25 sparse snapshot held documents across all tenants. Even though results were filtered post-scoring, BM25 inverse document frequency (IDF) and average document length calculations were derived from a combined multi-tenant corpus. This introduced cross-tenant score influence and information leakage. Furthermore, `RetrievalRuntime.search()` eagerly triggered BM25 index initialization even for dense-only queries.

## Decision
Implement tenant-isolated sparse indexing and lazy initialization:
- `IndexManager` maintains separate, immutable `IndexSnapshot` instances per tenant (`_tenant_snapshots: Dict[str, IndexSnapshot]`).
- Corpus fingerprints are tenant-scoped: `corpus_fingerprint(documents, tenant_id=...)` hashes `f"{tenant_id}::{doc.page_content}"`.
- BM25 corpus statistics (term frequency, document frequency, average document length) are calculated strictly from the tenant's own active documents.
- Sparse indexes are initialized lazily: dense-only retrieval queries never instantiate or rebuild BM25 snapshots.
- Tenant index rebuilds, document deletions, and version updates are scoped exclusively to the affected tenant without invalidating or locking other tenants' indexes.

## Consequences
- ✅ Mathematically isolated BM25 scoring; zero cross-tenant ranking bias or statistical leakage.
- ✅ Dense-only requests execute with minimal overhead and zero sparse memory footprint.
- ✅ Document updates in Tenant A do not trigger re-indexing in Tenant B.
- ✅ Concurrency safety across tenants with immutable snapshots.
- ⚠️ Separate BM25 indices increase total memory proportional to active tenant count; mitigated by LRU eviction in `IndexManager`.

## Related
- ADR 0001: BM25 Indexing Separated from Query Path
- ADR 0002: Versioned Retrieval Snapshots
- ADR 0011: Tenant-Scoped Cryptographic Credentials and RBAC
