# ADR 0011: Tenant-Scoped Cryptographic Credentials and RBAC

## Status
Accepted

## Date
2026-11-24

## Context
Iteration 1 relied on a single global API key (`X-API-Key`) where any authenticated caller could select any tenant via the `X-Tenant-ID` header and received all privileges (reader, ingestor, admin). This created critical privilege escalation, impersonation, and cross-tenant data leak risks. In production mode (`APP_ENV=production`), omitting or misconfiguring API keys did not fail closed.

## Decision
Implement a tenant-scoped credential model with cryptographic key hashing and granular Role-Based Access Control (RBAC):
- API keys are generated using cryptographically secure entropy (`secrets.token_urlsafe(32)`), prefixed with `rag_live_`.
- Plaintext API keys are never stored; persistent storage and in-memory indexes store only SHA-256 hashes (`hmac.compare_digest` constant-time verification).
- Each credential is bound to a single immutable `tenant_id`, explicit roles (`reader`, `ingestor`, `admin`), permissions, expiration timestamps, and revocation state.
- Client headers (`X-Tenant-ID`) can never override or widen the authenticated credential's tenant binding. If `X-Tenant-ID` does not match the credential's `tenant_id`, the request fails with HTTP 403 `tenant_mismatch`.
- Granular permissions enforce:
  - `reader`: Query, read authorized documents.
  - `ingestor`: Reader permissions + document ingestion and document modification.
  - `admin`: Ingestor permissions + document deletion, index refresh, credential provisioning, rotation, revocation, and operational runtime telemetry.
- Production mode fails closed (`APP_ENV=production` rejects all requests without valid credentials).
- Credential management endpoints (`/admin/credentials`, `/admin/credentials/{id}/revoke`, `/admin/credentials/{id}/rotate`, `/admin/audit/auth`) are restricted to administrative credentials.

## Consequences
- ✅ Eliminates cross-tenant credential escalation and header-spoofing attacks.
- ✅ Secrets are protected at rest; compromises of storage cannot extract plaintext credentials.
- ✅ Credential revocation and rotation are instantaneous with zero service downtime.
- ✅ Full audit trail for authentication attempts and security failures.
- ⚠️ Clients must store their plaintext secret upon creation; cannot be recovered if lost.

## Related
- ADR 0012: Tenant-Isolated Sparse Retrieval
- ADR 0014: Transactional Document Lifecycle and Crash Recovery
