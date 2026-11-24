# ADR 0010: Failure and Degraded Modes

## Status
Accepted

## Date
2026-11-24

## Context
The original codebase silently fell back to weaker implementations (in-memory store on Qdrant failure, dense-only when BM25 failed, fake reranker results) while reporting success. This is unacceptable for a reliability platform.

## Decision
Explicit failure/degraded modes everywhere:

| Component | Strict Mode | Degraded Mode (default) | Failure Behavior |
|---|---|---|---|
| Vector store (`STORE_BACKEND=auto`) | N/A | N/A | Raise on Qdrant failure; memory only if `ALLOW_IN_MEMORY_FALLBACK=true` |
| Reranker (`RERANK_POLICY`) | `strict`: fail request | `degraded`: skip, report `unavailable` | Never claim `applied` when skipped |
| Model init (`MODEL_INIT_MODE`) | `strict`: fail request | `degraded`: skip optional, report state | Required model failures block `/ready` |
| Index build | N/A | N/A | Failed rebuild preserves previous snapshot; `invalid_reason` set |
| Ingestion | N/A | N/A | Rollback on any failure; old version stays active |
| Judge | N/A | N/A | Missing credentials → `judge_status: unavailable`, exit 2 |
| Trace export | N/A | N/A | Bounded queue drops on full; dropped/failed counters exposed |

**Readiness** (`/ready`, `collect_readiness()`):
- `ready`: all required models ready, index ready, auth OK (if production).
- `degraded`: optional models failed, but required OK.
- `not_ready`: required failures or index invalidated.

**Structured errors**: `ErrorBody` with `code`, `message`, `component`, `retryable`, `request_id`. No silent exception swallowing.

**Telemetry**: one sink (sqlite default), bounded async queue, batch writer; dropped/failed counters in `/runtime/status`.

## Consequences
- ✅ No silent fallbacks; degraded status always explicit in API responses.
- ✅ Fail-closed: missing baseline, credentials, or model → error, not fake success.
- ✅ Observability: `/ready`, `/runtime/status`, `/metrics` expose real state.
- ✅ Tests verify: unavailable cross-encoder exits 2, breached threshold exits 1, missing baseline exits 1.
- ⚠️ No hard cancellation watchdog for hung local inference.
- ⚠️ Tokens from tracing are static-pricing estimates, not billing evidence.

## Related
- ADR 0005: Reranker Lifecycle is Application-Managed
- ADR 0002: Versioned Retrieval Snapshots