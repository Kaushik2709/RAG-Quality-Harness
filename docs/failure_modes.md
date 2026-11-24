# Failure Modes

This document catalogs known failure modes, their detection, and mitigation.

## 1. Vector Store Unavailable

| Scenario | Detection | Behavior | Mitigation |
|---|---|---|---|
| Qdrant connection refused | `/ready` deep probe, `store_readiness()` | `STORE_BACKEND=auto` raises; `ALLOW_IN_MEMORY_FALLBACK=false` (default) | Provision Qdrant; set `STORE_BACKEND=memory` for dev only |
| Qdrant collection missing | `store_readiness()` returns `collection_missing` | Index build fails; `IndexManager` sets `invalid_reason` | Run ingestion to create collection; `/index/refresh` |
| Qdrant disk full | Write error on upsert | Ingestion fails; rollback; old version stays active | Monitor Qdrant disk; set retention policies |
| In-memory store OOM | `MemoryError` on large corpus | Index build fails; `MAX_INDEX_CHUNKS` (250k) prevents | Use Qdrant for production; shard corpora |

## 2. Model Initialization Failure

| Scenario | Detection | Behavior | Mitigation |
|---|---|---|---|
| Embedding model load fails (HF Hub unavailable) | `ModelRegistry` state = `failed` | `/ready` = `not_ready`; required model failure blocks requests | `HF_HUB_OFFLINE=1` for offline; cache weights locally |
| Reranker load fails (no PyTorch/weights) | `ModelRegistry` state = `failed` | `RERANK_POLICY=degraded`: skip rerank, report `unavailable` | Install PyTorch; cache `BAAI/bge-reranker-base`; or set `RERANK_POLICY=disabled` |
| LLM provider credentials missing | `Generator` init error | `/ready` = `degraded`; generation skipped in eval | Set `GEMINI_API_KEY` or `CLAUDE_API_KEY` |
| Model init timeout | `MODEL_INIT_TIMEOUT_SECONDS` (120s) | `ModelRegistry` state = `failed` | Increase timeout; pre-warm models |

## 3. Retrieval Index Corruption / Staleness

| Scenario | Detection | Behavior | Mitigation |
|---|---|---|---|
| Index build fails mid-way | `IndexManager._invalid_reason` set | Old snapshot remains active; `/ready` = `not_ready` | `/index/refresh` retries; check logs for root cause |
| Corpus fingerprint drift | `IndexSnapshot.corpus_fingerprint` mismatch | New snapshot built on next refresh | Fingerprint recorded in eval reports for audit |
| BM25 index empty but chunks exist | `IndexSnapshot.chunk_count=0` with active docs | `/runtime/status` shows `chunk_count=0` | Re-ingest; check chunker/embedder |
| Stale index after deletion | `IndexManager.refresh_from_store()` not called | Deleted docs still searchable | Ingestion service calls refresh on delete; manual `/index/refresh` |

## 4. Ingestion Failure

| Scenario | Detection | Behavior | Mitigation |
|---|---|---|---|
| File too large | `MAX_DOCUMENT_CHARS` (2M) | `IngestionError` with `code=too_large` | Split document; increase limit |
| Parser failure (corrupt PDF) | Loader exception | `IngestionError`; no partial chunks written | Validate files before upload; sandbox parser |
| Embedding generation fails | Embedder exception | Staging version marked `failed`; rollback | Retry with backoff; check provider quota |
| Duplicate content | SHA-256 match in registry | Returns `duplicate=true`; no new chunks | Expected behavior; no action needed |
| Concurrent same-doc ingestion | Registry version check | Second request waits or fails | Idempotent design; use unique `doc_id` |

## 5. Generation Failure

| Scenario | Detection | Behavior | Mitigation |
|---|---|---|---|
| LLM provider timeout | `LLM_TIMEOUT_SECONDS` (30s) | `GenerationError`; `generation_status=timeout` | Increase timeout; check provider status |
| LLM rate limit (429) | Provider error | `GenerationError`; `retryable=true` | Exponential backoff; `LLM_MAX_RETRIES=0` by default |
| Citation validation fails | Quote not in referenced chunk | `citation_validation.valid=false`; `abstained=true` | Improve context assembly; check chunker |
| Guardrail blocks response | PII/injection detected | `guardrail_blocked=true`; sanitized response | Review document content; adjust patterns |

## 6. Evaluation Failure

| Scenario | Detection | Behavior | Mitigation |
|---|---|---|---|
| Judge credentials missing | `--judge` without API key | `judge_status=unavailable`; CLI exit 2 | Run without `--judge`; add secret for live eval |
| Judge invalid JSON | Judge output parse error | `judge_status=failed`; retry transient | Strict schema validation; log raw output |
| Baseline mismatch | `--baseline` with different config/hash | `compare_with_baseline` returns false; CLI exit 1 | Use `--accept-baseline` for new baseline; match config |
| Retrieval exception during eval | Retriever raises | Report failed; per-query error in details | Fix retriever; eval fails closed |

## 7. Observability Failure

| Scenario | Detection | Behavior | Mitigation |
|---|---|---|---|
| Trace exporter queue full | `TraceExporter.dropped` increments | Spans dropped; request continues | Increase `TRACE_QUEUE_SIZE`; check sink latency |
| SQLite/Postgres write fails | `TraceExporter.failed` increments | Spans lost; last_error recorded | Check disk/DB connectivity; monitor `failed` metric |
| Metrics endpoint down | `/metrics` 5xx | Prometheus scrape fails | Monitor exporter health; restart API |

## 8. Concurrency / Race Conditions

| Scenario | Detection | Behavior | Mitigation |
|---|---|---|---|
| Concurrent index refresh | `IndexManager._lock` | Serialized; second waits | Low frequency; acceptable |
| Concurrent same-doc ingest | Registry file lock | Second fails with version conflict | Use unique `doc_id`; retry |
| Concurrent query during refresh | Immutable snapshot swap | Queries see old or new snapshot atomically | No partial state exposure |

## 9. Security Failure

| Scenario | Detection | Behavior | Mitigation |
|---|---|---|---|
| Cross-tenant data access | Authorization check in `/query` | `403 Forbidden`; `require_role` enforces | Test isolation; audit tenant IDs |
| Prompt injection in document | Input/output guardrails | Sanitized; `guardrail_blocked=true` | Review patterns; never trust retrieved text |
| Unauthenticated API access | `API_KEY` set, no header | `401 Unauthorized` | Enforce auth in production |

## Detection Checklist (Runbook)

On any incident:
1. Check `/ready` and `/runtime/status` for component states.
2. Check `traces_local.db` or Postgres for recent error spans.
3. Check `metrics` endpoint for `rag_*_total{status="error"}` and `dropped/failed` counters.
4. Check API logs for structured `ErrorBody` responses.
5. Run `python -m src.evaluation.regression_runner --mode hybrid` to verify retrieval integrity.
6. Check `benchmarks/results/` for latency regressions.

## Severity Classification

| Severity | Criteria | Response |
|---|---|---|
| SEV-1 | `/ready` = `not_ready`, all queries fail | Immediate; page on-call |
| SEV-2 | Degraded: rerank unavailable, judge unavailable, elevated latency | Within 1 hour; investigate |
| SEV-3 | Single query errors, trace drops, ingestion warnings | Next business day; triage |

## Escalation

- SEV-1: Page on-call → runbook → postmortem within 48h.
- SEV-2: Ticket → fix within sprint → document in ADR if architectural.
- SEV-3: Backlog → fix → regression test.