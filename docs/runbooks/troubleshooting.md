# Troubleshooting Runbook

This guide details diagnostics, mitigation, and recovery procedures for common operational issues.

---

## 1. Fast Diagnostics Flowchart

1. Check endpoint health:
   ```bash
   curl -i http://localhost:8000/ready
   ```
2. If HTTP 503: inspect JSON body for failing component (`models`, `vector_store`, or `index`).
3. Check Prometheus metrics for error spikes:
   ```bash
   curl -s http://localhost:8000/metrics | grep -E "rag_errors_total|rag_guardrail_total"
   ```
4. Query recent trace error spans:
   ```bash
   sqlite3 /data/traces_local.db "SELECT stage, status, error_category, latency_ms FROM traces WHERE status='error' ORDER BY timestamp DESC LIMIT 20;"
   ```

---

## 2. Specific Failure Scenarios

### Scenario A: Qdrant Connection Refused
- **Symptoms:** `/ready` reports `vector_store: unavailable`, queries fail with 503.
- **Root Cause:** Qdrant container terminated, OOM-killed, or port 6333 unreachable.
- **Action:**
  1. Inspect container status: `docker compose ps qdrant`
  2. Inspect container logs: `docker compose logs --tail=100 qdrant`
  3. Restart Qdrant: `docker compose restart qdrant`
  4. Trigger index rebuild: `curl -X POST http://localhost:8000/index/refresh`

### Scenario B: Sparse BM25 Index Invalidation
- **Symptoms:** `/ready` reports `retrieval index invalidated`, status is `invalidated`.
- **Root Cause:** A document deletion occurred, but subsequent sparse index rebuild failed. The system intentionally fails closed to prevent serving deleted documents.
- **Action:**
  1. Trigger explicit refresh: `curl -X POST http://localhost:8000/index/refresh`
  2. Verify index state: `curl -s http://localhost:8000/runtime/status | jq .index`

### Scenario C: High Trace Dropped Count (`TraceExporter`)
- **Symptoms:** Telemetry HUD shows increasing `dropped` count; warning in logs.
- **Root Cause:** PostgreSQL or SQLite trace sink is experiencing high write latency, causing the bounded queue (10,000 items) to saturate.
- **Action:**
  1. Check disk I/O and SQLite lock contention.
  2. If using Postgres, inspect active connections and vacuum locks.
  3. Increase `TRACE_QUEUE_SIZE` or `TRACE_FLUSH_INTERVAL_SECONDS` if bursty traffic is expected.

### Scenario D: Cross-Encoder Reranker Unavailable
- **Symptoms:** Runtime status reports `rerank: {status: 'unavailable', degraded: true}`.
- **Root Cause:** PyTorch weights missing or HuggingFace Hub network call failed during cold start.
- **Action:**
  1. Under `RERANK_POLICY=degraded`, the pipeline safely bypasses reranking and continues serving RRF results without throwing 500.
  2. To restore reranking, download weights locally or provision GPU/PyTorch environment.
