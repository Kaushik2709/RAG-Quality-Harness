# Incident Response Runbook

## Severity Levels

| Severity | Definition | Target Resolution | Escalation |
|---|---|---|---|
| **SEV-1** | Service outage: `/ready` returns 503, all search and ingestion paths failing. | < 30 minutes | Page on-call infrastructure lead |
| **SEV-2** | Degraded mode: Reranker failed, judge unavailable, elevated query latencies (p95 > 2s). | < 2 hours | Notify on-call AI engineer |
| **SEV-3** | Non-critical errors: Isolated document parser failures, telemetry queue drops. | Next business day | JIRA ticket to Platform Backlog |

---

## 1. SEV-1 Immediate Triage Checklist

1. **Assess Ingress & Load:** Check if request rate exceeded burst thresholds (`rag_requests_total`).
2. **Inspect Process Readiness:**
   ```bash
   curl -s http://localhost:8000/ready | jq .
   ```
3. **Check Storage Dependencies:** Verify Qdrant and Postgres sockets:
   ```bash
   nc -zv qdrant 6333
   nc -zv postgres 5432
   ```
4. **Fail-Closed Verification:** Confirm that unauthenticated requests receive 401 and cross-tenant queries receive 403.
5. **Mitigation:**
   - If corrupted memory state: restart API service (`docker compose restart api`).
   - If index out of sync: trigger `POST /index/refresh`.

---

## 2. Cross-Tenant Data Isolation Audit

If a suspected tenant leakage is reported:
1. Immediately review access logs for `X-Tenant-ID` mismatches.
2. Verify all active chunks have valid `tenant_id` metadata:
   ```sql
   SELECT tenant_id, COUNT(*) FROM documents GROUP BY tenant_id;
   ```
3. Run the automated tenant boundary verification suite:
   ```bash
   python -m pytest tests/security/test_security_suite.py -k test_tenant_isolation_boundary
   ```

---

## 3. Post-Incident Review Protocol

Within 48 hours of any SEV-1 or SEV-2 incident, complete an incident post-mortem:
- **Timeline:** Timestamped log of detection, notification, mitigation, and permanent fix.
- **Root Cause Analysis (5 Whys):** Technical failure mechanism.
- **Impact:** Total failed queries, dropped traces, affected tenants.
- **Preventative Action Items:** Concrete automated test, alert, or safeguard added to prevent recurrence.
