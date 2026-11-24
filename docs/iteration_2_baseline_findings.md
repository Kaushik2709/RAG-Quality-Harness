# Iteration 2 Baseline Audit & Pre-Implementation Findings Report

**Date:** 2026-11-24  
**Starting Baseline Commit:** `8d92d6788805b833e44ad154f6448f5077169123`  
**Repository HEAD:** `8d92d6788805b833e44ad154f6448f5077169123` (`main`)  
**Audit Conducted by:** Staff AI Infrastructure Engineer, Senior Backend Engineer, Application Security Engineer & RAG Evaluation Researcher  

---

## 1. Executive Summary & Verification of Baseline

The baseline repository was inspected at commit `8d92d67`. Prior to any code modifications, the entire offline test suite, retrieval regression benchmarks, and CI workflows were executed and audited.

### Baseline Execution Results:
1. **Offline Pytest Suite:** 132 passed in 6.90s (`pytest -q`).
2. **Retrieval Regressions (`regression_runner`):**
   - Dense retrieval: Recall@5 = 1.0, MRR = 0.964, nDCG@5 = 0.976
   - BM25 retrieval: Recall@5 = 1.0, MRR = 0.982, nDCG@5 = 0.987
   - Hybrid retrieval: Recall@5 = 1.0, MRR = 0.982, nDCG@5 = 0.984
   - Diagnostic hybrid: Recall@5 = 1.0, MRR = 1.0, nDCG@5 = 1.0
3. **CI Workflows Audit:**
   - `.github/workflows/security.yml` did not execute `tests/security/test_security_suite.py`, running only selected unit tests.
   - Integration tests with real PostgreSQL and Qdrant were absent from standard CI.
   - Benchmark runs were restricted to synthetic hash embeddings with 10k chunks and lacked concurrency/hardware characterization.

---

## 2. Gap Analysis by Engineering Pillar

### P0 — Tenant-Scoped Authentication & Authorization
- **Current State:** A single global static `API_KEY` is checked in constant time, but anyone providing this key receives all roles (`reader`, `ingestor`, `admin`). The client selects tenant identity via `X-Tenant-ID` header.
- **Defects:**
  - No tenant-scoped credential registry.
  - Client header `X-Tenant-ID` can override or select any tenant without verification.
  - Plain API keys are not hashed; no support for key generation, expiration, revocation, rotation, or audit events.
  - Endpoints `/runtime/status`, `/traces`, `/evals/reports` lack tenant-scoping or authorization checks.
  - Production fails open if `API_KEY` is empty (dev mode defaults to anonymous admin).

### P0 — Tenant-Isolated Sparse Retrieval
- **Current State:** A global BM25 `IndexSnapshot` is constructed over all indexed documents regardless of tenant. While search post-filters by `doc.metadata["tenant_id"] == tenant_id`, the BM25 term frequencies and inverse document frequencies (IDF) are calculated across all tenants.
- **Defects:**
  - Cross-tenant documents distort BM25 scoring statistics.
  - `RetrievalRuntime.search()` unconditionally calls `ensure_index(store)` even when `mode="dense"`, forcing unnecessary BM25 index creation.
  - Deletion of Tenant A's document invalidates or rebuilds the entire global index instead of being scoped.

### P0 — Strict Claim-Level Citation & Grounding Validation
- **Current State:** `validate_citations()` checks chunk retrieval and substring presence. If at least one citation matches, `status = "partially_validated"` and `supported = True`, allowing answers with fabricated citations to be returned.
- **Defects:**
  - No strict rejection policy for partially validated answers.
  - No claim-level decomposition (claims mapped to evidence chunk IDs).
  - Cited chunks are not verified against active document version or tenant boundary.
  - Citation recall is incorrectly defined as `len(accepted_ids) / len(retrieved_chunks)` rather than coverage of claims or ground truth chunks.
  - Semantic entailment is conflated with lexical substring matching.

### P0 — Retrieved Document Prompt-Injection Policies
- **Current State:** `scan_documents()` flags prompt injection rules into `UntrustedScan`. In `generate_node`, if flagged, it only increments a metric and logs; the tainted text is still passed to the LLM.
- **Defects:**
  - No configurable enforcement actions (`allow`, `mark_untrusted`, `exclude`, `quarantine`).
  - No prompt-level evidence isolation for untrusted documents.
  - No comprehensive adversarial test suite covering indirect injection, role spoofing, fake delimiters, and exfiltration.

### P1 — Transactional Document Lifecycle & Crash Recovery
- **Current State:** `DocumentRegistry` is an in-memory/JSON file backed by a process-level `RLock`.
- **Defects:**
  - Not safe for multi-worker processes or crash recovery.
  - No durable PostgreSQL-backed registry with schema migrations.
  - No state machine transitions (`RECEIVED` -> `VALIDATING` -> `EXTRACTING` -> `CHUNKING` -> `EMBEDDING` -> `STAGING` -> `VERIFYING` -> `ACTIVATING` -> `ACTIVE`).
  - Cross-system coordination between PostgreSQL and Qdrant lacks transactional outbox / recovery reconciliation.
  - Deletion does full scans when registry metadata is missing.
  - PDF parser joins pages into a single string, dropping page numbers and section provenance.

### P1 — Scientific Human Evaluation & Dataset Integrity
- **Current State:** `eval/human_annotated/queries.jsonl` contains 10 queries, all marked `proposed`.
- **Defects:**
  - Paraphrase leakage: `human-001` is in `dev`, but its paraphrase `human-002` is in `test`.
  - Contradiction: `human-006` has `answerable: false` but non-empty `expected_relevant_doc_ids`.
  - No grouping before splitting by query family.
  - Missing formal LLM-as-a-Judge calibration study and statistical uncertainty disclosure.

### P1 — Advanced Retrieval Experiments (E5–E7)
- **Current State:** Experiment framework exists in `src/evaluation/experiments.py`, but E5 was marked unverified because cross-encoder weights were not run locally; query rewriting was purely rule-based; compression was character overlap.
- **Defects:**
  - Missing comparative before/after metrics (Recall, MRR, nDCG, latency, memory).
  - Lost-in-the-middle lacks repeated controlled trials with variance.
  - Adaptive retrieval relies only on lexical term overlap, causing false abstentions.

### P1 — Performance & Scalability Evidence
- **Current State:** Benchmark script only tests 10k chunks with synthetic hash embeddings and concurrency 1, 5, 10 on in-memory store.
- **Defects:**
  - Missing corpus scaling (1k, 10k, 50k).
  - Missing cold vs warm latency, p50/p95/p99, dense vs sparse vs fusion vs rerank stage breakdowns.
  - Missing hardware/environment specifications in benchmark outputs.

### P1 — Observability Security & Correctness
- **Current State:** `TraceRecord` has no tenant or subject ID. `/traces` and `/traces/{query_id}` allow cross-tenant inspection.
- **Defects:**
  - Traces can leak query data across tenants.
  - Exporter shutdown lacks clean queue draining and connection lifecycle guarantees.
  - Missing security metrics (auth failures, cross-tenant violations, injection blocks).

---

## 3. Implementation Plan & Prioritization

The remediation will be executed phase-by-phase in order of priority (P0 -> P1 -> P2), maintaining green test passes and measuring changes at every step.
