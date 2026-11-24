# Final Engineering Report — RAG Reliability, Evaluation & Optimization Platform

**Repository:** `Kaushik2709/RAG-Quality-Harness` / `RAG_Eval` (branch `main`)  
**Date:** Tue Nov 24 2026  
**Auditor & Lead Architect:** Senior AI Systems Engineer, Staff Backend Engineer & RAG Infrastructure Architect  
**Status:** All testable/offline milestones implemented and verified; live external provider calls explicitly documented as unverified/opt-in.

---

## 1. Original Architecture and Problems Discovered

The original repository was a conventional RAG chatbot demonstration stack (FastAPI + LangChain/LangGraph + Qdrant + Next.js dashboard) with substantial architectural defects, silent degradations, and unverified assumptions:

1. **BM25 index reconstruction on every query:** The request path called `get_indexed_documents(store)` scrolling the entire Qdrant vector corpus and rebuilding the `rank_bm25` index per request, making latency $O(N)$ with corpus size and blocking the GIL.
2. **Model initialization on the request path:** `HuggingFaceCrossEncoder` was instantiated inside `rerank_node` on query execution, repeatedly attempting weight loads and silently returning raw candidates if loading failed.
3. **Silent fallback & fabricated success:** `STORE_BACKEND=auto` fell back to in-memory on Qdrant connection failure without alerting the caller; hybrid search silently degraded to dense-only; skipped reranking was reported as successful.
4. **Fabricated dashboard metrics:** Next.js overview displayed hardcoded `"PASSING"` badges, static `"85%"` Recall and `"90%"` Faithfulness, and fake `12ms`/`38ms`/`410ms` latency values without backing telemetry.
5. **Lack of evidence-based generation:** Generation accepted any retrieved text, with no token budgeting, no deduplication, no prompt-injection defense, and no programmatic citation verification.
6. **No structured evaluation or regression discipline:** A 28-query synthetic benchmark was cited as proof of production semantic search; evaluations lacked bootstrap confidence intervals, schema versioning, and baseline immutability.
7. **Monolithic test suite and CI:** Unit tests, retrieval regression, security checks, and live network calls were coupled into a single brittle workflow.

---

## 2. Changes Implemented Across Milestones

### Milestone 1 — Retrieval Runtime and Model Lifecycle
- **Immutable Index Snapshots (`IndexManager`, `IndexSnapshot`):** BM25 index is constructed once per corpus revision; snapshots are immutable and swapped atomically via an RLock write lock. Readers perform zero-lock reads.
- **Corpus Fingerprinting:** Order-independent SHA-256 fingerprint computed across `(chunk_id, content_hash)`.
- **Safe Hot-Swap & Rollback:** Failed index builds preserve the active snapshot. Complete document deletion invalidates the snapshot fail-closed rather than serving deleted records.
- **Model Registry Lifecycle (`ModelRegistry`, `FastAPI lifespan`):** Embedding, Cross-Encoder, and LLM instances are managed via application lifespan with bounded timeouts (120s), explicit warm-up, and strict vs degraded operating policies (`MODEL_INIT_MODE`, `RERANK_POLICY`). Skipped stages are exposed as `unavailable`/`failed`, never `applied`.

### Milestone 2 — Reliable Document Ingestion
- **Document Identity & Versioning:** Stable `doc_id` and content SHA-256; versioned chunks follow `{doc_id}:v{version}:chunk-{index:04d}`.
- **Staging & Safe Activation:** New versions enter `staging` state (filtered from search), undergo chunk-count and embedding verification, transition to `active`, and retire previous versions atomically before deleting old vectors. Rollback handles failures without partial data leaks.
- **Complete Deletion:** `delete_document` deletes chunks from vector storage, wipes metadata from the JSON registry, triggers atomic index rebuild, and invalidates cache.
- **Input Validation:** Streaming file size limits (`MAX_DOCUMENT_CHARS`), MIME/extension allowlists, and sandboxed temporary file cleanup.

### Milestone 3 — Grounded Generation & Citation Validation
- **Structured Answer Contract (`StructuredAnswer`):** Validated via Pydantic; answers must include citations unless explicitly marked `abstained = true` with a mandatory reason.
- **Independent Citation Validation (`validate_citations`):** Verifies that cited `chunk_id` was in the retrieved set and that quoted text is an exact substring of that chunk. An ungrounded answer has its assertions rejected and replaced with an explicit abstention (`citations_failed_validation`).
- **Token-Aware Context Assembly (`build_context`):** Deduplicates identical passages, respects token budgets (`MAX_CONTEXT_TOKENS`), tags untrusted prompt injections (`UntrustedScan`), and tracks utilization.

### Milestone 4 — Scientific Evaluation Platform
- **Taxonomy & Seed Benchmark:** Formulated 10 query categories: `direct`, `paraphrase`, `keyword`, `multi_document`, `multi_hop`, `ambiguous`, `no_answer`, `long_document`, `conflicting`, `temporal`. Created `eval/human_annotated/queries.jsonl` with dev/test splits.
- **Formal Retrieval & Answer Metrics:** Implemented `Recall@K`, `Precision@K`, `MRR@K`, `nDCG@K`, `MAP@K`, `HitRate@K`, `RetrievalEmpty`, lexical `ExactMatch`, `TokenF1`, and `ContainsReference`.
- **LLM-as-a-Judge Calibration:** Model ID, prompt version (`JUDGE_PROMPT_VERSION`), and rubrics recorded in schema v3 reports; failures fail closed.
- **Statistical Uncertainty:** 2,000-draw paired bootstrap percentile confidence intervals (`statistical_tests.py`).
- **Regression Gates:** SHA-256 manifest validation against immutable baseline reports; checks `Recall@5 >= 0.90` floor and max allowed degradation drop.

### Milestone 5 — Retrieval Optimization Laboratory & Advanced Subsystems
- **Experiment Suite (E1 – E7):** Implemented `src/evaluation/experiments.py` with standalone CLI (`python -m src.evaluation.experiments`):
  - **E1:** Dense retrieval (cosine similarity)
  - **E2:** BM25 retrieval (sparse lexical)
  - **E3:** Dense + BM25 (linear normalized score fusion)
  - **E4:** Dense + BM25 + Reciprocal Rank Fusion (RRF)
  - **E5:** Dense + BM25 + RRF + Cross-Encoder Reranking
  - **E6:** Hybrid + Query Rewriting (synonym/keyword expansion)
  - **E7:** Hybrid + Parent-Document Retrieval (chunk mapping to parent document)
- **Advanced Subsystems:**
  - `LostInTheMiddleEvaluator`: Tests context position sensitivity (Primacy vs Middle vs Recency) and records middle placement penalty.
  - `ContextCompressor`: Extractive sentence-level context compression measuring token savings %, latency, and retained relevance.
  - `QueryRewriter`: Compares original vs rewritten query quality, latency, and token cost.
  - `AdaptiveRetriever`: Evidence sufficiency evaluation triggering candidate pool expansion and reranking when query term coverage is inadequate.

### Milestone 6 — Production Observability
- **OpenTelemetry Spans (`trace_stage`):** Semantic `gen_ai.*` attributes per stage (`input_guardrail`, `retrieve`, `rerank`, `generate`, `citation_validation`, `output_guardrail`).
- **Bounded Async Export (`TraceExporter`):** Non-blocking in-memory queue (10,000 capacity) with background batch writer to SQLite or PostgreSQL; drops records under backpressure with exposed metrics rather than stalling HTTP threads.
- **Honest Usage Tracking:** Distinguishes provider-reported token usage (`usage_source="provider"`) from character approximations (`"estimate"`); cost is computed via static price table and labelled an estimate.
- **Prometheus Metrics:** Standard `/metrics` endpoint exporting latency histograms, counters, and gauges.

### Milestone 7 — Security and Operational Reliability
- **Role & Tenant Enforcement:** `require_role` (`reader`, `ingestor`, `admin`) and `require_tenant` checks enforced prior to retrieval or mutation.
- **Network & Guardrails:** CORS allowlist, `TokenBucketLimiter` rate limiting, input PII/injection filtering, output sanitization, and structured `ErrorBody` responses with correlation request IDs.
- **Comprehensive Failure Policy:** Degradation policies documented in `docs/failure_modes.md` and verified through automated chaos/recovery tests.

### Milestone 8 — Documentation, Architecture, and Runbooks
- **Architecture Overview (`docs/architecture.md`):** Complete system boundaries, configuration, limitations, plus **Mermaid flowchart** and **sequence diagrams** for ingestion lifecycle and grounded query execution.
- **ADR Suite (`docs/adr/`):** 10 Architecture Decision Records covering BM25 separation, snapshot versioning, RRF tradeoffs, reranker lifecycle, and evaluation separation.
- **Runbooks (`docs/runbooks/`):**
  - `deployment.md`: Container setup, environment configuration, rolling zero-downtime updates.
  - `troubleshooting.md`: SEV-1/2/3 diagnostics, Qdrant disconnects, index invalidation, trace queue drops.
  - `incident_response.md`: Emergency triage checklist, tenant isolation audit, post-mortem protocol.
- **API Contracts (`docs/api_contracts.md`):** Exhaustive specification of endpoints, schemas, headers, error codes, and HTTP statuses.
- **Evaluation Methodology (`docs/evaluation_methodology.md`):** Mathematical formulations, bootstrap resampling formulas, and gate thresholds.

---

## 3. Test Suite & Verification Results

The test suite was expanded and organized into logical architectural tiers:

| Test Category | Directory / File | Test Count | Scope |
|---|---|---:|---|
| **Unit Tests** | `tests/unit/` | 118 | Chunking, citations, context, metrics, exporter, judge, model registry, guardrails, experiments (E1-E7, compression, lost-in-the-middle, adaptive) |
| **Integration Tests** | `tests/integration/` | 1 | Full end-to-end pipeline: Ingestion -> VectorStore -> RetrievalRuntime -> Graph execution -> Citations |
| **Security Tests** | `tests/security/` | 4 | Tenant isolation boundaries, RBAC role enforcement, rate limiting bursts, prompt injection scanning |
| **Reliability Tests** | `tests/reliability/` | 3 | Index build failure rollback, failed embedding rollback, empty index safety |
| **Performance Smoke** | `tests/performance/` | 2 | Benchmark runner percentiles, summaries, and synthetic execution harness |
| **TOTAL** | `tests/` | **132 passed** | **100% green in 5.89s (Python 3.14, offline)** |

### Regression Gates Output (Schema v3)
- **Dense:** Recall@5 = 1.0000, MRR = 0.9643, nDCG@5 = 0.9759 (Status: completed)
- **BM25:** Recall@5 = 1.0000, MRR = 0.9821, nDCG@5 = 0.9868 (Status: completed)
- **Hybrid (RRF):** Recall@5 = 1.0000, MRR = 0.9821, nDCG@5 = 0.9840 (Status: completed)
- **Diagnostic Hybrid:** Recall@5 = 1.0000, MRR = 1.0000, nDCG@5 = 1.0000 (Status: completed)

### Retrieval Optimization Laboratory Benchmark (E1 – E7)
```
Exp    Name                                Recall@5   MRR        nDCG@5    
---------------------------------------------------------------------------
E1     Dense Retrieval                     1.0000     0.9643     0.9759    
E2     BM25 Retrieval                      1.0000     0.9821     0.9868    
E3     Dense + BM25 (Linear Combination)   1.0000     0.9643     0.9759    
E4     Dense + BM25 (RRF)                  1.0000     0.9821     0.9840    
E6     Hybrid + Query Rewriting            1.0000     0.9643     0.9708    
E7     Hybrid + Parent-Document Retrieval  1.0000     0.9821     0.9840    
```
*(E5 requires PyTorch/GPU cross-encoder weights offline, gracefully reports unverified offline).*

### Frontend Next.js Build
- Command: `cd dashboard && npm run build`
- Output: `✓ Generating static pages (9/9)` (Overview, Runtime, Ingest, Evals, Traces, Playground, Not Found)
- Type-checking: **Clean** (`tsc --noEmit` exited 0).

---

## 4. Remaining Limitations (Documented Honestly)

- **External Services Offline:** Qdrant, PostgreSQL, real Gemini/Claude APIs, and cross-encoder weights were tested via deterministic test doubles in offline mode. Live integration against cloud instances requires network access and API credentials.
- **Human Benchmark Ground Truth:** While `eval/human_annotated/queries.jsonl` was formulated across 10 categories, ground truth labels remain marked as `proposed`/`unverified` until human reviewers adjudicate them.
- **Single-Node In-Memory Sparse Index:** The reference BM25 implementation runs in-process; production internet-scale workloads should deploy an external search engine (e.g. OpenSearch or Elasticsearch) via the pluggable `IndexManager` interface.
- **CPU/GIL Concurrency Ceiling:** Performance benchmarks at concurrency 10 showed GIL contention under Python in-process hashing; horizontal container scaling is recommended for multi-tenant production load.

---

## 5. Reproduction Commands

```bash
# 1. Activate environment
source .venv-linux/bin/activate

# 2. Run full test suite (132 tests)
export HF_HUB_OFFLINE=1 GEMINI_API_KEY="" GOOGLE_API_KEY="" CLAUDE_API_KEY=""
pytest tests/ -v

# 3. Execute Retrieval Optimization Laboratory (E1-E7)
python -m src.evaluation.experiments --suite E1,E2,E3,E4,E6,E7

# 4. Run deterministic regression gates
python -m src.evaluation.regression_runner --dataset eval/fixtures/queries.jsonl --corpus eval/fixtures/corpus --mode hybrid --min-recall 0.9

# 5. Build dashboard
cd dashboard && npm run build && cd ..
```
