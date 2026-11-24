# Evaluation report: verified evidence

Verified locally on 2026-10-11 (Asia/Calcutta). No cloud-model quality benchmark or production performance claim is asserted.

## Root causes confirmed before repair

1. The old runner indexed a fresh in-memory store, then compiled a graph whose retrieval node independently called `get_vector_store()`. Evaluation could search a different collection, including development data.
2. BM25 received `state["documents"]`, initially empty and later only query candidates. Hybrid requests silently became dense-only; reranking rebuilt retrieval and could search another store again.
3. The inherited dataset contains only 10 questions against three documents. Its annotation procedure and independent review are not recorded. Recall@5 at this scale is weak evidence.
4. Metrics accepted ID substrings, counted duplicate precision/nDCG gains, and the runner concatenated source/ID strings with IDs, double-counting the same retrieved items. Reranking kept only three chunks while reporting Recall@5.
5. The original CI ran unit tests only. One unit test invoked live retrieval/generation. Missing baselines were implicitly accepted/created, and missing comparisons could pass.
6. README documented Gemini 1.5 Flash/text-embedding-004, while runtime defaults are Gemini 2.5 Flash/gemini-embedding-001.
7. Qdrant collection creation passed `scalar_quantile` to `ScalarQuantization`; the installed schema requires `scalar`. The constructor was corrected and validated locally, without claiming a live-server integration test.
8. The reports API and dashboard supplied invented scores without runs and displayed unconditional success/groundedness labels.

## Executed validation

- Before changes: existing environment, `python -m pytest -q`: **17 passed**, two deprecation warnings, 36.51 seconds. The safe-query test performed live generation; it is now mocked and guarded against network access.
- After changes: existing environment: **114 passed** in ~5s on Python 3.14 with `HF_HUB_OFFLINE=1` and no keys. Fresh Python 3.14 environment from `requirements-offline.txt`: **114 passed**, 4.22s baseline / ~5s final run. Network calls and credentials are blocked in tests.
- Fresh environment `pip check`: **No broken requirements found**. Complete dependency resolution pinned in `requirements-offline.txt`.
- `python -m compileall -q src tests`: passed.
- Dashboard `npm run build`: Next.js compiled, type-checked, and generated **9 static pages** (`/`, `/_not-found`, `/evals`, `/ingest`, `/playground`, `/runtime`, `/traces`).
- CLI exit-code tests verify unavailable cross-encoder mode exits **2**, a breached quality threshold exits **1**, and a missing requested baseline exits **1**.
- Tests exercise strict IDs, empty/duplicate/missing IDs, precision denominator, nDCG ideal normalization, stable chunk IDs, corpus isolation, complete-index BM25, fusion bounds, deterministic rankings, unknown labels, retrieval exceptions/timeouts/empty results, invalid judge JSON, nonfinite/out-of-range scores, judge timeouts, index lifecycle/concurrency, ingestion idempotency/versioning/rollback/deletion, citations/context, model lifecycle/readiness, metrics/exporter, security/API, pipeline graph, evaluation extensions.

## Split CI workflows (deterministic, offline)

Normal CI does not depend on paid LLM APIs, network, or credentials. Separate workflows:

- `.github/workflows/ci.yml` — offline unit tests, `compileall`, `pip check` (Python 3.14, `requirements-offline.txt`)
- `.github/workflows/retrieval.yml` — dense / BM25 / hybrid / diagnostic regression gates (Recall@5 ≥ 0.90, MRR ≥ 0.85, nDCG@5 ≥ 0.85)
- `.github/workflows/security.yml` — security/API suite; rejects live credentials in tests
- `.github/workflows/frontend.yml` — `npm ci`, `tsc --noEmit`, `npm run build`
- `.github/workflows/performance.yml` — retrieval benchmarks at concurrency 1/5/10 (10k chunks), weekly/scheduled
- `.github/workflows/live-evaluation.yml` — manual opt-in live embeddings/generation/judge (requires `GEMINI_API_KEY` secret; uploads reports even if unavailable)

Missing baseline, missing credentials, failed benchmark, or unavailable model must not be reported as a successful evaluation.


Environment issue: sandbox process setup failed before execution (`setup refresh had errors` / `setup helper exited`). Approved elevated workspace commands allowed inspection and validation. This was not a repository test failure. During clean-environment validation the network-blocking fixture initially rejected Windows asyncio's local socket-pair setup; that test harness bug was corrected and all tests rerun. No unresolved implementation test failures remain. GitHub-hosted workflows have been authored but not remotely executed here.

## Measured retrieval results

Dataset: **28 explicitly synthetic queries**, **16 documents / 16 chunks** (12 fictional policy documents, four distractors). Includes 12 direct questions, 12 near-neighbor questions, and four comparisons with two relevant documents each. These are agent-authored controlled regression fixtures, **not human-verified ground truth**. The inherited 10-query dataset is preserved separately and does not substantiate historical resume metrics.

All three runs use the exact same corpus/query files, a new isolated LangChain in-memory vector store, deterministic normalized 512-dimensional SHA-256 lexical hash embeddings, candidate K=10, document-level evaluation K=5, 500-character recursive chunking with 50 overlap. Hybrid uses 0.6/0.4 weighted RRF with constant 60. Dense here means cosine search over lexical hash vectors, not a pretrained semantic model. Duplicate chunks collapse to unique documents before document-level scoring. MRR is truncated to the evaluated top five.

| Configuration | Recall@5 | Precision@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| dense | 1.0000 | 0.2286 | 0.9643 | 0.9759 |
| bm25 | 1.0000 | 0.2286 | 0.9821 | 0.9868 |
| hybrid | 1.0000 | 0.2286 | 0.9821 | 0.9840 |

Precision@5 is low because each query has only one or two labeled relevant documents, and its denominator is always five. Perfect recall on this small synthetic set is not a production-quality claim. Hybrid did not outperform BM25 in nDCG on these queries.

Single-pass local wall-clock measurements, milliseconds (28 samples per executed query stage; not a production load test):

| Configuration | Stage | p50 ms | p95 ms |
|---|---|---:|---:|
| dense | dense | 0.3438 | 0.4079 |
| dense | retrieve | 0.3715 | 0.4526 |
| dense | query_total | 0.3883 | 0.4761 |
| bm25 | bm25 | 0.0590 | 0.0925 |
| bm25 | retrieve | 0.0805 | 0.1360 |
| bm25 | query_total | 0.0951 | 0.1542 |
| hybrid | dense | 0.3305 | 0.4772 |
| hybrid | bm25 | 0.0795 | 0.1025 |
| hybrid | fusion | 0.0093 | 0.0141 |
| hybrid | retrieve | 0.4473 | 0.6182 |
| hybrid | query_total | 0.4649 | 0.6407 |

Query-level evidence: no retrieval exceptions and no relevant-document misses within the top five in these three runs. First-ranked document errors:

- dense: synthetic-rigel-delete, synthetic-compare-orion-retention.
- bm25: synthetic-rigel-delete.
- hybrid: synthetic-rigel-delete.

Reranking, generation and judge stages were not executed for these measured runs; their percentiles are null with zero samples. No measured faithfulness, cloud token cost, semantic model improvement, or cross-encoder improvement is available. Provider-reported generation/judge tokens will be retained in opt-in live runs if returned; exact dollar cost and embedding token usage remain unmeasured.

Machine-readable evidence, including hashes, ranked IDs, individual misses/errors, actual installed versions and timings:

- [Dense report](eval/verified_reports/offline_dense.json)
- [BM25 report](eval/verified_reports/offline_bm25.json)
- [Hybrid report](eval/verified_reports/offline_hybrid.json)

## Reproduction and regression gates

Follow [eval/METHODOLOGY.md](eval/METHODOLOGY.md). Install `requirements-offline.txt` in Python 3.14, run `python -m pytest -q`, then run each fixture configuration:

```bash
python -m src.evaluation.regression_runner --dataset eval/fixtures/queries.jsonl --corpus eval/fixtures/corpus --mode dense --min-recall 0.9
python -m src.evaluation.regression_runner --dataset eval/fixtures/queries.jsonl --corpus eval/fixtures/corpus --mode bm25 --min-recall 0.9
python -m src.evaluation.regression_runner --dataset eval/fixtures/queries.jsonl --corpus eval/fixtures/corpus --mode hybrid --min-recall 0.9
```

Offline CI runs these commands and all tests on every PR/main push. Fixture tests require Recall@5 >= 0.90, MRR >= 0.85, and nDCG@5 >= 0.85 for each configuration. These floors define the initial synthetic regression contract, not measured confidence intervals. Baseline comparisons are opt-in, fail closed, and reject mismatched dataset/corpus/configuration/dependency versions. Expensive LLM evaluation is a separate manual workflow. Missing judge credentials are unavailable, never a fabricated successful score.

## Performance benchmarks

Measured locally on a single-process in-memory store (1,000 documents / 10,000 chunks, 30 queries per concurrency scenario, Python 3.14, `sha256-lexical-512-v1` hash embeddings). These are local wall-clock samples, **not production capacity claims**. Throughput did not scale with concurrency (CPU/GIL-bound); results must be reported honestly.

| Mode | Concurrency | p50 ms | p95 ms | p99 ms | Throughput (qps) | Errors |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 1 | 35.406 | 54.023 | 73.07 | 24.351 | 0 |
| BM25 | 5 | 226.172 | 306.462 | 376.848 | 23.140 | 0 |
| BM25 | 10 | 358.918 | 640.677 | 699.426 | 21.350 | 0 |
| Dense | 1 | 213.750 | 240.849 | 245.545 | 4.614 | 0 |
| Dense | 5 | 868.326 | 1190.500 | 1295.719 | 5.529 | 0 |
| Dense | 10 | 1777.295 | 2202.470 | 2357.553 | 5.288 | 0 |
| Hybrid | 1 | 240.399 | 279.041 | 283.203 | 4.041 | 0 |
| Hybrid | 5 | 1068.257 | 1638.708 | 1764.900 | 4.632 | 0 |
| Hybrid | 10 | 2287.228 | 3036.012 | 3524.090 | 4.105 | 0 |

Memory (tracemalloc peak): BM25 ~207 MB, Dense ~676 MB, Hybrid ~554 MB. Index build ~352–359 ms; total setup (embed+write+build) ~7.7–25.4 s depending on mode.

Machine-readable results: `benchmarks/results/retrieval_{bm25,dense,hybrid}_10000.json`. Regenerate with:

```bash
python benchmarks/retrieval_benchmark.py --chunks 10000 --concurrency 1,5,10 --output benchmarks/results
```


## Files changed by this repair

- `src/retrieval/{vector_store,retriever,reranker}.py`: shared default fallback, complete collection reads, explicit retrieval modes, bounded ID-based fusion, strict evaluation reranker initialization.
- `src/pipeline/graph.py`: injectable vector store/corpus, reranking without another retrieval.
- `src/ingestion/chunker.py`: stable per-document chunk IDs and validated zero-overlap/size settings.
- `src/evaluation/{retrieval_metrics,golden_dataset,regression_runner,judge}.py` and new `offline.py`: exact metrics, dataset validation, isolated CLI, honest status/errors, machine reports, provider token metadata.
- `src/generation/generator.py`: bounded provider transport timeout and disabled implicit model retries.
- `src/api/main.py`, targeted edits to `dashboard/src/app/evals/page.tsx`: preserve ingestion/query/report routes; remove fake scores and success badges, show unavailable judge state, avoid implicit baseline overwrite.
- `tests/conftest.py`, `tests/unit/test_evaluation_reliability.py`, `test_guardrails.py`, `test_judge.py`: offline boundary, regression/failure/math/API validation.
- `.github/workflows/{ci,retrieval,security,frontend,performance,live-evaluation}.yml`: split deterministic CI (unit/retrieval/security/frontend/performance) plus opt-in live evaluation.
- `infra/{docker-compose.yml,docker-compose.full.yml,Dockerfile.api,Dockerfile.dashboard}`: corrected `LLM_PROVIDER` default (`gemini`, never `mock`), fail-closed API healthcheck, optional dashboard service, Qdrant healthcheck with `curl`.
- `dashboard/src/app/page.tsx`: overview backed by `/ready`, `/runtime/status`, `/traces`, `/evals/reports`; removed hardcoded `PASSING`, `85%`/`90%`, and fabricated latency defaults.
- `dashboard/src/app/runtime/page.tsx` (new): live readiness, index, telemetry, config, documents, refresh/delete operations backed by real endpoints.
- `dashboard/src/components/Navbar.tsx`: added Runtime nav link.
- `dashboard/src/app/ingest/page.tsx`: index specs sourced from `/runtime/status` (store backend, chunk count, version, rerank policy) instead of hardcoded Qdrant/768-dim claims.
- `eval/human_annotated/README.md` (new): unlabeled human-annotation template, JSONL schema, provenance rules.
- `eval/synthetic_regression/README.md` (new): synthetic regression suite documentation and reproduction commands.
- `eval/verified_reports/{offline_dense,offline_bm25,offline_hybrid,diagnostic_hybrid}.json`: regenerated schema-v3 reports with manifest, per-category metrics, confidence intervals, unanswerable stats, lexical answer metrics.
- `benchmarks/results/retrieval_{bm25,dense,hybrid}_10000.json`: measured 10k-chunk benchmarks at concurrency 1/5/10.
- `.gitignore`: added `/data/ingestion/` and benchmark artifacts.

Existing user edits to README/dashboard and existing video outputs were preserved. No commit, merge or deployment was performed.

## Remaining limitations

The **65% to 90%** resume claim remains **unverified**. So do historical MRR 0.95, nDCG +25%, faithfulness 95%, guardrail speed/cost and quantization performance claims. No reviewed held-out real-user benchmark exists. Real pretrained embeddings and cross-encoder quality were not measured; the existing environment reports PyTorch unavailable. Local/cloud model weight revisions and provider availability are not locked. No real Qdrant-server integration or live LLM validation was run after this repair; API compatibility was tested with deterministic injected dependencies, and the dashboard build passed. CI workflows are authored but not remotely executed on GitHub/Linux here. Latencies are machine-dependent single-pass samples. Concurrent production ingestion is not an atomic BM25/dense snapshot; legacy Qdrant documents lacking unique chunk IDs need reingestion. A hung local inference process has no hard cancellation watchdog. Tokens from existing tracing are estimates, not billing evidence. Docker images are defined but not built/pushed in this environment; `docker-compose.yml` is validated for syntax but full multi-container startup was not exercised here.

## Resume-ready statements supported by this work

- Built an isolated RAG retrieval regression harness comparing dense, BM25 and hybrid retrieval with exact-ID Recall@K, Precision@K, MRR and nDCG metrics.
- Added a reproducible 28-query synthetic regression suite over 16 documents, including distractors and multiple-relevant-document cases, with machine-readable per-query results and latency percentiles.
- Expanded automated coverage from 17 to **114 passing tests**, covering corpus isolation, metric edge cases, retrieval failures/timeouts, judge validation, API ingestion/query continuity, index lifecycle/concurrency, ingestion rollback/deletion, citations/context, model lifecycle/readiness, metrics/exporter, security/API, pipeline graph, and evaluation extensions.
- Added split offline CI quality gates (unit/retrieval/security/frontend/performance) plus a separate opt-in live LLM evaluation workflow with explicit unavailable/skipped states.
- Measured and published 10k-chunk retrieval benchmarks at concurrency 1/5/10 for hybrid, dense, and BM25, honestly reporting that throughput did not scale with concurrency (CPU/GIL-bound).
- Removed fabricated dashboard scores/success/latency defaults; added a real `/runtime` page and rewired the overview and ingest pages to live API endpoints (`/ready`, `/runtime/status`, `/documents`, `/index/refresh`, `/traces`, `/evals/reports`).
- Corrected Docker configuration: `LLM_PROVIDER` defaults to `gemini` (never `mock`), fail-closed `/ready` healthcheck, optional dashboard service in `docker-compose.full.yml`.
