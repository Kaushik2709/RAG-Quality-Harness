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
- After changes: existing environment: **47 passed**, one Starlette/httpx deprecation warning, 10.57 seconds.
- Fresh Python 3.14 environment installed from offline dependencies: **47 passed**, 4.34 seconds in the final full Python test run. Network calls and credentials are blocked in tests. A prior clean run passed in 4.33 seconds; runtime varies.
- Fresh environment `pip check`: **No broken requirements found**. The complete successful dependency resolution is pinned in `requirements-offline.txt`.
- `python -m compileall -q src tests`: passed.
- Dashboard `npm run build`: Next.js 14.2.35 compiled, checked types, and generated all eight static pages. API smoke tests cover health, empty reports, blocked requests, and ingestion-to-query continuity using deterministic retrieval and mocked generation/reranking.
- CLI exit-code tests verify unavailable cross-encoder mode exits **2**, a breached quality threshold exits **1**, and a missing requested baseline exits **1**.
- Tests exercise strict IDs, empty/duplicate/missing IDs, precision denominator, nDCG ideal normalization, stable chunk IDs, corpus isolation, complete-index BM25, fusion bounds, deterministic rankings, unknown labels, retrieval exceptions/timeouts/empty results, invalid judge JSON, nonfinite/out-of-range scores, and judge timeouts.

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

## Files changed by this repair

- `src/retrieval/{vector_store,retriever,reranker}.py`: shared default fallback, complete collection reads, explicit retrieval modes, bounded ID-based fusion, strict evaluation reranker initialization.
- `src/pipeline/graph.py`: injectable vector store/corpus, reranking without another retrieval.
- `src/ingestion/chunker.py`: stable per-document chunk IDs and validated zero-overlap/size settings.
- `src/evaluation/{retrieval_metrics,golden_dataset,regression_runner,judge}.py` and new `offline.py`: exact metrics, dataset validation, isolated CLI, honest status/errors, machine reports, provider token metadata.
- `src/generation/generator.py`: bounded provider transport timeout and disabled implicit model retries.
- `src/api/main.py`, targeted edits to `dashboard/src/app/evals/page.tsx`: preserve ingestion/query/report routes; remove fake scores and success badges, show unavailable judge state, avoid implicit baseline overwrite.
- `tests/conftest.py`, `tests/unit/test_evaluation_reliability.py`, `test_guardrails.py`, `test_judge.py`: offline boundary, regression/failure/math/API validation.
- `.github/workflows/{eval-regression,llm-evaluation}.yml`, `requirements-offline.txt`, `.gitignore`: deterministic CI, separate opt-in live job, pinned environment and local artifact exclusions.
- `eval/fixtures/`, `eval/verified_reports/`, `eval/METHODOLOGY.md`, `EVALUATION_REPORT.md`, `README.md`, `.env.example`: reproducible data/evidence, provenance and aligned model defaults. Local ignored `RESUME_XFACTORS.md` has an explicit unverified-claims notice.

Existing user edits to README/dashboard and existing video outputs were preserved. No commit, merge or deployment was performed.

## Remaining limitations

The **65% to 90%** resume claim remains **unverified**. So do historical MRR 0.95, nDCG +25%, faithfulness 95%, guardrail speed/cost and quantization performance claims. No reviewed held-out real-user benchmark exists. Real pretrained embeddings and cross-encoder quality were not measured; the existing environment reports PyTorch unavailable. Local/cloud model weight revisions and provider availability are not locked. No real Qdrant-server integration or live LLM validation was run after this repair; API compatibility was tested with deterministic injected dependencies, and the dashboard build passed. CI has not yet been run on GitHub/Linux. Latencies are machine-dependent single-pass samples. Concurrent production ingestion is not an atomic BM25/dense snapshot; legacy Qdrant documents lacking unique chunk IDs need reingestion. A hung local inference process has no hard cancellation watchdog. Tokens from existing tracing are estimates, not billing evidence.

## Resume-ready statements supported by this work

- Built an isolated RAG retrieval regression harness comparing dense, BM25 and hybrid retrieval with exact-ID Recall@K, Precision@K, MRR and nDCG metrics.
- Added a reproducible 28-query synthetic regression suite over 16 documents, including distractors and multiple-relevant-document cases, with machine-readable per-query results and latency percentiles.
- Expanded automated coverage from 17 to 47 passing tests, covering corpus isolation, metric edge cases, retrieval failures/timeouts, judge validation and API ingestion/query continuity.
- Added offline retrieval CI quality gates and a separate opt-in LLM evaluation workflow with explicit unavailable/skipped states.
