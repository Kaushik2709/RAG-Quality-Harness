# Evaluation methodology

## Dataset provenance

Two datasets are kept separate. `eval/golden_dataset.jsonl` contains the 10 inherited questions against the three repository sample documents. The original author, annotation procedure, independent review, and historical runs are not recorded. Its labels are **inherited, unverified**; the name "golden" is historical. Source text supports many answers, but this is not independent human validation and sample text is not an authoritative factual reference.

`eval/fixtures/queries.jsonl` contains 28 agent-authored **synthetic software-policy regression cases**, not manually verified ground truth or real user questions. The corpus is 12 fictional policy documents plus four distractors (astronomy name overlap, general timeouts, general backups, cooking). There are 12 direct questions, 12 near-neighbor questions, and four comparisons each labeled with two relevant documents. Labels follow the explicitly authored fictional policy statements. No LLM labels are claimed as human judgments. All 16 documents are indexed in each configuration; none are synthesized during the run. This suite tests implementation behavior, not semantic generalization. At document-level K=5 this is still a small, relatively easy benchmark; perfect recall is not evidence of production quality.

For a defensible future real benchmark: collect representative user questions, freeze source revisions, record source/license and query provenance, ask reviewers to annotate relevant document IDs independently (including multiple relevant and no-answer cases), adjudicate disagreement, and reserve a held-out set. Store review status/evidence in metadata. Do not merge proposed automatic labels into a human-verified score. No-answer quality is currently covered by empty-result unit tests, not aggregate relevance metrics.

## Comparable configurations

`--mode dense`, `bm25`, `hybrid`, or `hybrid-rerank`. Dense uses cosine similarity in LangChain InMemoryVectorStore. BM25 uses lowercase word/hyphen tokenization, BM25Okapi defaults and positive-score candidates. Hybrid uses weighted reciprocal rank fusion (0.6 dense, 0.4 BM25; constant 60), deduplicating by chunk ID and bounding the candidate list. Hybrid-rerank applies the real configured cross-encoder to those exact candidates, never a second search. Its initialization must succeed; it cannot silently fall back in evaluation.

Every run builds a new explicit in-memory store, ingests the same complete corpus and passes the same chunks to BM25. It never connects to Qdrant. Candidate K=10, evaluation K=5. Reranking keeps up to 10 candidates for evaluation; the interactive API's default generation context remains three chunks. Chunking defaults are 500 characters, 50 overlap, recursive. Reports record actual settings (environment variables may override defaults).

`--backend offline` uses deterministic SHA-256 lexical hashing into 512 dimensions, normalized by L2. It needs no downloaded model or paid service and is **not a semantic embedding benchmark**. `local` uses BAAI/bge-small-en-v1.5 and may download weights. `google` uses the configured Google embedding model and requires credentials. Offline hybrid-rerank is explicitly unavailable, not a fake reranker result. Real embedding/cross-encoder comparisons remain to be measured on a reviewed dataset.

## Metrics and identity

Document IDs are corpus-relative POSIX paths. Chunk IDs combine document ID, within-document index and content SHA-256 prefix. Reordering other documents does not change a chunk ID. Source/content changes and chunking changes invalidate the corpus/configuration comparison. For API ingestion supply unique document IDs; duplicate basenames in old collections require reingestion/migration.

Labels are document-level: repeated chunks of the same document collapse **before** assigning document ranks. Score only the first five unique documents. MRR in reports is therefore truncated to that evaluated list (MRR@5); the standalone `mrr` function uses the entire supplied list. This differs from historical reports, which concatenated IDs and sources and inflated ranks/hits with substring matches. Scores from the old runner are not comparable.

Metric functions use exact IDs only. Duplicate retrieved IDs consume positions but gain relevance once; missing/empty IDs consume positions and gain zero. Expected duplicates are ignored. Recall = unique relevant hits / unique relevant labels. Precision = unique hits / K, including short lists. Binary nDCG uses logarithmic discounts and an ideal list of min(K, number of relevant labels). Empty expected sets and nonpositive K return zero. Dataset validation rejects empty/duplicate labels, empty datasets, duplicate query IDs, and labels outside the corpus. Per-query failures stay in aggregate denominators and make the run failed.

## Reproduction

Use Python 3.14 for the fully pinned offline dependencies:

```bash
python -m venv .venv
# Activate the environment for your OS, then:
python -m pip install -r requirements-offline.txt
python -m pytest -q
python -m src.evaluation.regression_runner --dataset eval/fixtures/queries.jsonl --corpus eval/fixtures/corpus --mode dense --min-recall 0.9
python -m src.evaluation.regression_runner --dataset eval/fixtures/queries.jsonl --corpus eval/fixtures/corpus --mode bm25 --min-recall 0.9
python -m src.evaluation.regression_runner --dataset eval/fixtures/queries.jsonl --corpus eval/fixtures/corpus --mode hybrid --min-recall 0.9
```

Run inherited questions separately with `python -m src.evaluation.regression_runner`. Outputs default to `eval/reports/`; committed evidence is in `eval/verified_reports/`. Each JSON contains corpus/dataset SHA-256, model identifiers, installed dependency versions, per-query ranked document/chunk IDs, missed documents, errors, and stage p50/p95. The report API returns an empty list when no runs exist. It exposes status and nullable unavailable judge values; no placeholder scores are injected.

Explicit optional baseline workflow:

```bash
python -m src.evaluation.regression_runner --accept-baseline --baseline eval/baseline_metrics.json
python -m src.evaluation.regression_runner --baseline eval/baseline_metrics.json
```

Missing/incompatible baselines fail comparison. Baseline creation is never implicit. Comparison requires matching schema, corpus hash, query hash, configuration and recorded dependencies, then checks Recall@5/MRR/nDCG@5 against a maximum drop of 0.05 and the recall floor. To compare deliberate model/configuration changes, retain separate reports on the same corpus/queries instead of passing incompatible configurations to the regression gate.

Offline CI runs all tests and all three fixture configurations. Regression tests require Recall@5 >= 0.90, MRR >= 0.85 and nDCG@5 >= 0.85; these are explicit initial acceptance floors for this controlled fixture, not statistical significance claims. The CLI additionally checks a configurable recall floor. Tests prohibit network access and remove credentials. The live workflow is manual only, uses repository secrets, and uploads reports even if unavailable/failed. It does not assert a production benchmark.

## Optional paid evaluation

Install `requirements.txt` and configure credentials privately, then:

```bash
python -m src.evaluation.regression_runner --backend google --mode hybrid --judge
python -m src.evaluation.regression_runner --backend local --mode hybrid-rerank
```

LLM generation/judging runs only with `--judge`. Missing credentials produce `judge_status: unavailable`, omit judge metrics, and cause CLI exit 2. Retrieval errors cause failed reports. Invalid judge JSON and provider timeout exceptions are tested; provider generation calls have a 30-second transport timeout, with judge retries restricted to transient availability/rate-limit failures. No hard cancellation watchdog exists for a hung local model. Offline CI jobs have a total timeout.

Latency is local wall time with linear-interpolated p50/p95 across one pass of 28 queries. Setup/ingestion is reported separately; dense includes query embedding, BM25 includes sparse ranking, fusion is timed separately. These are not repeated, warmed production load tests; machine noise affects timings. Unexecuted stages report null percentiles and zero samples. Offline tokens/cost are null/unmeasured. Opt-in live reports retain provider-reported generation/judge token metadata when available, with response coverage; embedding tokens and dollar cost remain unmeasured. existing tracing uses whitespace estimates and static pricing, not exact billing measurements. No exact dollar-savings claim is supported.

The historical **65% to 90% Recall@5** claim, MRR 0.95, nDCG +25%, faithfulness 95%, guardrail latency/savings and quantization performance claims lack reproducible run evidence and remain **unverified**. Do not use them on a resume. See EVALUATION_REPORT.md for measured results and supported bullets.
