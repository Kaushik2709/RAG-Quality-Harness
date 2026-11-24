# Scientific RAG Evaluation Methodology

This document defines the evaluation methodology, metric formulations, statistical testing rigor, calibration protocols, and regression gate criteria for the RAG Reliability, Evaluation & Optimization Platform.

---

## 1. Core Principles

1. **Deterministic vs Semantic Separation:** Unit and software regression evaluations must remain completely deterministic, reproducible offline, and independent of external paid APIs.
2. **Honest Label Provenance:** Synthetic datasets must never be represented as proof of human-validated semantic quality. Unverified or machine-suggested labels remain categorized as `proposed` or `unverified` until verified by human reviewers.
3. **No Silent Fallbacks:** Evaluators must fail closed. When an embedding provider, cross-encoder reranker, or LLM judge is unavailable, the evaluation status is explicitly marked as `unavailable` or `skipped`; it is never recorded as a passing score.
4. **Resampling Uncertainty:** Single-number metric averages without confidence intervals or sample variance analysis must not be used to claim production quality improvements.

---

## 2. Benchmark Suite Separation

The platform maintains three distinct evaluation corpora and dataset tiers:

| Tier | Dataset Location | Purpose | Dependencies | Gate Criteria |
|---|---|---|---|---|
| **Synthetic Software Regression** | `eval/fixtures/queries.jsonl` (28 queries) | Catch software regressions, ranking mutations, and indexing failures | Offline (`HashEmbeddings`) | Recall@5 ≥ 0.90, MRR ≥ 0.85, nDCG@5 ≥ 0.85 |
| **Diagnostic Evaluation** | `eval/fixtures/diagnostic_queries.jsonl` (10 queries) | Targeted diagnostic probes for near-neighbor, distractor, and unanswerable queries | Offline (`HashEmbeddings`) | Recall@5 ≥ 0.90, MRR = 1.00, nDCG@5 = 1.00 |
| **Human-Annotated Benchmark** | `eval/human_annotated/queries.jsonl` | Representative production quality evaluation with adjudicated human ground truth | Local models or live providers | Independent per-category score tracking |

---

## 3. Query Taxonomy & Categories

The evaluation platform classifies queries into 10 structured behavioral categories:

1. **Direct Factual Lookup (`direct`):** Simple single-fact queries with high lexical and semantic overlap with target document sections.
2. **Semantic Paraphrasing (`paraphrase`):** Queries phrased using synonyms, passive voice, or domain colloquialisms that differ lexically from the source text.
3. **Keyword-Heavy (`keyword`):** Telegraphic search queries lacking natural sentence syntax (e.g. `lyra export ttl purge duration days`).
4. **Multi-Document Synthesis (`multi_document`):** Questions whose complete answer requires aggregating facts dispersed across distinct documents.
5. **Multi-Hop Reasoning (`multi_hop`):** Questions where fact A from document 1 leads to an entity or condition in document 2 to reach the answer.
6. **Ambiguous Queries (`ambiguous`):** Underspecified queries lacking necessary context (e.g. `What is the timeout threshold?`).
7. **No-Answer Queries (`no_answer`):** Queries asking about topics entirely absent from the corpus, testing proper abstention behavior.
8. **Long-Document Evidence (`long_document`):** Evidence situated inside extensive documentation sections where context truncation or position bias can occur.
9. **Conflicting Evidence (`conflicting`):** Documents containing seemingly contradictory statements (e.g. soft-delete grace period vs irreversible purge).
10. **Temporal & Version Queries (`temporal`):** Queries qualified by date, revision year, or lifecycle state (e.g. `2026 revision requirements`).

---

## 4. Retrieval Metrics Formulation

Let $Q$ be the set of queries. For a query $q \in Q$, let:
- $R_k(q) = [d_1, d_2, \dots, d_k]$ be the ordered list of retrieved document identifiers up to rank $k$.
- $E(q)$ be the ground-truth set of relevant document identifiers.
- $hit(d) = \mathbb{I}(d \in E(q))$ indicating binary relevance.

### 4.1 Recall@K
Fraction of ground-truth relevant documents retrieved in the top $K$:
$$\text{Recall@K}(q) = \frac{|\{d \in R_k(q) \mid hit(d) = 1\}|}{|E(q)|} \quad (\text{for } |E(q)| > 0)$$

### 4.2 Precision@K
Fraction of retrieved documents in the top $K$ that are relevant:
$$\text{Precision@K}(q) = \frac{\sum_{i=1}^k hit(d_i)}{K}$$

### 4.3 Mean Reciprocal Rank (MRR)
Reciprocal of the rank of the first relevant document:
$$\text{RR}(q) = \begin{cases} \frac{1}{\min \{i \mid hit(d_i) = 1\}}, & \text{if } \exists i \text{ with } hit(d_i) = 1 \\ 0, & \text{otherwise} \end{cases}$$
$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \text{RR}(q)$$

### 4.4 Normalized Discounted Cumulative Gain (nDCG@K)
Evaluates ranking quality with logarithmic position discount:
$$\text{DCG@K}(q) = \sum_{i=1}^K \frac{hit(d_i)}{\log_2(i + 1)}$$
$$\text{IDCG@K}(q) = \sum_{i=1}^{\min(|E(q)|, K)} \frac{1}{\log_2(i + 1)}$$
$$\text{nDCG@K}(q) = \frac{\text{DCG@K}(q)}{\text{IDCG@K}(q)}$$

### 4.5 Mean Average Precision (MAP@K)
$$\text{AP@K}(q) = \frac{1}{\min(|E(q)|, K)} \sum_{i=1}^K \text{Precision@i}(q) \cdot hit(d_i)$$
$$\text{MAP@K} = \frac{1}{|Q|} \sum_{q \in Q} \text{AP@K}(q)$$

### 4.6 Retrieval Empty Rate & Hit Rate
- **Hit Rate@K:** $\mathbb{I}(\text{Recall@K}(q) > 0)$
- **Retrieval Empty Rate:** Fraction of queries returning zero candidate documents.

---

## 5. Generation & Grounding Metrics

### 5.1 Deterministic Lexical Metrics
- **Exact Match (EM):** Case-folded and punctuation-normalized string equality.
- **Token F1:** Harmonic mean of token-level precision and recall against reference answers.
- **Contains Reference:** Binary indicator of whether the normalized reference string appears in the response.

### 5.2 Citation Validation
- **Citation Precision:**
  $$\frac{|\text{Accepted Citations}|}{|\text{Total Claimed Citations}|}$$
- **Citation Recall:**
  $$\frac{|\text{Retrieved Chunks Cited}|}{|\text{Retrieved Chunks with Quoted Text}|}$$
- **Failure Policy:** If a generated answer makes non-abstained assertions but all citations fail validation, the answer is rejected and replaced with an explicit abstention response (`citations_failed_validation`).

### 5.3 LLM-as-a-Judge Protocol
When live LLM evaluation is enabled via `--judge`:
- Model identifier and judge prompt version (`JUDGE_PROMPT_VERSION`) are recorded in the report metadata.
- Outputs are constrained to validated Pydantic schemas:
  - `faithfulness` $\in [0.0, 1.0]$: Groundedness of claims against retrieved context.
  - `answer_relevancy` $\in [0.0, 1.0]$: Direct responsiveness to user query.
  - `context_precision` $\in [0.0, 1.0]$: Concentration of relevant facts in retrieved context.
- Missing credentials or API errors result in `judge_status = "unavailable"` or `"failed"`; errors are never converted into successful scores.

---

## 6. Statistical Significance & Confidence Intervals

To avoid drawing false conclusions from small score differences, the platform computes **paired bootstrap percentile confidence intervals**:

### Bootstrap Resampling Procedure
1. Let $S = [m(q_1), m(q_2), \dots, m(q_N)]$ be the metric values for $N$ queries.
2. For $b = 1, \dots, B$ (default $B = 2000$ draws):
   - Sample $N$ values with replacement from $S$ using a fixed pseudo-random seed (`EVAL_BOOTSTRAP_SEED = 12345`).
   - Compute sample mean $\bar{m}_b$.
3. Compute the $\alpha / 2$ and $1 - \alpha / 2$ percentiles of $[\bar{m}_1, \dots, \bar{m}_B]$ (default $\alpha = 0.05$ for a 95% confidence interval).
4. For system comparisons $A$ vs $B$, evaluate paired difference:
   $$\Delta_b = \text{mean}(A_b) - \text{mean}(B_b)$$
   A configuration improvement is only considered significant if the 95% CI of $\Delta$ strictly excludes zero.

---

## 7. Baseline Immutability & Regression Gates

The regression runner (`src.evaluation.regression_runner`) validates incoming reports against baseline metrics (`eval/baseline_metrics.json` or `eval/verified_reports/`):

### Comparison Identity Check
Two reports are comparable only if their identity matches exactly:
- `schema_version` (must be version 3)
- `dataset.sha256`
- `dataset.corpus_sha256`
- `config` (fusion weights, chunking parameters, embedding and reranker model IDs)
- `versions` (Python and core library dependency versions)

If the comparison identity fails, the regression check fails closed with `incompatible_identity`.

### Quality Gate Floors
- **Recall Floor:** Current $\text{Recall@5} \ge 0.90$.
- **Max Allowed Drop:** $\text{Metric}_{\text{current}} \ge \text{Metric}_{\text{baseline}} - 0.02$.
- **Faithfulness Floor (when judged):** $\ge 0.85$.
