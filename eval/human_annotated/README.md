# Human-Annotated Evaluation Dataset

This directory is reserved for **human-reviewed retrieval and generation evaluation**.

## Status

**Empty / unlabeled template.** No human annotations exist yet. The inherited synthetic fixtures live in `eval/fixtures/` and `eval/synthetic_regression/`.

## Purpose

Provide a representative, human-verified benchmark for RAG quality that is clearly separated from synthetic regression fixtures:

- Direct factual lookup
- Semantic paraphrasing
- Keyword-heavy queries
- Multi-document / multi-hop questions
- Ambiguous queries
- No-answer questions
- Long-document questions
- Conflicting evidence
- Temporal / document-version questions

## Required file format (JSONL)

Each line must be a valid JSON object with these fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `query_id` | string | Yes | Stable, unique query identifier |
| `query` | string | Yes | Natural-language user question |
| `reference_answer` | string | No | Human-written reference answer |
| `expected_relevant_doc_ids` | string[] | Yes | Document IDs judged relevant |
| `expected_relevant_chunk_ids` | string[] | No | Chunk IDs judged relevant |
| `answerable` | boolean | Yes | Whether the question can be answered from the corpus |
| `category` | string | Yes | One of: `direct`, `paraphrase`, `keyword`, `multi_document`, `multi_hop`, `ambiguous`, `no_answer`, `long_document`, `conflicting`, `temporal` |
| `annotation_status` | string | Yes | `unverified`, `single_reviewer`, `adjudicated` |
| `reviewer` | string | No | Reviewer name / handle |
| `source_corpus_version` | string | No | Corpus version labels were validated against |
| `split` | string | Yes | `dev` or `test` |
| `metadata` | object | No | Additional provenance / difficulty notes |

## Recommended workflow

1. Collect ~150–200 representative user questions against a frozen corpus revision.
2. Freeze source revisions and record source/license and query provenance.
3. Ask reviewers to annotate relevant document IDs independently (including no-answer cases).
4. Adjudicate disagreements and reserve a held-out `test` split.
5. Store review status / evidence in `metadata`.
6. Never merge automatic/LLM-suggested labels into a human-verified score.

## Category aliases

Aliases are supported for backward compatibility:
- `near-neighbor` → `near_neighbor`

Unknown categories are descriptive metadata, not fatal.

## Provenance rules

- Labels are document-level: repeated chunks of the same document collapse before document ranking.
- Only score the first K unique documents.
- Empty expected sets return zero for all metrics.
- Dataset validation rejects empty/duplicate labels, empty datasets, duplicate query IDs, and labels outside the corpus.

## Verification

Human labels for this directory remain **unverified** until reviewers complete adjudication. Do not cite scores from this directory as production quality evidence until human review is recorded.

## Reproduction

Run against this directory only after files are added:

```bash
python -m src.evaluation.regression_runner \
  --dataset eval/human_annotated/queries.jsonl \
  --corpus eval/human_annotated/corpus \
  --mode hybrid --min-recall 0.9 --split test
```
