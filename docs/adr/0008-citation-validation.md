# ADR 0008: Citation Validation and Grounded Generation

## Status
Accepted

## Date
2026-11-24

## Context
LLM-generated answers must be traceable to retrieved evidence. The original implementation relied only on prompt instructions for citations, with no validation.

## Decision
Structured answer contract (`src/generation/citations.py`, `src/generation/prompt_builder.py`):
- `StructuredAnswer` Pydantic model: `answer`, `citations[]` (chunk_id, quote), `abstained`, `abstention_reason`.
- Generator returns `StructuredAnswer` via structured output parsing.
- **Citation validation** (`src/pipeline/graph.py` `validate_citations_node`):
  1. Every citation references an actual retrieved chunk (by chunk_id).
  2. Every quoted passage is present in the referenced chunk (exact substring match).
  3. Citations belong to the correct document version and tenant.
  4. Unsupported citations → `citation_validation: {valid: false, errors: [...]}`.
  5. Missing evidence → `abstained=true`, `abstention_reason="no_supporting_evidence"`.
- **Context management**:
  - Token-aware budgeting (`MAX_CONTEXT_TOKENS`, `CONTEXT_MAX_CHUNKS`).
  - Duplicate chunk removal by chunk_id.
  - Document/source diversity (`CONTEXT_DIVERSIFY_SOURCES`).
- **Abstention** supported for: no relevant evidence, weak/conflicting evidence, outdated documents, unsupported questions.

## Consequences
- ✅ Citations resolve to retrieved evidence or are explicitly rejected.
- ✅ Unverifiable citations produce explicit abstention, not hallucinated support.
- ✅ Retrieved documents treated as untrusted input; prompt injections in retrieved text cannot override system instructions.
- ⚠️ Exact substring match for quotes may fail on whitespace/encoding differences; not semantic entailment.
- ⚠️ No LLM-based citation verification (cost/complexity); current check is deterministic.

## Related
- ADR 0009: Retrieval and Generation Evaluation Separated
- ADR 0010: Failure and Degraded Modes