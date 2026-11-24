# ADR 0013: Strict Claim-Level Grounding and Citation Validation

## Status
Accepted

## Date
2026-11-24

## Context
Iteration 1 validated citations by checking chunk ID existence and verbatim substring matching. However, a valid quote does not prove that every factual claim in the generated answer is supported by evidence. Furthermore, the pipeline allowed answers with partially supported claims to pass through, creating hallucinations and ungrounded statements.

## Decision
Decompose generation into claim-level representations and enforce configurable citation validation policies:
- `StructuredAnswer` includes an explicit `claims` array containing `claim_id`, `text`, and supporting `citation_ids`.
- Citation verification enforces six invariant checks:
  1. Chunk existence in retrieved context.
  2. Tenant ownership match (`chunk.tenant_id == query.tenant_id`).
  3. Active document version status (`version_state == "active"`).
  4. Verbatim quote substring existence in referenced chunk text.
  5. Claim-to-evidence coverage (every claim maps to at least one valid citation).
  6. Numerical consistency check (preventing numerical distortion or misattributed quantities).
- Configurable policies via `CITATION_VALIDATION_POLICY`:
  - `strict` (default for production): Rejects the complete answer if any required evidence or claim fails; returns safe abstention.
  - `review`: Flags partially supported answers for internal inspection (`review_required: true`).
  - `disabled`: Bypasses rejection for development or comparative evaluation.

## Consequences
- ✅ Guarantees that fully validated answers contain zero unsupported factual claims.
- ✅ Completely prevents tenant evidence leakage into generated responses.
- ✅ Replaces substring heuristic matches with structured semantic claim coverage.
- ⚠️ Strict rejection may increase abstention rate on complex multi-hop queries when context chunks lack complete facts; requires higher retrieval precision.

## Related
- ADR 0008: Citation Validation and Grounded Generation
- ADR 0015: Retrieved Evidence Injection Defense
