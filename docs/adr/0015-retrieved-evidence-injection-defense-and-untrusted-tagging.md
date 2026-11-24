# ADR 0015: Retrieved-Evidence Injection Defense and Untrusted Tagging

## Status
Accepted

## Date
2026-11-24

## Context
Untrusted third-party documents indexed in RAG pipelines frequently contain prompt injection payloads, system instruction overrides, markdown image exfiltration beacons, and delimiter collisions. If injected text is rendered unsafely into the generator LLM prompt, the model can be coerced into ignoring instructions, exfiltrating data, or hallucinating false policy guidance.

## Decision
Implement a multi-tier evidence security taxonomy and defensive prompt rendering pipeline:
- Comprehensive 9-category risk taxonomy in `src/security/document_guardrails.py`:
  1. Direct instruction override (`ignore previous instructions`, `new system prompt`).
  2. Role impersonation (`SYSTEM:`, `ASSISTANT:`, `You are now`).
  3. Delimiter and boundary collision (`</evidence>`, ````markdown`, fake boundaries).
  4. Exfiltration and markdown payload injections (`![exfil](https://...)`).
  5. Multi-line encoded payload obfuscation (base64, hex, unicode escapes).
  6. Canary token and secret extraction probes.
  7. Denial of Service and recursive payload bombs.
  8. Markdown link hijacking (`[click here](javascript:...)`).
  9. Indirect prompt injection via cross-document instructions.
- Configurable evidence security policy via `RETRIEVED_DOCUMENT_SECURITY_POLICY`:
  - `mark_untrusted` (default): Tags suspicious chunks with `is_untrusted_evidence: true` and renders them inside fortified untrusted XML wrappers with system instructions to treat content purely as quoted data.
  - `exclude`: Removes suspicious chunks from the LLM generator prompt entirely while recording audit events.
  - `quarantine`: Isolates the chunk in `QuarantineRegistry` and blocks it from retrieval until human security review.
  - `allow`: Logs security detections without modifying prompt context.
- Evidence blocks escape XML delimiters and strip control characters before injection into generator prompts.

## Consequences
- ✅ Immune to delimiter breakouts, fake role blocks, and markdown image data exfiltration.
- ✅ Granular policy choice between strict exclusion and safe marked-untrusted context inclusion.
- ✅ Zero latency overhead on clean, normal enterprise documents.
- ⚠️ Excluded chunks may prevent answers if legitimate text inadvertently matches heuristic trigger words; mitigated by precise multi-token pattern matching.

## Related
- ADR 0013: Strict Claim-Level Grounding and Citation Validation
