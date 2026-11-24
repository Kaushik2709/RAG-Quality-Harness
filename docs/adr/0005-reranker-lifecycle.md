# ADR 0005: Reranker Lifecycle is Application-Managed

## Status
Accepted

## Date
2026-11-24

## Context
The cross-encoder reranker was originally constructed on the query path, potentially loading model weights repeatedly. This adds unpredictable latency and prevents observability of model readiness.

## Decision
Manage reranker as an application-lifecycle component via `src/runtime/model_registry.py`:
- `ModelRegistry` initializes embedding, reranker, and LLM clients on startup (or lazily on first use).
- `ModelState` tracks: `ready`, `loading`, `failed`, `unavailable`.
- `MODEL_INIT_MODE` configures behavior:
  - `strict`: fail requests when a required model is unavailable.
  - `degraded` (default): skip explicitly optional stages (rerank) and report `unavailable` status.
- `RERANK_POLICY`:
  - `strict`: rerank required; fail if unavailable.
  - `degraded` (default): skip rerank, report `rerank_status: "unavailable"`.
  - `disabled`: never rerank.
- `MODEL_INIT_MAX_ATTEMPTS`, `MODEL_INIT_BACKOFF_SECONDS`, `MODEL_INIT_TIMEOUT_SECONDS` bound initialization.
- FastAPI lifespan or explicit warmup initializes models.
- `/ready` (deep=true) probes model readiness; `/runtime/status` reports model states.

## Consequences
- ✅ No model loading on the request path.
- ✅ Reranker failures are observable; never silently claimed as success.
- ✅ Degraded mode is explicit and reported; never a silent fallback.
- ✅ Bounded retries and timeouts prevent hung initialization.
- ⚠️ Real cross-encoder (BAAI/bge-reranker-base) and HuggingFace embeddings unavailable offline; only unit-tested with fakes.
- ⚠️ No hard cancellation watchdog for a hung local inference process.

## Related
- ADR 0010: Failure and Degraded Modes
- ADR 0001: BM25 Indexing Separated from Query Path (reranking compresses existing candidates, no second search)