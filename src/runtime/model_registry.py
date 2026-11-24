"""Lifecycle-managed model instances.

Why this exists
---------------
The previous implementation constructed a `HuggingFaceCrossEncoder` inside
`rerank_node` for every request. That repeatedly loads model weights on the
request path and silently returned the unranked candidate list when loading
failed. Both behaviours are unacceptable in production.

This registry initialises each model at most once (with a bounded retry budget
and timeout), caches the failure, warms models explicitly, and exposes an
honest status. In `strict` mode an unavailable required model raises; in
`degraded` mode the caller receives `None` and must record that the stage was
skipped. A skipped stage is never reported as a successful one.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

from src.config import settings
from src.runtime.errors import ModelUnavailable


class ModelRole(str, Enum):
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    LLM = "llm"


class ModelState(str, Enum):
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"
    DISABLED = "disabled"
    WARMING = "warming"


@dataclass
class _Entry:
    role: ModelRole
    model_id: str
    factory: Optional[Callable[[], Any]]
    required: bool
    state: ModelState = ModelState.NOT_INITIALIZED
    value: Any = None
    error: Optional[str] = None
    attempts: int = 0
    init_ms: Optional[float] = None
    warmed: bool = False
    initialized_at: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def status(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "model_id": self.model_id,
            "state": self.state.value,
            "required": self.required,
            "attempts": self.attempts,
            "init_ms": round(self.init_ms, 2) if self.init_ms is not None else None,
            "warmed": self.warmed,
            "initialized_at": self.initialized_at,
            "error": self.error,
        }


def _utcnow() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _call_with_timeout(factory: Callable[[], Any], timeout_seconds: float) -> Any:
    """Run a potentially blocking loader with a bounded wait.

    Python cannot safely kill a thread. On timeout the worker is abandoned as a
    daemon; its eventual result is discarded. This bounds request latency but is
    not a hard cancellation primitive, which is documented in docs/failure_modes.md.
    """
    box: Dict[str, Any] = {}
    done = threading.Event()

    def runner() -> None:
        try:
            box["value"] = factory()
        except BaseException as exc:  # noqa: BLE001 - surfaced verbatim as a status string
            box["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=runner, name="model-init", daemon=True)
    thread.start()
    if not done.wait(timeout_seconds):
        raise TimeoutError(f"model initialization exceeded {timeout_seconds:.1f}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


class ModelRegistry:
    def __init__(
        self,
        *,
        mode: str = "degraded",
        rerank_policy: str = "degraded",
        max_attempts: int = 2,
        backoff_seconds: float = 0.25,
        timeout_seconds: float = 120.0,
        factories: Optional[Dict[ModelRole, Callable[[], Any]]] = None,
        model_ids: Optional[Dict[ModelRole, str]] = None,
    ):
        if mode not in {"strict", "degraded"}:
            raise ValueError("mode must be strict or degraded")
        if rerank_policy not in {"strict", "degraded", "disabled"}:
            raise ValueError("rerank_policy must be strict, degraded or disabled")
        self.mode = mode
        self.rerank_policy = rerank_policy
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.timeout_seconds = timeout_seconds
        self._factories = factories or {}
        self._model_ids = model_ids or {}

        reranker_disabled = rerank_policy == "disabled"
        self._entries: Dict[ModelRole, _Entry] = {
            ModelRole.EMBEDDING: _Entry(
                ModelRole.EMBEDDING,
                self._model_ids.get(ModelRole.EMBEDDING, settings.EMBEDDING_MODEL),
                self._factories.get(ModelRole.EMBEDDING),
                required=True,
            ),
            ModelRole.RERANKER: _Entry(
                ModelRole.RERANKER,
                self._model_ids.get(ModelRole.RERANKER, settings.RERANKER_MODEL),
                self._factories.get(ModelRole.RERANKER),
                required=rerank_policy == "strict",
                state=ModelState.DISABLED if reranker_disabled else ModelState.NOT_INITIALIZED,
            ),
            ModelRole.LLM: _Entry(
                ModelRole.LLM,
                self._model_ids.get(ModelRole.LLM, self._llm_model_id()),
                self._factories.get(ModelRole.LLM),
                required=True,
            ),
        }

    @staticmethod
    def _llm_model_id() -> str:
        return settings.GEMINI_MODEL if settings.LLM_PROVIDER == "gemini" else settings.CLAUDE_MODEL

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_embeddings(self, *, retry_failed: bool = False) -> Any:
        return self._get(ModelRole.EMBEDDING, retry_failed=retry_failed)

    def get_llm(self, *, retry_failed: bool = False) -> Any:
        return self._get(ModelRole.LLM, retry_failed=retry_failed)

    def get_reranker(self, *, retry_failed: bool = False) -> Any:
        return self._get(ModelRole.RERANKER, retry_failed=retry_failed)

    def _get(self, role: ModelRole, *, retry_failed: bool = False) -> Any:
        entry = self._entries[role]
        if entry.state == ModelState.READY:
            return entry.value
        if entry.state in {ModelState.DISABLED}:
            return None
        if entry.state == ModelState.FAILED and not retry_failed:
            return self._unavailable(entry)

        with entry.lock:
            if entry.state == ModelState.READY:
                return entry.value
            if entry.state == ModelState.DISABLED:
                return None
            if entry.state == ModelState.FAILED and not retry_failed:
                return self._unavailable(entry)
            return self._initialize(entry)

    def _initialize(self, entry: _Entry) -> Any:
        if entry.factory is None:
            entry.state = ModelState.FAILED
            entry.error = "no factory registered"
            return self._unavailable(entry)

        entry.state = ModelState.INITIALIZING
        started = time.perf_counter()
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.max_attempts + 1):
            entry.attempts += 1
            try:
                value = _call_with_timeout(entry.factory, self.timeout_seconds)
                if value is None:
                    raise RuntimeError("factory returned None")
                entry.value = value
                entry.state = ModelState.READY
                entry.error = None
                entry.init_ms = (time.perf_counter() - started) * 1000
                entry.initialized_at = _utcnow()
                return value
            except BaseException as exc:  # noqa: BLE001 - failure is reported, not swallowed
                last_error = exc
                if attempt < self.max_attempts and self.backoff_seconds:
                    time.sleep(self.backoff_seconds * attempt)

        entry.state = ModelState.FAILED
        entry.value = None
        entry.init_ms = (time.perf_counter() - started) * 1000
        entry.error = f"{type(last_error).__name__}: {last_error}"
        return self._unavailable(entry)

    def _unavailable(self, entry: _Entry) -> Any:
        if self.mode == "strict" or entry.required:
            raise ModelUnavailable(
                f"{entry.role.value} model '{entry.model_id}' is unavailable: {entry.error}",
                component=entry.role.value,
                retryable=True,
            )
        return None

    # ------------------------------------------------------------------
    # Warmup / shutdown / status
    # ------------------------------------------------------------------
    def warmup(self, *, embeddings: bool = True, reranker: bool = True, llm: bool = False) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        if embeddings:
            results["embedding"] = self._warm(
                ModelRole.EMBEDDING, lambda model: model.embed_query("warmup query")
            )
        if reranker:
            results["reranker"] = self._warm(
                ModelRole.RERANKER,
                lambda model: model.score([("warmup query", "warmup passage")]),
            )
        if llm:
            results["llm"] = self._warm(ModelRole.LLM, lambda model: None)
        return results

    def _warm(self, role: ModelRole, probe: Callable[[Any], Any]) -> Dict[str, Any]:
        try:
            model = self._get(role)
            if model is None:
                return {"role": role.value, "warmed": False, "reason": "disabled or unavailable"}
            probe(model)
            entry = self._entries[role]
            entry.warmed = True
            return {"role": role.value, "warmed": True}
        except Exception as exc:  # noqa: BLE001
            return {"role": role.value, "warmed": False, "reason": f"{type(exc).__name__}: {exc}"}

    def status(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "rerank_policy": self.rerank_policy,
            "models": {role.value: self._entries[role].status() for role in self._entries},
        }

    def reranking_available(self) -> bool:
        return self._entries[ModelRole.RERANKER].state in {ModelState.READY, ModelState.NOT_INITIALIZED}

    def close(self) -> None:
        for entry in self._entries.values():
            value = entry.value
            close = getattr(value, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - shutdown must not mask the original failure
                    pass
            entry.value = None


def build_default_registry() -> ModelRegistry:
    """Wire real factories lazily. Imports stay inside the factories so that a
    missing optional dependency is reported as a model failure, not an import
    failure of the whole application."""
    from src.ingestion.embedder import get_embeddings_model
    from src.generation.generator import get_chat_model
    from src.retrieval.reranker import build_cross_encoder

    return ModelRegistry(
        mode=settings.MODEL_INIT_MODE,
        rerank_policy=settings.RERANK_POLICY,
        max_attempts=settings.MODEL_INIT_MAX_ATTEMPTS,
        backoff_seconds=settings.MODEL_INIT_BACKOFF_SECONDS,
        timeout_seconds=settings.MODEL_INIT_TIMEOUT_SECONDS,
        factories={
            ModelRole.EMBEDDING: get_embeddings_model,
            ModelRole.RERANKER: lambda: build_cross_encoder(settings.RERANKER_MODEL),
            ModelRole.LLM: get_chat_model,
        },
    )
