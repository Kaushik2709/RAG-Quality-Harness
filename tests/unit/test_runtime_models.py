"""Model lifecycle, readiness and structured runtime failure behaviour."""
from __future__ import annotations

import time

import pytest

from src.runtime.errors import ModelUnavailable
from src.runtime.model_registry import ModelRegistry, ModelRole, ModelState
from src.runtime.readiness import collect_readiness


def _registry(**kwargs):
    return ModelRegistry(
        mode=kwargs.pop("mode", "degraded"),
        rerank_policy=kwargs.pop("rerank_policy", "degraded"),
        max_attempts=kwargs.pop("max_attempts", 1),
        backoff_seconds=0,
        timeout_seconds=kwargs.pop("timeout_seconds", 5),
        factories=kwargs.pop(
            "factories",
            {
                ModelRole.EMBEDDING: lambda: "embeddings",
                ModelRole.RERANKER: lambda: "reranker",
                ModelRole.LLM: lambda: "llm",
            },
        ),
        **kwargs,
    )


def test_degraded_registry_reports_unavailable_optional_reranker():
    def broken():
        raise RuntimeError("weights missing")

    registry = _registry(
        mode="degraded",
        rerank_policy="degraded",
        factories={ModelRole.EMBEDDING: lambda: "e", ModelRole.RERANKER: broken, ModelRole.LLM: lambda: "l"},
    )
    assert registry.get_reranker() is None
    status = registry.status()["models"]["reranker"]
    assert status["state"] == ModelState.FAILED.value
    assert "weights missing" in status["error"]
    # A second call does not retry by default.
    assert registry.get_reranker() is None
    assert status["attempts"] == 1
    # Explicit retry is available.
    assert registry.get_reranker(retry_failed=True) is None
    assert registry.status()["models"]["reranker"]["attempts"] == 2


def test_strict_registry_fails_closed_and_strict_reranker_raises():
    registry = _registry(mode="strict", rerank_policy="strict", factories={})
    with pytest.raises(ModelUnavailable):
        registry.get_embeddings()
    with pytest.raises(ModelUnavailable):
        registry.get_reranker()
    assert registry.status()["models"]["embedding"]["state"] == ModelState.FAILED.value


def test_disabled_reranker_is_not_initialized():
    called = []
    registry = _registry(
        rerank_policy="disabled",
        factories={ModelRole.RERANKER: lambda: called.append(1)},
    )
    assert registry.get_reranker() is None
    assert called == []
    assert registry.status()["models"]["reranker"]["state"] == ModelState.DISABLED.value


def test_registry_initializes_once_and_warmup_marks_state():
    calls = []

    class _Embeddings:
        def embed_query(self, text):
            return [1.0]

    def factory():
        calls.append(1)
        return _Embeddings()

    registry = _registry(factories={ModelRole.EMBEDDING: factory})
    first = registry.get_embeddings()
    second = registry.get_embeddings()
    assert first is second
    assert calls == [1]
    result = registry.warmup(embeddings=True, reranker=False, llm=False)
    assert result["embedding"]["warmed"] is True
    assert registry.status()["models"]["embedding"]["warmed"] is True


def test_initialization_timeout_is_bounded():
    def slow():
        time.sleep(1)
        return "late"

    registry = _registry(timeout_seconds=0.05, factories={ModelRole.RERANKER: slow})
    started = time.perf_counter()
    assert registry.get_reranker() is None
    assert time.perf_counter() - started < 0.5
    assert "TimeoutError" in registry.status()["models"]["reranker"]["error"]


def test_readiness_treats_disabled_auth_as_degraded_in_production(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "APP_ENV", "production")
    monkeypatch.setattr(settings, "API_KEY", "")
    monkeypatch.setattr(settings, "TRACE_SINK", "none")
    monkeypatch.setattr(settings, "MODEL_INIT_MODE", "degraded")
    report = collect_readiness(deep=False)
    assert report["ready"] is False
    assert "API_KEY is required when APP_ENV=production" in report["reasons"]
    assert report["components"]["authentication"]["status"] == "disabled"
