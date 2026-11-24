"""Unit/regression tests must never depend on credentials or network services.

Every test also starts from a clean process-local runtime: model registry,
retrieval index, exporter worker, rate limiter, metrics and the cached default
vector store. Without this, a failure in one test can make the next test pass
or fail depending on execution order.
"""
import socket

import pytest

from src.config import settings


def _reset_process_state() -> None:
    from src.api.dependencies import reset_limiter
    from src.observability.metrics import metrics
    from src.retrieval.retrieval_runtime import reset_retrieval_runtime
    from src.retrieval.vector_store import reset_default_vector_store
    from src.runtime.lifecycle import reset_model_registry
    from src.tracing.exporter import reset_exporter

    reset_limiter()
    reset_exporter()
    reset_retrieval_runtime()
    reset_model_registry()
    reset_default_vector_store()
    metrics.reset()
    try:
        # The API caches ingestion services by store identity; Python may reuse
        # an id after GC, so clear the cache between tests.
        from src.api import main

        main._ingestion_services.clear()
    except Exception:  # noqa: BLE001 - API import is not required for unit tests
        pass


@pytest.fixture(autouse=True)
def reset_runtime_state():
    _reset_process_state()
    yield
    _reset_process_state()


@pytest.fixture(autouse=True)
def offline_test_boundary(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    monkeypatch.setattr(settings, "CLAUDE_API_KEY", "")
    original_connect = socket.socket.connect
    original_pair = socket.socketpair
    creating_pair = False
    def guarded_connect(sock, address):
        if creating_pair and address[0] in {"127.0.0.1", "::1"}:
            return original_connect(sock, address)
        raise AssertionError("Network calls are prohibited in offline tests")
    def local_pair(*args, **kwargs):
        nonlocal creating_pair
        creating_pair = True
        try:
            return original_pair(*args, **kwargs)
        finally:
            creating_pair = False
    def blocked(*args, **kwargs):
        raise AssertionError("Network calls are prohibited in offline tests")
    monkeypatch.setattr(socket, "socketpair", local_pair)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", blocked)
