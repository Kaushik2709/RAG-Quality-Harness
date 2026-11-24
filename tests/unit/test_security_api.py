"""Authentication, authorization, rate limiting and API boundary tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import reset_limiter
from src.runtime.errors import AuthenticationError, AuthorizationError
from src.security.authentication import Identity, authenticate
from src.security.authorization import require_role, require_tenant
from src.security.ratelimit import TokenBucketLimiter


@pytest.fixture
def client(monkeypatch):
    from src.api import main

    monkeypatch.setattr(main, "BASE_DIR", "/nonexistent-rag-test-dir")
    monkeypatch.setattr("src.api.main.settings.RATE_LIMIT_PER_MINUTE", 0)
    return TestClient(main.app)


def test_authentication_disabled_is_explicit(monkeypatch):
    monkeypatch.setattr("src.security.authentication.settings.API_KEY", "")
    identity = authenticate(None, "tenant-a")
    assert identity.authenticated is False
    assert identity.auth_method == "disabled"
    assert identity.tenant_id == "tenant-a"


def test_authentication_enforces_key_and_tenant_allowlist(monkeypatch):
    monkeypatch.setattr("src.security.authentication.settings.API_KEY", "secret-key")
    monkeypatch.setattr("src.security.authentication.settings.TENANT_IDS", "t1,t2")
    with pytest.raises(AuthenticationError):
        authenticate(None)
    with pytest.raises(AuthenticationError):
        authenticate("wrong")
    with pytest.raises(AuthenticationError):
        authenticate("secret-key", "not-allowed")
    identity = authenticate("secret-key", "t1")
    assert identity.authenticated is True
    assert identity.roles == ("reader", "ingestor", "admin")


def test_authorization_helpers():
    reader = Identity(subject="u", tenant_id="t1", roles=("reader",))
    require_role(reader, "reader")
    with pytest.raises(AuthorizationError):
        require_role(reader, "admin")
    require_tenant(reader, "t1")
    with pytest.raises(AuthorizationError):
        require_tenant(reader, "t2")


def test_token_bucket_burst_and_refill(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr("src.security.ratelimit.time.monotonic", lambda: clock["now"])
    limiter = TokenBucketLimiter(per_minute=60, burst=2)
    assert limiter.allow("key")[0] is True
    assert limiter.allow("key")[0] is True
    allowed, retry_after = limiter.allow("key")
    assert allowed is False
    assert retry_after > 0
    clock["now"] += 1.0
    assert limiter.allow("key")[0] is True
    limiter.reset()
    assert limiter.allow("key")[0] is True


def test_api_requires_key_when_configured(client, monkeypatch):
    monkeypatch.setattr("src.api.main.settings.API_KEY", "secret-key")
    response = client.post("/query", json={"query": "hello"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    allowed = client.post("/query", json={"query": "hello"}, headers={"X-API-Key": "secret-key"})
    # Authentication passed; the request reaches the pipeline and fails closed
    # because no model/index is configured in this unit environment.
    assert allowed.status_code != 401


def test_api_rate_limit_and_correlation_id(client, monkeypatch):
    monkeypatch.setattr("src.api.main.settings.RATE_LIMIT_PER_MINUTE", 1)
    monkeypatch.setattr("src.api.main.settings.RATE_LIMIT_BURST", 1)
    reset_limiter()
    first = client.get("/", headers={"X-Request-ID": "req-fixed"})
    assert first.status_code == 200
    assert first.headers["X-Request-ID"] == "req-fixed"
    second = client.get("/")
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "rate_limited"
    assert second.headers["Retry-After"]


def test_api_rejects_unsupported_upload_extension(client):
    response = client.post(
        "/ingest/file",
        files={"file": ("malware.exe", b"not a document", "application/octet-stream")},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_document"


def test_api_upload_content_type_mismatch_is_rejected(client):
    response = client.post(
        "/ingest/file",
        files={"file": ("policy.txt", b"text", "image/png")},
    )
    assert response.status_code == 415


def test_api_upload_size_limit_is_enforced(client, monkeypatch):
    monkeypatch.setattr("src.api.main.settings.MAX_UPLOAD_BYTES", 4)
    response = client.post(
        "/ingest/file",
        files={"file": ("policy.txt", b"much more than four bytes", "text/plain")},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "document_too_large"


def test_api_delete_missing_document_returns_404(client, monkeypatch):
    from langchain_core.vectorstores import InMemoryVectorStore

    from src.api import main
    from src.evaluation.offline import HashEmbeddings

    store = InMemoryVectorStore(embedding=HashEmbeddings())
    monkeypatch.setattr(main, "get_vector_store", lambda: store)
    response = client.delete("/documents/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "document_not_found"


def test_api_readiness_metrics_and_runtime_status_are_available(client):
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready")
    assert ready.status_code in {200, 503}
    assert "components" in ready.json()
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "rag_uptime_seconds" in metrics.text
    runtime = client.get("/runtime/status").json()
    assert runtime["config"]["store_backend"] in {"auto", "qdrant", "memory"}
    assert "readiness" in runtime
