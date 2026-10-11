"""Unit/regression tests must never depend on credentials or network services."""
import socket
import pytest
from src.config import settings


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
