"""Normalisation facade kept separate from fingerprints for testability."""
from __future__ import annotations

from src.ingestion.fingerprint import content_sha256, normalize_text


def normalize_document(text: str) -> tuple[str, str]:
    normalized = normalize_text(text)
    return normalized, content_sha256(normalized)


__all__ = ["normalize_document", "normalize_text", "content_sha256"]
