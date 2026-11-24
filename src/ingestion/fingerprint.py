"""Document identity and content fingerprints.

Document identity is scoped by tenant and supplied by the caller (or derived
from tenant + logical source). It is never the global filename. Content is
hashed separately so repeated uploads are idempotent and true changes create a
new version.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Optional

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def content_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def derive_document_id(tenant_id: str, source: str, explicit: Optional[str] = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    basis = f"{tenant_id}\x00{(source or 'api_text').strip().lower()}"
    return f"doc-{hashlib.sha256(basis.encode('utf-8')).hexdigest()[:20]}"


def chunk_fingerprint(chunk_ids) -> str:
    digest = hashlib.sha256()
    for chunk_id in sorted(chunk_ids):
        digest.update(str(chunk_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
