"""Bounded document loaders with explicit failures.

The previous loader returned an empty Document carrying an `error` key, which
downstream ingestion happily indexed as an empty chunk. Loaders now raise a
structured error and enforce parser resource limits.
"""
from __future__ import annotations

import os
from html.parser import HTMLParser
from typing import List

from langchain_core.documents import Document

from src.config import settings
from src.runtime.errors import IngestionError, UnsupportedDocument

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}


class DocumentLoadError(IngestionError):
    code = "document_load_failed"


class _TextExtractor(HTMLParser):
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self._skip += 1
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _load_text(path: str, filename: str) -> List[Document]:
    size = os.path.getsize(path)
    if size > settings.MAX_DOCUMENT_CHARS * 4:
        raise DocumentLoadError(f"{filename} exceeds the upload size limit", component="loader")
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        content = handle.read(settings.MAX_DOCUMENT_CHARS + 1)
    if len(content) > settings.MAX_DOCUMENT_CHARS:
        raise DocumentLoadError(
            f"{filename} exceeds MAX_DOCUMENT_CHARS={settings.MAX_DOCUMENT_CHARS}", component="loader"
        )
    return [Document(page_content=content, metadata={"source": filename, "file_name": filename})]


def _load_html(path: str, filename: str) -> List[Document]:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        raw = handle.read(settings.MAX_DOCUMENT_CHARS + 1)
    if len(raw) > settings.MAX_DOCUMENT_CHARS:
        raise DocumentLoadError(f"{filename} exceeds MAX_DOCUMENT_CHARS", component="loader")
    parser = _TextExtractor()
    parser.feed(raw)
    text = "".join(parser.parts)
    return [Document(page_content=text, metadata={"source": filename, "file_name": filename})]


def _load_pdf(path: str, filename: str) -> List[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentLoadError("pypdf is not installed; PDF ingestion is unavailable", component="loader") from exc
    try:
        reader = PdfReader(path)
        if len(reader.pages) > settings.MAX_PDF_PAGES:
            raise DocumentLoadError(
                f"{filename} has {len(reader.pages)} pages, exceeding MAX_PDF_PAGES={settings.MAX_PDF_PAGES}",
                component="loader",
            )
        pages = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(
                    Document(
                        page_content=text,
                        metadata={"source": filename, "file_name": filename, "page": idx + 1},
                    )
                )
        return pages
    except DocumentLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(f"failed to parse PDF {filename}: {type(exc).__name__}", component="loader") from exc


def load_document(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocument(f"unsupported document extension '{ext}'", component="loader")
    if ext == ".pdf":
        return _load_pdf(file_path, filename)
    if ext in {".html", ".htm"}:
        return _load_html(file_path, filename)
    return _load_text(file_path, filename)
