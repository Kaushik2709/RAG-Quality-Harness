"""Structure-aware chunking with stable identities.

Chunks retain document identity, version, heading path, page number and
position. Stable chunk IDs are derived from the document ID, within-document
position and a content digest, so reordering other documents does not change
them. Content changes do change the digest, which is deliberate: a changed
chunk is a different unit of evidence.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from src.config import settings

_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class PurePythonTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        if not text.strip():
            return []
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(text), step):
            chunk = text[i:i + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        result = []
        for doc in documents:
            for chunk in self.split_text(doc.page_content):
                result.append(Document(page_content=chunk, metadata=dict(doc.metadata)))
        return result


try:
    from langchain_text_splitters.character import CharacterTextSplitter, RecursiveCharacterTextSplitter
except (ImportError, OSError):  # pragma: no cover - dependency is pinned offline
    RecursiveCharacterTextSplitter = PurePythonTextSplitter
    CharacterTextSplitter = PurePythonTextSplitter


def get_text_splitter(
    strategy: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
):
    strat = strategy or settings.CHUNKER_STRATEGY
    c_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    c_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    if c_size <= 0 or c_overlap < 0 or c_overlap >= c_size:
        raise ValueError("Require chunk_size > chunk_overlap >= 0")
    if strat not in {"fixed", "recursive"}:
        raise ValueError("Unknown chunking strategy")

    if strat == "fixed":
        return CharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap, separator=" ")
    return RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def heading_path(metadata: Dict) -> str:
    value = metadata.get("section_path") or ""
    return value if isinstance(value, str) else ""


def split_sections(text: str) -> List[Tuple[str, str]]:
    """Split markdown into (heading_path, body) sections.

    A document without markdown headings is a single section with an empty
    heading path. This is intentionally conservative: guessing structure from
    arbitrary prose tends to be worse than not claiming structure at all.
    """
    if not text.strip():
        return []
    lines = text.splitlines()
    stack: List[Tuple[int, str]] = []
    sections: List[Tuple[str, List[str]]] = []
    current_path = ""
    current: List[str] = []

    def flush() -> None:
        nonlocal current
        body = "\n".join(current).strip()
        if body:
            sections.append((current_path, body))
        current = []

    for line in lines:
        match = _MARKDOWN_HEADING.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2).strip()
            stack = [(lvl, name) for lvl, name in stack if lvl < level]
            stack.append((level, title))
            current_path = " > ".join(name for _, name in stack)
            current = [line]
        else:
            current.append(line)
    flush()
    if not sections:
        return [("", text.strip())]
    return sections


def chunk_documents(
    docs: List[Document],
    strategy: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    document_version: Optional[int] = None,
) -> List[Document]:
    splitter = get_text_splitter(strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[Document] = []
    for doc in docs:
        doc_id = doc.metadata.get("doc_id") or doc.metadata.get("source")
        if not doc_id:
            raise ValueError("Documents require a stable doc_id or source")
        version = document_version if document_version is not None else doc.metadata.get("document_version", 1)
        sections = split_sections(doc.page_content)
        for section_path, body in sections:
            section_doc = Document(page_content=body, metadata=dict(doc.metadata))
            section_doc.metadata["section_path"] = section_path
            section_doc.metadata["section_title"] = section_path.split(" > ")[-1] if section_path else ""
            chunks.extend(splitter.split_documents([section_doc]))

    counts: Dict[str, int] = {}
    result: List[Document] = []
    for chunk in chunks:
        chunk = Document(page_content=chunk.page_content, metadata=dict(chunk.metadata))
        doc_id = str(chunk.metadata.get("doc_id") or chunk.metadata.get("source"))
        idx = counts.get(doc_id, 0)
        counts[doc_id] = idx + 1
        if idx >= settings.MAX_CHUNKS_PER_DOCUMENT:
            raise ValueError(f"document '{doc_id}' exceeds MAX_CHUNKS_PER_DOCUMENT={settings.MAX_CHUNKS_PER_DOCUMENT}")
        digest = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:16]
        version = chunk.metadata.get("document_version", document_version if document_version is not None else 1)
        chunk.metadata.update(
            doc_id=doc_id,
            document_version=version,
            chunk_id=f"{doc_id}::c{idx}::{digest}",
            chunk_index=idx,
            content_sha256=hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest(),
        )
        result.append(chunk)
    return result
