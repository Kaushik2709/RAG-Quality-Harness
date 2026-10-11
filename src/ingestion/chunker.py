from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from src.config import settings

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
            text_chunks = self.split_text(doc.page_content)
            for chunk in text_chunks:
                result.append(Document(page_content=chunk, metadata=dict(doc.metadata)))
        return result

try:
    from langchain_text_splitters.character import RecursiveCharacterTextSplitter, CharacterTextSplitter
except (ImportError, OSError):
    RecursiveCharacterTextSplitter = PurePythonTextSplitter
    CharacterTextSplitter = PurePythonTextSplitter

def get_text_splitter(
    strategy: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
):
    """
    Returns a standard TextSplitter instance (LangChain or Pure Python Fallback).
    """
    strat = strategy or settings.CHUNKER_STRATEGY
    c_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    c_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    if c_size <= 0 or c_overlap < 0 or c_overlap >= c_size:
        raise ValueError("Require chunk_size > chunk_overlap >= 0")
    if strat not in {"fixed", "recursive"}:
        raise ValueError("Unknown chunking strategy")

    try:
        if strat == "fixed":
            return CharacterTextSplitter(
                chunk_size=c_size,
                chunk_overlap=c_overlap,
                separator=" "
            )
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=c_size,
                chunk_overlap=c_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
    except (Exception, OSError):
        return PurePythonTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)

def chunk_documents(
    docs: List[Document],
    strategy: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Chunks LangChain Document objects using TextSplitters.
    """
    splitter = get_text_splitter(strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    # Content identity and per-document position are stable across corpus ordering.
    import hashlib
    counts = {}
    for c in chunks:
        doc_id = c.metadata.get("doc_id") or c.metadata.get("source")
        if not doc_id:
            raise ValueError("Documents require a stable doc_id or source")
        idx = counts.get(doc_id, 0)
        counts[doc_id] = idx + 1
        digest = hashlib.sha256(c.page_content.encode("utf-8")).hexdigest()[:16]
        c.metadata.update(doc_id=doc_id, chunk_id=f"{doc_id}::c{idx}::{digest}", chunk_index=idx)
    return chunks
