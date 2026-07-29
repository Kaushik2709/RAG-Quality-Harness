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
    c_size = chunk_size or settings.CHUNK_SIZE
    c_overlap = chunk_overlap or settings.CHUNK_OVERLAP

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

    # Attach chunk indexing metadata
    for idx, c in enumerate(chunks):
        doc_id = c.metadata.get("doc_id", c.metadata.get("source", "doc"))
        c.metadata["chunk_id"] = f"{doc_id}_c{idx}"
        c.metadata["chunk_index"] = idx

    return chunks

