import pytest
from langchain_core.documents import Document
from src.ingestion.chunker import chunk_documents

def test_langchain_chunker():
    sample_text = (
        "Retrieval-Augmented Generation (RAG) enhances LLM responses using external document contexts. "
        "Chunking splits long documents into smaller segments to optimize embedding storage. "
        "LangChain RecursiveCharacterTextSplitter splits hierarchically on paragraphs and sentences."
    )
    doc = Document(page_content=sample_text, metadata={"doc_id": "test_doc_01", "source": "test.txt"})

    chunks = chunk_documents([doc], strategy="recursive", chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 0
    assert all(isinstance(c, Document) for c in chunks)
    assert chunks[0].metadata["doc_id"] == "test_doc_01"
