"""Dedicated reliability test suite covering failure recovery, snapshot preservation, and rollbacks."""
import pytest
from langchain_core.documents import Document

from src.evaluation.offline import HashEmbeddings
from src.ingestion.ingestion_service import DocumentRegistry, IngestionService
from src.retrieval.index_manager import IndexManager
from src.retrieval.retrieval_runtime import RetrievalRuntime
from src.retrieval.vector_store import get_vector_store
from src.runtime.errors import IngestionError


def test_failed_index_refresh_preserves_old_snapshot():
    index_mgr = IndexManager()
    d1 = Document(page_content="Document 1 content", metadata={"chunk_id": "c1", "doc_id": "d1", "version_state": "active"})

    # 1. Build healthy initial snapshot
    snap1 = index_mgr.rebuild([d1])
    assert snap1.chunk_count == 1
    assert snap1.version == 1

    # 2. Simulate build failure
    class BrokenStore:
        pass

    with pytest.raises(Exception):
        index_mgr.rebuild([Document(page_content="Bad", metadata={})])

    # 3. Old snapshot must still be active and intact!
    current = index_mgr.snapshot
    assert current.version == 1
    assert current.chunk_count == 1
    assert current.documents[0].metadata["chunk_id"] == "c1"


def test_ingestion_rollback_on_failed_embedding(tmp_path):
    class FailingEmbedder:
        def embed_documents(self, texts):
            raise RuntimeError("Provider connection timeout during embedding")

    embeddings = HashEmbeddings()
    store = get_vector_store(embeddings=embeddings, use_in_memory=True)
    index_mgr = IndexManager()
    registry = DocumentRegistry(tmp_path / "reg.json")

    # Ingestion service with failing embedder
    failing_service = IngestionService(
        store,
        embeddings=FailingEmbedder(),
        index_manager=index_mgr,
        registry=registry,
    )

    with pytest.raises(IngestionError) as exc_info:
        failing_service.ingest_text("Test content", doc_id="doc-rollback-1")

    assert "embedding_failed" in exc_info.value.code or "Provider connection timeout" in str(exc_info.value)

    # Verify no active or staging chunks leaked into store
    runtime = RetrievalRuntime(index_manager=index_mgr)
    runtime.ensure_index(store)
    assert runtime.status()["chunk_count"] == 0


def test_empty_retrieval_index_is_valid():
    index_mgr = IndexManager()
    # Building index with 0 documents (e.g. after deleting everything) should succeed with chunk_count=0
    snap = index_mgr.rebuild([])
    assert snap.chunk_count == 0
    results = snap.search("query", k=5)
    assert results == []
