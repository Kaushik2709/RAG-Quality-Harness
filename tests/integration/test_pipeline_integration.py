"""Integration test exercising the full offline RAG pipeline end-to-end."""
import pytest
from langchain_core.documents import Document

from src.evaluation.offline import HashEmbeddings
from src.generation.citations import StructuredAnswer
from src.generation.generator import GenerationResult
from src.ingestion.ingestion_service import DocumentRegistry, IngestionService
from src.pipeline.graph import compile_rag_graph
from src.retrieval.index_manager import IndexManager
from src.retrieval.retrieval_runtime import RetrievalRuntime
from src.retrieval.vector_store import get_vector_store


class _MockModelRegistry:
    def status(self):
        return {"models": {"llm": {"model_id": "test-mock-llm"}, "reranker": {"model_id": "none"}}}

    def get_llm(self):
        return object()

    def get_reranker(self):
        return None


def test_full_rag_pipeline_integration(tmp_path, monkeypatch):
    # 1. Initialize store & runtime
    embeddings = HashEmbeddings()
    store = get_vector_store(embeddings=embeddings, use_in_memory=True)
    index_mgr = IndexManager()
    runtime = RetrievalRuntime(index_manager=index_mgr)

    registry = DocumentRegistry(tmp_path / "registry.json")
    ingestion = IngestionService(store, embeddings=embeddings, index_manager=index_mgr, registry=registry)

    # 2. Ingest document
    content = "The Nova service authentication timeout is 45 seconds for all production environments."
    res = ingestion.ingest_text(content, doc_id="nova-policy", tenant_id="tenant-alpha")
    assert res.status == "created"
    assert res.activated is True
    assert res.chunks_created == 1

    # 3. Verify index snapshot updated
    runtime.ensure_index(store)
    status = runtime.status()
    assert status["chunk_count"] == 1
    assert status["status"] == "ready"

    # 4. Search via RetrievalRuntime with tenant isolation
    search_res = runtime.search(store, "Nova authentication timeout", top_k=5, mode="hybrid", tenant_id="tenant-alpha")
    assert len(search_res.documents) == 1
    assert "45 seconds" in search_res.documents[0].page_content

    # Tenant mismatch returns 0
    search_wrong_tenant = runtime.search(store, "Nova authentication timeout", top_k=5, mode="hybrid", tenant_id="tenant-beta")
    assert len(search_wrong_tenant.documents) == 0

    chunk_id = search_res.documents[0].metadata["chunk_id"]

    # 5. Compile graph with injected store and runtime and mock generation
    monkeypatch.setattr("src.pipeline.graph.get_model_registry", lambda: _MockModelRegistry())
    monkeypatch.setattr(
        "src.pipeline.graph.generate_structured",
        lambda query, context, chat_model=None: GenerationResult(
            structured=StructuredAnswer(
                answer="The Nova service timeout is 45 seconds.",
                citations=[{"chunk_id": chunk_id, "quote": "timeout is 45 seconds"}],
            ),
            model_id="test-mock-llm",
            usage={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
            usage_source="provider",
        ),
    )

    graph = compile_rag_graph(vector_store=store, retrieval_runtime=runtime)
    initial_state = {
        "query": "What is the Nova service timeout?",
        "query_id": "test-int-001",
        "tenant_id": "tenant-alpha",
        "top_k": 5,
        "use_hybrid": True,
    }
    final_state = graph.invoke(initial_state)

    # Retrieval, rerank, generation, citation validation nodes executed
    assert "documents" in final_state
    assert len(final_state["documents"]) == 1
    assert final_state["retrieval"]["mode"] == "hybrid"
    assert "context" in final_state
    assert "The Nova service timeout is 45 seconds." in final_state["generation"]
    assert final_state["citation_validation"]["status"] == "validated"
    assert final_state["citation_validation"]["citation_precision"] == 1.0
