
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever

try:
    from langchain_classic.retrievers.ensemble import EnsembleRetriever
except (ImportError, OSError):
    try:
        from langchain.retrievers import EnsembleRetriever
    except (ImportError, OSError):
        from langchain_community.retrievers import EnsembleRetriever

from src.retrieval.vector_store import get_vector_store
from src.ingestion.embedder import get_embeddings_model
from src.config import settings

def build_retriever(
    vectorstore=None,
    documents: Optional[List[Document]] = None,
    top_k: Optional[int] = None,
    use_hybrid: Optional[bool] = None
) -> BaseRetriever:
    """
    Returns a standard LangChain BaseRetriever (EnsembleRetriever for hybrid search).
    """
    k = top_k or settings.TOP_K_RETRIEVAL
    hybrid = use_hybrid if use_hybrid is not None else settings.USE_HYBRID_SEARCH

    vstore = vectorstore or get_vector_store()
    dense_retriever = vstore.as_retriever(search_kwargs={"k": k})

    if not hybrid or not documents:
        return dense_retriever

    try:
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k

        # LangChain EnsembleRetriever with 60% dense / 40% BM25 weighting
        ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[0.6, 0.4]
        )
        return ensemble_retriever
    except Exception as e:
        print(f"[Retriever] Hybrid BM25 error: {e}. Falling back to dense vector retriever.")
        return dense_retriever
