from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from src.config import settings

try:
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
except (ImportError, OSError):
    try:
        from langchain.retrievers import ContextualCompressionRetriever
    except (ImportError, OSError):
        from langchain.retrievers.document_compressors import ContextualCompressionRetriever

def build_reranked_retriever(
    base_retriever: BaseRetriever,
    model_name: Optional[str] = None,
    top_n: Optional[int] = None
) -> BaseRetriever:
    """
    Wraps a LangChain BaseRetriever with a CrossEncoder reranking compressor.
    """
    m_name = model_name or settings.RERANKER_MODEL
    top_k = top_n or settings.TOP_K_RERANK

    try:
        try:
            from langchain_community.document_compressors import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            model = HuggingFaceCrossEncoder(model_name=m_name)
            compressor = CrossEncoderReranker(model=model, top_n=top_k)
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
        except (Exception, OSError) as inner_err:
            print(f"[Reranker] Warning initializing HuggingFaceCrossEncoder ({inner_err}). Returning base retriever.")
            return base_retriever
    except (Exception, OSError) as e:
        print(f"[Reranker] Warning loading CrossEncoderReranker ({e}). Returning base retriever.")
        return base_retriever

