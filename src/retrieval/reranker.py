"""Cross-encoder reranking with explicit failure semantics.

`build_cross_encoder` is the lifecycle factory used by `ModelRegistry`; it loads
weights once. `build_reranked_retriever` is kept for evaluation and backward
compatibility. Neither silently claims that reranking happened: a failed load
either raises (strict) or returns the untouched input with a recorded reason.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import settings


class CrossEncoderScorer:
    """Thin adapter that keeps provider-specific model objects out of callers."""

    def __init__(self, model: Any, model_id: str):
        self.model = model
        self.model_id = model_id

    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        return [float(value) for value in self.model.score(list(pairs))]


def build_cross_encoder(model_name: Optional[str] = None) -> CrossEncoderScorer:
    model_id = model_name or settings.RERANKER_MODEL
    try:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    except ImportError as exc:
        raise RuntimeError("langchain-community / sentence-transformers are not installed") from exc
    model = HuggingFaceCrossEncoder(model_name=model_id)
    return CrossEncoderScorer(model=model, model_id=model_id)


def rerank_documents(
    reranker: Any,
    query: str,
    documents: Sequence[Document],
    top_n: int,
) -> Tuple[List[Document], dict]:
    """Return ranked documents plus a score map. Empty input is not an error."""
    docs = list(documents)
    if not docs or top_n <= 0:
        return [], {}
    scores = reranker.score([(query, doc.page_content) for doc in docs])
    if len(scores) != len(docs):
        raise RuntimeError("cross-encoder returned an unexpected number of scores")
    order = sorted(
        range(len(docs)),
        key=lambda idx: (-scores[idx], str(docs[idx].metadata.get("chunk_id", ""))),
    )
    ranked = [docs[idx] for idx in order[:top_n]]
    score_map = {
        str(docs[idx].metadata.get("chunk_id", idx)): float(scores[idx]) for idx in order
    }
    return ranked, score_map


class RerankerCompressor:
    def __init__(self, reranker: Any, top_n: int):
        self.reranker = reranker
        self.top_n = top_n
        self.last_scores: dict = {}

    def compress_documents(self, documents: Sequence[Document], query: str, callbacks=None) -> List[Document]:
        ranked, scores = rerank_documents(self.reranker, query, documents, self.top_n)
        self.last_scores = scores
        return ranked


class RerankedRetriever:
    """Small BaseRetriever-compatible shim used by evaluation and legacy callers."""

    def __init__(self, base_retriever: BaseRetriever, compressor: RerankerCompressor):
        self.base_retriever = base_retriever
        self.base_compressor = compressor

    def invoke(self, query: str, config=None, **kwargs):
        return self.base_compressor.compress_documents(self.base_retriever.invoke(query), query)


def build_reranked_retriever(
    base_retriever: BaseRetriever,
    model_name: Optional[str] = None,
    top_n: Optional[int] = None,
    strict: bool = False,
) -> Any:
    top_k = top_n or settings.TOP_K_RERANK
    try:
        scorer = build_cross_encoder(model_name)
        return RerankedRetriever(base_retriever, RerankerCompressor(scorer, top_k))
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise RuntimeError(f"cross-encoder unavailable: {type(exc).__name__}: {exc}") from exc
        # Legacy non-strict mode is only used by callers that explicitly opt in.
        # The returned object exposes why reranking did not occur.
        return UnavailableReranker(base_retriever, reason=f"{type(exc).__name__}: {exc}")


class UnavailableReranker:
    def __init__(self, base_retriever: BaseRetriever, reason: str):
        self.base_retriever = base_retriever
        self.reason = reason
        self.rerank_status = "unavailable"

    def invoke(self, query: str, config=None, **kwargs):
        return self.base_retriever.invoke(query)
