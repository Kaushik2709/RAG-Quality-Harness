"""Explicit corpus retrieval and identity-based reciprocal rank fusion."""
import re
from time import perf_counter
from typing import Any, List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from src.config import settings


def tokenize(text):
    return re.findall(r"[\w-]+", text.lower())


class FixedRetriever(BaseRetriever):
    """Pass existing candidates to a compressor without another retrieval."""
    documents: List[Document]

    def _get_relevant_documents(self, query, *, run_manager):
        return list(self.documents)


class CorpusRetriever(BaseRetriever):
    vectorstore: Any = None
    documents: List[Document]
    mode: str
    k: int
    bm25: Any = None
    last_latency_ms: dict = {}

    def _get_relevant_documents(self, query, *, run_manager):
        self.last_latency_ms = {}
        start = perf_counter()
        dense = self.vectorstore.similarity_search(query, k=self.k) if self.mode != "bm25" else []
        if self.mode != "bm25":
            self.last_latency_ms["dense"] = (perf_counter()-start)*1000
        start = perf_counter()
        sparse = []
        if self.mode != "dense" and self.documents:
            scores = self.bm25.get_scores(tokenize(query))
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.documents[i].metadata["chunk_id"]))
            sparse = [self.documents[i] for i in order[:self.k] if scores[i] > 0]
        if self.mode != "dense":
            self.last_latency_ms["bm25"] = (perf_counter()-start)*1000
        if self.mode == "dense":
            return dense[:self.k]
        if self.mode == "bm25":
            return sparse
        start = perf_counter()
        fused, docs = {}, {}
        for weight, ranked in ((0.6, dense), (0.4, sparse)):
            seen = set()
            for rank, doc in enumerate(ranked, 1):
                key = doc.metadata["chunk_id"]
                if key in seen:
                    continue
                seen.add(key)
                docs[key] = doc
                fused[key] = fused.get(key, 0) + weight / (60 + rank)
        result = [docs[key] for key in sorted(fused, key=lambda key: (-fused[key], key))[:self.k]]
        self.last_latency_ms["fusion"] = (perf_counter()-start)*1000
        return result


def build_retriever(vectorstore=None, documents: Optional[List[Document]] = None,
                    top_k=None, use_hybrid=None, mode=None) -> BaseRetriever:
    k = top_k if top_k is not None else settings.TOP_K_RETRIEVAL
    if k <= 0:
        raise ValueError("top_k must be positive")
    mode = mode or ("hybrid" if (use_hybrid if use_hybrid is not None else settings.USE_HYBRID_SEARCH) else "dense")
    if mode not in {"dense", "bm25", "hybrid"}:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    if vectorstore is None and mode != "bm25":
        from src.retrieval.vector_store import get_vector_store
        vectorstore = get_vector_store()
    if documents is None and mode != "dense":
        from src.retrieval.vector_store import get_indexed_documents
        documents = get_indexed_documents(vectorstore)
    documents = list(documents or [])
    ids = [d.metadata.get("chunk_id") for d in documents]
    if any(not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Corpus requires unique nonempty chunk_id values")
    bm25 = None
    if mode != "dense" and documents:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([tokenize(d.page_content) or ["__empty__"] for d in documents])
    return CorpusRetriever(vectorstore=vectorstore, documents=documents, mode=mode, k=k, bm25=bm25)
