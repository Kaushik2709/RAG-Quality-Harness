"""Request-path retrieval orchestration.

Responsibilities:
* reuse the active immutable index snapshot across requests;
* never scroll the collection during a normal query;
* keep dense and sparse retrieval independently observable;
* enforce tenant/version filters at the store/snapshot boundary;
* return stage scores so fusion and reranking can be audited.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain_core.documents import Document

from src.config import settings
from src.retrieval.fusion import fused_scores, reciprocal_rank_fusion
from src.retrieval.index_manager import IndexManager, IndexSnapshot, corpus_fingerprint
from src.retrieval.vector_store import get_retrieval_documents, search_dense


@dataclass
class RetrievalResult:
    documents: List[Document]
    mode: str
    index_version: Optional[int] = None
    corpus_fingerprint: Optional[str] = None
    dense_scores: Dict[str, float] = field(default_factory=dict)
    sparse_scores: Dict[str, float] = field(default_factory=dict)
    fused_scores: Dict[str, float] = field(default_factory=dict)
    stage_latency_ms: Dict[str, float] = field(default_factory=dict)
    candidate_count: int = 0
    degraded_reasons: List[str] = field(default_factory=list)

    def metadata(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "index_version": self.index_version,
            "corpus_fingerprint": self.corpus_fingerprint,
            "candidate_count": self.candidate_count,
            "stage_latency_ms": {k: round(v, 3) for k, v in self.stage_latency_ms.items()},
            "degraded_reasons": list(self.degraded_reasons),
        }


def tenant_predicate(tenant_id: str) -> Callable[[Document], bool]:
    """Legacy documents without tenant metadata belong to the default tenant;
    missing version_state is treated as active for backward compatibility with
    pre-versioning collections. New writes always set both explicitly."""
    def predicate(doc: Document) -> bool:
        return (
            doc.metadata.get("tenant_id", "default") == tenant_id
            and doc.metadata.get("version_state", "active") == "active"
        )

    return predicate


class RetrievalRuntime:
    def __init__(self, index_manager: Optional[IndexManager] = None):
        self.index_manager = index_manager or IndexManager()

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------
    def refresh_from_store(self, store, documents: Optional[Sequence[Document]] = None) -> IndexSnapshot:
        """Explicit refresh. This is the only path that may scroll the corpus."""
        if documents is None:
            documents = get_retrieval_documents(store)
        return self.index_manager.rebuild(documents, store=store)

    def ensure_index(self, store, documents: Optional[Sequence[Document]] = None) -> IndexSnapshot:
        """Return a snapshot for this store, building once if necessary.

        If a snapshot for the same store already exists it is reused without a
        collection read, even when an ingest job adds documents. Ingestion is
        responsible for calling `refresh_from_store` after activation.
        """
        if documents is not None:
            return self.index_manager.ensure(documents, store=store)
        start = time.perf_counter()
        snapshot = self.index_manager.snapshot
        if snapshot is not None and snapshot.store_id == id(store):
            return snapshot
        loaded = get_retrieval_documents(store)
        self.index_manager.last_build_documents_loaded = len(loaded)
        built = self.index_manager.rebuild(loaded, store=store)
        built_ms = (time.perf_counter() - start) * 1000
        # build_ms only covers tokenization/BM25; keep the store read visible.
        object.__setattr__(built, "build_ms", built_ms)
        return built

    def invalidate(self, reason: str) -> None:
        self.index_manager.invalidate(reason)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        store,
        query: str,
        *,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        tenant_id: str = "default",
        dense_k: Optional[int] = None,
    ) -> RetrievalResult:
        k = top_k if top_k is not None else settings.TOP_K_RETRIEVAL
        selected_mode = mode or settings.RETRIEVAL_MODE
        if k <= 0:
            raise ValueError("top_k must be positive")
        if selected_mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"unknown retrieval mode: {selected_mode}")

        snapshot = self.ensure_index(store)
        predicate = tenant_predicate(tenant_id)
        latencies: Dict[str, float] = {}
        dense: List[Document] = []
        sparse: List[Document] = []
        dense_score_map: Dict[str, float] = {}
        sparse_score_map: Dict[str, float] = {}

        if selected_mode in {"dense", "hybrid"}:
            start = time.perf_counter()
            pairs = search_dense(
                store,
                query,
                dense_k or k,
                metadata_filters={} if not self._is_qdrant(store) else {
                    "tenant_id": tenant_id,
                    "version_state": "active",
                },
                predicate=predicate if not self._is_qdrant(store) else None,
            )
            dense = [doc for doc, _ in pairs]
            dense_score_map = {
                str(doc.metadata.get("chunk_id", idx)): float(score)
                for idx, (doc, score) in enumerate(pairs)
            }
            latencies["dense"] = (time.perf_counter() - start) * 1000

        if selected_mode in {"bm25", "hybrid"}:
            start = time.perf_counter()
            sparse_pairs = snapshot.search(query, k, predicate=predicate)
            sparse = [doc for doc, _ in sparse_pairs]
            sparse_score_map = {
                str(doc.metadata.get("chunk_id", idx)): float(score)
                for idx, (doc, score) in enumerate(sparse_pairs)
            }
            latencies["bm25"] = (time.perf_counter() - start) * 1000

        if selected_mode == "dense":
            documents = dense[:k]
            fused = {}
        elif selected_mode == "bm25":
            documents = sparse[:k]
            fused = {}
        else:
            start = time.perf_counter()
            ranked_lists = [
                (settings.FUSION_DENSE_WEIGHT, dense),
                (settings.FUSION_SPARSE_WEIGHT, sparse),
            ]
            documents = reciprocal_rank_fusion(ranked_lists, rrf_k=settings.FUSION_RRF_K, limit=k)
            fused = fused_scores(ranked_lists, rrf_k=settings.FUSION_RRF_K)
            latencies["fusion"] = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            documents=documents,
            mode=selected_mode,
            index_version=snapshot.version,
            corpus_fingerprint=snapshot.corpus_fingerprint,
            dense_scores=dense_score_map,
            sparse_scores=sparse_score_map,
            fused_scores=fused,
            stage_latency_ms=latencies,
            candidate_count=len(documents),
        )

    def warm_index(self, store, documents: Optional[Sequence[Document]] = None) -> Dict[str, Any]:
        try:
            snapshot = self.refresh_from_store(store, documents)
            return {"status": "ready", "index_version": snapshot.version, "chunks": snapshot.chunk_count}
        except Exception as exc:  # noqa: BLE001 - exposed as readiness, never hidden
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _is_qdrant(store) -> bool:
        from langchain_core.vectorstores import InMemoryVectorStore

        return not isinstance(store, InMemoryVectorStore)

    def status(self) -> Dict[str, Any]:
        return self.index_manager.status()


_default_runtime: Optional[RetrievalRuntime] = None
_default_lock = __import__("threading").Lock()


def get_retrieval_runtime() -> RetrievalRuntime:
    global _default_runtime
    if _default_runtime is None:
        with _default_lock:
            if _default_runtime is None:
                _default_runtime = RetrievalRuntime()
    return _default_runtime


def get_default_index_manager() -> IndexManager:
    return get_retrieval_runtime().index_manager


def reset_retrieval_runtime() -> None:
    global _default_runtime
    with _default_lock:
        if _default_runtime is not None:
            _default_runtime.index_manager.reset()
        _default_runtime = None
