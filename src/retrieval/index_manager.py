"""Versioned, tenant-isolated immutable retrieval index snapshots.

The BM25 index is constructed per tenant revision, keeping snapshots immutable
and atomically swapped. Readers never take the write lock, so queries are not
blocked while a new snapshot is constructed.

Tenant isolation guarantee:
Each tenant maintains a distinct BM25 index and corpus statistics. Documents
from Tenant B do not alter IDF or term frequency weights for Tenant A.
Deletion in Tenant A invalidates only Tenant A's snapshot.
"""
from __future__ import annotations

import datetime
import hashlib
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from src.config import settings
from src.runtime.errors import IndexBuildError, IndexUnavailable

_TOKEN_RE = re.compile(r"[\w-]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def resolve_tenant_id(documents: Sequence[Document], tenant_id: Optional[str] = None) -> str:
    if tenant_id is not None and tenant_id != "default":
        return tenant_id
    if documents:
        first_tenant = documents[0].metadata.get("tenant_id")
        if first_tenant and all(d.metadata.get("tenant_id") == first_tenant for d in documents):
            return str(first_tenant)
    return tenant_id or "default"


def corpus_fingerprint(documents: Sequence[Document], tenant_id: Optional[str] = None) -> str:
    """Stable SHA-256 over tenant scope and (chunk id, content hash), order independent."""
    resolved_tenant = resolve_tenant_id(documents, tenant_id)
    digest = hashlib.sha256()
    digest.update(f"tenant:{resolved_tenant}\n".encode("utf-8"))
    rows = []
    for doc in documents:
        t_id = doc.metadata.get("tenant_id", "default")
        v_state = doc.metadata.get("version_state", "active")
        # Include documents matching tenant_id and active state (or unversioned legacy documents)
        if t_id == resolved_tenant and v_state == "active":
            chunk_id = str(doc.metadata.get("chunk_id", ""))
            content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
            rows.append((chunk_id, content_hash))
    for chunk_id, content_hash in sorted(rows):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class IndexSnapshot:
    version: int
    corpus_fingerprint: str
    documents: Tuple[Document, ...]
    bm25: Any
    built_at: str
    build_ms: float
    chunk_count: int
    tenant_id: str = "default"
    store_id: Optional[int] = field(default=None, compare=False)
    scorer: Optional[Callable[[List[str]], Sequence[float]]] = field(default=None, compare=False)

    def search(
        self,
        query: str,
        k: int,
        predicate: Optional[Callable[[Document], bool]] = None,
    ) -> List[Tuple[Document, float]]:
        return sparse_search(self.documents, self.bm25, query, k, predicate)


def sparse_search(
    documents: Sequence[Document],
    bm25: Any,
    query: str,
    k: int,
    predicate: Optional[Callable[[Document], bool]] = None,
) -> List[Tuple[Document, float]]:
    """Rank a sparse candidate set, including the all-negative-IDF case.

    `rank_bm25` floors negative IDFs at `epsilon * average_idf`. On a small
    corpus where the query terms occur in most documents the whole score vector
    can be negative; selecting only positive scores would return nothing even
    though every document matches. In that case we fall back to deterministic
    query/document token overlap. A query with no lexical overlap still returns
    no results.
    """
    if bm25 is None or k <= 0 or not documents:
        return []
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    ranked = sorted(
        range(len(scores)),
        key=lambda idx: (-scores[idx], str(documents[idx].metadata.get("chunk_id", ""))),
    )
    positive = [idx for idx in ranked if scores[idx] > 0]
    if positive:
        pool = [(idx, float(scores[idx])) for idx in positive]
    else:
        query_terms = set(tokens)
        overlaps = []
        for idx, doc in enumerate(documents):
            overlap = len(query_terms & set(tokenize(doc.page_content)))
            if overlap:
                overlaps.append((idx, float(overlap)))
        overlaps.sort(key=lambda item: (-item[1], str(documents[item[0]].metadata.get("chunk_id", ""))))
        pool = overlaps
    results: List[Tuple[Document, float]] = []
    for idx, score in pool:
        doc = documents[idx]
        if predicate is not None and not predicate(doc):
            continue
        results.append((doc, score))
        if len(results) >= k:
            break
    return results


class IndexManager:
    """Tenant-scoped snapshot manager with immutable per-tenant BM25 indexes.

    Maintains separate immutable IndexSnapshot instances per tenant. Building,
    updating, or invalidating one tenant's sparse index never impacts other tenants.
    """

    def __init__(self, *, max_chunks: int | None = None):
        self.max_chunks = max_chunks or settings.MAX_INDEX_CHUNKS
        self._lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._snapshots: Dict[str, IndexSnapshot] = {}  # tenant_id -> snapshot
        self._versions: Dict[str, int] = defaultdict(int)  # tenant_id -> version
        self._invalid_reasons: Dict[str, Optional[str]] = {}  # tenant_id -> reason
        self._global_invalid_reason: Optional[str] = None
        self._last_error: Optional[str] = None
        self._last_attempt_at: Optional[str] = None
        self._builds = 0
        self._failures = 0
        self._refresh_ms: Dict[str, float] = {}
        self.last_build_documents_loaded = 0

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get_snapshot(self, tenant_id: str = "default") -> Optional[IndexSnapshot]:
        with self._lock:
            if self._global_invalid_reason:
                raise IndexUnavailable(
                    f"all indexes invalidated: {self._global_invalid_reason}", component="sparse_index"
                )
            reason = self._invalid_reasons.get(tenant_id)
            if reason:
                raise IndexUnavailable(
                    f"index for tenant '{tenant_id}' invalidated: {reason}", component="sparse_index"
                )
            return self._snapshots.get(tenant_id)

    @property
    def snapshot(self) -> Optional[IndexSnapshot]:
        """Backward-compatible default tenant snapshot."""
        with self._lock:
            if self._global_invalid_reason:
                raise IndexUnavailable(
                    f"all indexes invalidated: {self._global_invalid_reason}", component="sparse_index"
                )
            snap = self._snapshots.get("default")
            if snap is not None:
                return snap
            if self._snapshots:
                return next(iter(self._snapshots.values()))
            return None

    @property
    def _snapshot(self) -> Optional[IndexSnapshot]:
        """Backward-compatible alias for existing unit tests."""
        return self.snapshot

    def require_snapshot(self, tenant_id: str = "default") -> IndexSnapshot:
        snapshot = self.get_snapshot(tenant_id)
        if snapshot is None:
            raise IndexUnavailable(
                f"retrieval index for tenant '{tenant_id}' has not been built", component="sparse_index"
            )
        return snapshot

    def ensure(
        self,
        documents: Sequence[Document],
        *,
        store: Any = None,
        tenant_id: Optional[str] = None,
        fingerprint: str | None = None,
    ) -> IndexSnapshot:
        """Reuse an existing snapshot for tenant_id; build only when the tenant corpus differs."""
        target_tenant = resolve_tenant_id(documents, tenant_id)
        tenant_docs = [
            d
            for d in documents
            if d.metadata.get("tenant_id", "default") == target_tenant
            and d.metadata.get("version_state", "active") == "active"
        ]
        fp = fingerprint or corpus_fingerprint(tenant_docs, tenant_id=target_tenant)
        store_id = id(store) if store is not None else None

        with self._lock:
            if self._global_invalid_reason:
                raise IndexUnavailable(
                    f"index invalidated: {self._global_invalid_reason}", component="sparse_index"
                )
            invalid = self._invalid_reasons.get(target_tenant)
            if invalid:
                raise IndexUnavailable(
                    f"index for tenant '{target_tenant}' invalidated: {invalid}", component="sparse_index"
                )
            snapshot = self._snapshots.get(target_tenant)
            if snapshot and snapshot.corpus_fingerprint == fp and (store is None or snapshot.store_id == store_id):
                return snapshot

        # Under the build lock, double check and build
        with self._build_lock:
            with self._lock:
                snapshot = self._snapshots.get(target_tenant)
                invalid = self._invalid_reasons.get(target_tenant)
                if invalid:
                    raise IndexUnavailable(
                        f"index for tenant '{target_tenant}' invalidated: {invalid}", component="sparse_index"
                    )
                if snapshot and snapshot.corpus_fingerprint == fp and (store is None or snapshot.store_id == store_id):
                    return snapshot
            return self._rebuild_locked(tenant_docs, store=store, tenant_id=target_tenant, fingerprint=fp)

    def rebuild(
        self,
        documents: Sequence[Document],
        *,
        store: Any = None,
        tenant_id: Optional[str] = None,
        fingerprint: str | None = None,
    ) -> IndexSnapshot:
        """Build outside the read lock, then swap atomically."""
        target_tenant = resolve_tenant_id(documents, tenant_id)
        tenant_docs = [
            d
            for d in documents
            if d.metadata.get("tenant_id", "default") == target_tenant
            and d.metadata.get("version_state", "active") == "active"
        ]
        with self._build_lock:
            return self._rebuild_locked(tenant_docs, store=store, tenant_id=target_tenant, fingerprint=fingerprint)

    def _rebuild_locked(
        self,
        documents: Sequence[Document],
        *,
        store: Any = None,
        tenant_id: str = "default",
        fingerprint: str | None = None,
    ) -> IndexSnapshot:
        started = time.perf_counter()
        fp = fingerprint or corpus_fingerprint(documents, tenant_id=tenant_id)
        docs = tuple(self._immutable_copy(documents))
        with self._lock:
            self._last_attempt_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._builds += 1
        try:
            if len(docs) > self.max_chunks:
                raise IndexBuildError(
                    f"corpus of {len(docs)} chunks exceeds MAX_INDEX_CHUNKS={self.max_chunks}",
                    component="sparse_index",
                )
            ids = [str(d.metadata.get("chunk_id", "")).strip() for d in docs]
            if any(not chunk_id for chunk_id in ids):
                raise IndexBuildError("every chunk requires a non-empty chunk_id", component="sparse_index")
            if len(set(ids)) != len(ids):
                raise IndexBuildError("chunk_id values must be unique within a corpus", component="sparse_index")
            if corpus_fingerprint(docs, tenant_id=tenant_id) != fp:
                raise IndexBuildError("corpus changed while the index was being built", component="sparse_index")
            try:
                from rank_bm25 import BM25Okapi
            except ImportError as exc:  # pragma: no cover
                raise IndexBuildError("rank_bm25 is not installed", component="sparse_index") from exc

            bm25 = BM25Okapi([tokenize(d.page_content) or ["__empty__"] for d in docs]) if docs else None
            elapsed = (time.perf_counter() - started) * 1000

            with self._lock:
                self._versions[tenant_id] += 1
                version = self._versions[tenant_id]
                snapshot = IndexSnapshot(
                    version=version,
                    corpus_fingerprint=fp,
                    documents=docs,
                    bm25=bm25,
                    built_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    build_ms=elapsed,
                    chunk_count=len(docs),
                    tenant_id=tenant_id,
                    store_id=id(store) if store is not None else None,
                )
                self._snapshots[tenant_id] = snapshot
                self._invalid_reasons.pop(tenant_id, None)
                self._global_invalid_reason = None
                self._last_error = None
                self._refresh_ms[tenant_id] = elapsed
                return snapshot
        except Exception as exc:
            with self._lock:
                self._failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, IndexBuildError):
                raise
            raise IndexBuildError(f"index rebuild failed: {type(exc).__name__}: {exc}", component="sparse_index") from exc

    def invalidate(self, reason: str, tenant_id: Optional[str] = None) -> None:
        """Stop serving snapshot for tenant_id (or all tenants if None)."""
        with self._lock:
            if tenant_id:
                self._invalid_reasons[tenant_id] = reason
                self._snapshots.pop(tenant_id, None)
            else:
                self._global_invalid_reason = reason
                self._snapshots.clear()
            self._last_error = reason

    def reset(self) -> None:
        with self._lock:
            self._snapshots.clear()
            self._versions.clear()
            self._invalid_reasons.clear()
            self._global_invalid_reason = None
            self._last_error = None
            self._builds = 0
            self._failures = 0
            self._refresh_ms.clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def status(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            target_tenant = tenant_id or (
                "default"
                if "default" in self._snapshots
                else (next(iter(self._snapshots.keys())) if self._snapshots else "default")
            )
            snapshot = self._snapshots.get(target_tenant)
            invalid_reason = self._global_invalid_reason or self._invalid_reasons.get(target_tenant)
            return {
                "status": "invalidated" if invalid_reason else ("ready" if snapshot else "not_built"),
                "tenant_id": target_tenant,
                "version": snapshot.version if snapshot else None,
                "corpus_fingerprint": snapshot.corpus_fingerprint if snapshot else None,
                "chunk_count": snapshot.chunk_count if snapshot else 0,
                "built_at": snapshot.built_at if snapshot else None,
                "build_ms": round(snapshot.build_ms, 2) if snapshot else None,
                "builds": self._builds,
                "failures": self._failures,
                "last_attempt_at": self._last_attempt_at,
                "last_error": self._last_error,
                "invalid_reason": invalid_reason,
                "refresh_ms": round(self._refresh_ms.get(target_tenant, 0.0), 2),
                "max_chunks": self.max_chunks,
                "active_tenants": sorted(list(self._snapshots.keys())),
            }

    @staticmethod
    def _immutable_copy(documents: Iterable[Document]) -> Iterable[Document]:
        for doc in documents:
            yield Document(page_content=doc.page_content, metadata=dict(doc.metadata))
