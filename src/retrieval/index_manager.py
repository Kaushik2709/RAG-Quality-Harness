"""Versioned, immutable retrieval index snapshots.

The previous request path called `get_indexed_documents(store)`, scrolling the
whole collection, and rebuilt BM25 on every query. This module builds that
index once per corpus revision, keeps snapshots immutable, and atomically swaps
them. Readers never take the write lock, so queries are not blocked while a new
snapshot is constructed.

A failed build leaves the previous snapshot active. If a deletion cannot be
reflected in a new snapshot, the index is invalidated (fail closed) rather than
continuing to serve documents the owner asked to delete.
"""
from __future__ import annotations

import datetime
import hashlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from src.config import settings
from src.runtime.errors import IndexBuildError, IndexUnavailable

_TOKEN_RE = re.compile(r"[\w-]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def corpus_fingerprint(documents: Sequence[Document]) -> str:
    """Stable SHA-256 over (chunk id, content hash), order independent."""
    digest = hashlib.sha256()
    rows = []
    for doc in documents:
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
    store_id: Optional[int] = field(default=None, compare=False)
    scorer: Optional[Callable[[List[str]], Sequence[float]]] = field(default=None, compare=False)

    def search(self, query: str, k: int, predicate: Optional[Callable[[Document], bool]] = None) -> List[Tuple[Document, float]]:
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
    """Corpus-scoped snapshot manager.

    One manager can serve multiple stores (the API has one, evaluation builds
    isolated ones). A snapshot is reused only while the store identity and the
    caller-provided corpus fingerprint match; otherwise an explicit rebuild is
    required.
    """

    def __init__(self, *, max_chunks: int | None = None):
        self.max_chunks = max_chunks or settings.MAX_INDEX_CHUNKS
        self._lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._snapshot: Optional[IndexSnapshot] = None
        self._version = 0
        self._invalid_reason: Optional[str] = None
        self._last_error: Optional[str] = None
        self._last_attempt_at: Optional[str] = None
        self._builds = 0
        self._failures = 0
        self._refresh_ms: Optional[float] = None
        self.last_build_documents_loaded = 0

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    @property
    def snapshot(self) -> Optional[IndexSnapshot]:
        with self._lock:
            if self._invalid_reason:
                raise IndexUnavailable(f"index invalidated: {self._invalid_reason}", component="sparse_index")
            return self._snapshot

    def require_snapshot(self) -> IndexSnapshot:
        snapshot = self.snapshot
        if snapshot is None:
            raise IndexUnavailable("retrieval index has not been built", component="sparse_index")
        return snapshot

    def ensure(self, documents: Sequence[Document], *, store: Any = None, fingerprint: str | None = None) -> IndexSnapshot:
        """Reuse an existing snapshot; build only when the corpus differs.

        This is the normal query path. It performs no collection scroll when a
        matching snapshot exists, which is the core regression this module fixes.
        """
        fp = fingerprint or corpus_fingerprint(documents)
        store_id = id(store) if store is not None else None
        with self._lock:
            snapshot = self._snapshot
            invalid = self._invalid_reason
        if invalid:
            raise IndexUnavailable(f"index invalidated: {invalid}", component="sparse_index")
        if snapshot and snapshot.corpus_fingerprint == fp and (store is None or snapshot.store_id == store_id):
            return snapshot
        # Double-check under the build lock. Concurrent first requests would
        # otherwise each build the same snapshot after waiting for the lock.
        with self._build_lock:
            with self._lock:
                snapshot = self._snapshot
                invalid = self._invalid_reason
            if invalid:
                raise IndexUnavailable(f"index invalidated: {invalid}", component="sparse_index")
            if snapshot and snapshot.corpus_fingerprint == fp and (store is None or snapshot.store_id == store_id):
                return snapshot
            return self._rebuild_locked(documents, store=store, fingerprint=fp)

    def rebuild(self, documents: Sequence[Document], *, store: Any = None, fingerprint: str | None = None) -> IndexSnapshot:
        """Build outside the read lock, then swap atomically.

        Serialised with `_build_lock` so concurrent ingestion events do not
        stampede model/index construction. Regular readers are never blocked by
        this lock.
        """
        with self._build_lock:
            return self._rebuild_locked(documents, store=store, fingerprint=fingerprint)

    def _rebuild_locked(
        self, documents: Sequence[Document], *, store: Any = None, fingerprint: str | None = None
    ) -> IndexSnapshot:
        started = time.perf_counter()
        fp = fingerprint or corpus_fingerprint(documents)
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
            if corpus_fingerprint(docs) != fp:
                raise IndexBuildError("corpus changed while the index was being built", component="sparse_index")
            try:
                from rank_bm25 import BM25Okapi
            except ImportError as exc:  # pragma: no cover - dependency is pinned offline
                raise IndexBuildError("rank_bm25 is not installed", component="sparse_index") from exc
            # An empty corpus is a valid state (all documents deleted), not a
            # failed index build.
            bm25 = BM25Okapi([tokenize(d.page_content) or ["__empty__"] for d in docs]) if docs else None
            elapsed = (time.perf_counter() - started) * 1000
            with self._lock:
                self._version += 1
                snapshot = IndexSnapshot(
                    version=self._version,
                    corpus_fingerprint=fp,
                    documents=docs,
                    bm25=bm25,
                    built_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    build_ms=elapsed,
                    chunk_count=len(docs),
                    store_id=id(store) if store is not None else None,
                )
                self._snapshot = snapshot
                self._invalid_reason = None
                self._last_error = None
                self._refresh_ms = elapsed
                return snapshot
        except Exception as exc:
            with self._lock:
                self._failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, IndexBuildError):
                raise
            raise IndexBuildError(f"index rebuild failed: {type(exc).__name__}: {exc}", component="sparse_index") from exc

    def invalidate(self, reason: str) -> None:
        """Stop serving a snapshot that may contain deleted or unauthorized data."""
        with self._lock:
            self._invalid_reason = reason
            self._last_error = reason

    def reset(self) -> None:
        with self._lock:
            self._snapshot = None
            self._invalid_reason = None
            self._last_error = None
            self._version = 0
            self._builds = 0
            self._failures = 0
            self._refresh_ms = None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            snapshot = self._snapshot
            return {
                "status": "invalidated" if self._invalid_reason else ("ready" if snapshot else "not_built"),
                "version": snapshot.version if snapshot else None,
                "corpus_fingerprint": snapshot.corpus_fingerprint if snapshot else None,
                "chunk_count": snapshot.chunk_count if snapshot else 0,
                "built_at": snapshot.built_at if snapshot else None,
                "build_ms": round(snapshot.build_ms, 2) if snapshot else None,
                "builds": self._builds,
                "failures": self._failures,
                "last_attempt_at": self._last_attempt_at,
                "last_error": self._last_error,
                "invalid_reason": self._invalid_reason,
                "refresh_ms": round(self._refresh_ms, 2) if self._refresh_ms is not None else None,
                "max_chunks": self.max_chunks,
            }

    @staticmethod
    def _immutable_copy(documents: Iterable[Document]) -> Iterable[Document]:
        for doc in documents:
            yield Document(page_content=doc.page_content, metadata=dict(doc.metadata))
