"""Immutable index snapshot lifecycle and concurrency tests."""
from __future__ import annotations

import threading

import pytest
from langchain_core.documents import Document

from src.retrieval.index_manager import IndexManager, corpus_fingerprint
from src.retrieval.retrieval_runtime import RetrievalRuntime
from src.runtime.errors import IndexBuildError, IndexUnavailable


def _docs(prefix: str = "doc", count: int = 3):
    return [
        Document(
            page_content=f"{prefix} retention policy number {index}",
            metadata={"chunk_id": f"{prefix}::c{index}", "doc_id": prefix, "tenant_id": "t1"},
        )
        for index in range(count)
    ]


def test_fingerprint_is_order_independent_and_content_sensitive():
    docs = _docs()
    assert corpus_fingerprint(docs) == corpus_fingerprint(list(reversed(docs)))
    changed = _docs()
    changed[0] = Document(page_content="different text", metadata=dict(changed[0].metadata))
    assert corpus_fingerprint(changed) != corpus_fingerprint(docs)


def test_snapshot_search_filters_and_bounds():
    manager = IndexManager()
    snapshot = manager.rebuild(_docs())
    results = snapshot.search("retention", k=2, predicate=lambda d: d.metadata["doc_id"] == "doc")
    assert len(results) == 2
    assert all(score > 0 for _, score in results)
    assert snapshot.search("retention", k=0) == []


def test_ensure_reuses_snapshot_without_rebuilding():
    manager = IndexManager()
    docs = _docs()
    marker = object()
    first = manager.ensure(docs, store=marker)
    builds = manager.status()["builds"]
    second = manager.ensure(docs, store=marker)
    assert first is second
    assert manager.status()["builds"] == builds
    # A different store identity must not reuse another store's snapshot.
    other = manager.ensure(docs, store=object())
    assert other is not first
    assert manager.status()["builds"] == builds + 1


def test_runtime_reads_collection_only_once(monkeypatch):
    runtime = RetrievalRuntime()
    docs = _docs()
    calls = []

    def loader(store):
        calls.append(1)
        return docs

    monkeypatch.setattr("src.retrieval.retrieval_runtime.get_retrieval_documents", loader)
    store = object()
    first = runtime.ensure_index(store)
    second = runtime.ensure_index(store)
    assert first is second
    assert len(calls) == 1


def test_concurrent_first_requests_build_once():
    manager = IndexManager()
    docs = _docs(count=20)
    store = object()
    barrier = threading.Barrier(8)
    versions = []

    def worker():
        barrier.wait()
        versions.append(manager.ensure(docs, store=store).version)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert versions == [1] * 8
    assert manager.status()["builds"] == 1


def test_invalidated_index_fails_closed():
    manager = IndexManager()
    manager.rebuild(_docs())
    manager.invalidate("deleted document could not be refreshed")
    with pytest.raises(IndexUnavailable):
        manager.snapshot
    with pytest.raises(IndexUnavailable):
        manager.ensure(_docs())


def test_empty_index_is_valid_but_duplicate_and_oversized_are_rejected():
    manager = IndexManager()
    empty = manager.rebuild([])
    assert empty.chunk_count == 0
    assert empty.search("anything", 5) == []
    duplicates = _docs() + [_docs()[0]]
    with pytest.raises(IndexBuildError):
        manager.rebuild(duplicates)
    with pytest.raises(IndexBuildError):
        IndexManager(max_chunks=1).rebuild(_docs(count=2))
    assert manager.status()["failures"] == 1


def test_chunks_require_ids():
    with pytest.raises(IndexBuildError):
        IndexManager().rebuild([Document(page_content="text", metadata={})])
