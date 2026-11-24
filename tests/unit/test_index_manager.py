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


def test_dense_only_request_does_not_construct_sparse_index(monkeypatch):
    from langchain_core.vectorstores import InMemoryVectorStore
    from src.evaluation.offline import HashEmbeddings

    store = InMemoryVectorStore(embedding=HashEmbeddings())
    manager = IndexManager()
    runtime = RetrievalRuntime(index_manager=manager)

    # In dense mode, ensure_index should never be called and index remains not_built
    res = runtime.search(store, "policy query", mode="dense", tenant_id="tenant-lazy")
    assert res.mode == "dense"
    assert res.index_version is None
    assert manager.status(tenant_id="tenant-lazy")["status"] == "not_built"


def test_tenant_specific_bm25_scores_independent_of_other_tenants():
    """Verify cross-tenant documents do not influence tenant-specific BM25 statistics."""
    manager = IndexManager()

    doc_a = [
        Document(
            page_content="critical compliance retention document for alpha",
            metadata={"chunk_id": "a-1", "doc_id": "doc-a", "tenant_id": "tenant-a", "version_state": "active"},
        )
    ]
    snap_a_before = manager.rebuild(doc_a, tenant_id="tenant-a")
    score_before = snap_a_before.search("retention", k=1)[0][1]

    # Add 20 documents to Tenant B with heavy repetitions of 'retention'
    docs_b = [
        Document(
            page_content=f"retention retention retention archive policy beta {i}",
            metadata={"chunk_id": f"b-{i}", "doc_id": f"doc-b-{i}", "tenant_id": "tenant-b", "version_state": "active"},
        )
        for i in range(20)
    ]
    # Build Tenant B's index
    snap_b = manager.rebuild(docs_b, tenant_id="tenant-b")
    assert snap_b.tenant_id == "tenant-b"

    # Search Tenant A again - scores MUST be identical because corpus statistics are isolated
    snap_a_after = manager.get_snapshot("tenant-a")
    score_after = snap_a_after.search("retention", k=1)[0][1]
    assert score_before == score_after


def test_concurrent_multi_tenant_requests_maintain_isolated_snapshots():
    manager = IndexManager()
    docs_a = [
        Document(
            page_content=f"alpha document content {i}",
            metadata={"chunk_id": f"alpha-{i}", "doc_id": "doc-a", "tenant_id": "tenant-alpha", "version_state": "active"},
        )
        for i in range(5)
    ]
    docs_b = [
        Document(
            page_content=f"beta document content {i}",
            metadata={"chunk_id": f"beta-{i}", "doc_id": "doc-b", "tenant_id": "tenant-beta", "version_state": "active"},
        )
        for i in range(10)
    ]

    barrier = threading.Barrier(4)
    results = {}

    def worker_alpha():
        barrier.wait()
        snap = manager.ensure(docs_a, tenant_id="tenant-alpha")
        results["alpha"] = snap.chunk_count

    def worker_beta():
        barrier.wait()
        snap = manager.ensure(docs_b, tenant_id="tenant-beta")
        results["beta"] = snap.chunk_count

    t1 = threading.Thread(target=worker_alpha)
    t2 = threading.Thread(target=worker_alpha)
    t3 = threading.Thread(target=worker_beta)
    t4 = threading.Thread(target=worker_beta)

    for t in (t1, t2, t3, t4):
        t.start()
    for t in (t1, t2, t3, t4):
        t.join(timeout=5)

    assert results["alpha"] == 5
    assert results["beta"] == 10
    assert manager.status("tenant-alpha")["chunk_count"] == 5
    assert manager.status("tenant-beta")["chunk_count"] == 10


def test_tenant_index_invalidation_affects_only_intended_tenant():
    manager = IndexManager()
    docs_a = [
        Document(
            page_content="alpha doc",
            metadata={"chunk_id": "ca", "doc_id": "da", "tenant_id": "tenant-a", "version_state": "active"},
        )
    ]
    docs_b = [
        Document(
            page_content="beta doc",
            metadata={"chunk_id": "cb", "doc_id": "db", "tenant_id": "tenant-b", "version_state": "active"},
        )
    ]

    manager.rebuild(docs_a, tenant_id="tenant-a")
    manager.rebuild(docs_b, tenant_id="tenant-b")

    # Invalidate Tenant A only
    manager.invalidate("tenant a document deleted", tenant_id="tenant-a")

    with pytest.raises(IndexUnavailable):
        manager.get_snapshot("tenant-a")

    # Tenant B remains healthy and active!
    snap_b = manager.get_snapshot("tenant-b")
    assert snap_b is not None
    assert snap_b.chunk_count == 1

