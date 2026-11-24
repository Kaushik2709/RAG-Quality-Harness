"""Versioned ingestion: idempotency, activation, rollback and deletion."""
from __future__ import annotations

import threading

import pytest
from langchain_core.vectorstores import InMemoryVectorStore

from src.evaluation.offline import HashEmbeddings
from src.ingestion.ingestion_service import DocumentRegistry, IngestionService
from src.retrieval.index_manager import IndexManager
from src.runtime.errors import DocumentNotFound, IngestionError


def _service():
    store = InMemoryVectorStore(embedding=HashEmbeddings())
    manager = IndexManager()
    registry = DocumentRegistry(None)
    service = IngestionService(store, embeddings=HashEmbeddings(), index_manager=manager, registry=registry)
    return service, store, manager, registry


def _active_chunks(store, tenant="default", doc_id=None):
    result = []
    for row in store.store.values():
        metadata = row["metadata"]
        if metadata.get("version_state") != "active":
            continue
        if metadata.get("tenant_id", "default") != tenant:
            continue
        if doc_id is not None and metadata.get("doc_id") != doc_id:
            continue
        result.append(row)
    return result


def test_ingest_activates_once_and_is_idempotent():
    service, store, manager, registry = _service()
    first = service.ingest_text("Orion audit logs are retained for 30 days.", doc_id="policy")
    assert first.status == "created"
    assert first.activated and first.version == 1
    assert first.chunks_created > 0
    assert len(_active_chunks(store, doc_id="policy")) == first.chunks_created
    assert manager.status()["status"] == "ready"

    second = service.ingest_text("Orion audit logs are retained for 30 days.", doc_id="policy")
    assert second.duplicate is True
    assert second.status == "unchanged"
    assert second.version == 1
    assert second.chunks_created == 0
    assert registry.get("default", "policy").active_version == 1


def test_update_retires_and_deletes_previous_version():
    service, store, manager, _ = _service()
    first = service.ingest_text("Original retention window is thirty days.", doc_id="policy")
    second = service.ingest_text("Updated retention window is ninety days.", doc_id="policy")
    assert second.version == 2
    assert second.status == "updated"
    assert second.index_version > first.index_version
    active = _active_chunks(store, doc_id="policy")
    assert active
    assert all(row["metadata"]["document_version"] == 2 for row in active)
    assert all(row["metadata"]["version_state"] == "active" for row in store.store.values())
    assert manager.require_snapshot().chunk_count == len(store.store)


def test_tenant_isolation_for_same_doc_id():
    service, store, _, _ = _service()
    service.ingest_text("Tenant one policy.", doc_id="policy", tenant_id="t1")
    service.ingest_text("Tenant two policy.", doc_id="policy", tenant_id="t2")
    assert {row["metadata"]["tenant_id"] for row in store.store.values()} == {"t1", "t2"}
    assert {record["tenant_id"] for record in service.list_documents(tenant_id="t1")} == {"t1"}
    assert len(service.list_documents(tenant_id="t2")) == 1


def test_delete_removes_vectors_registry_and_index():
    service, store, manager, registry = _service()
    result = service.ingest_text("Delete me completely.", doc_id="policy")
    deleted = service.delete_document("policy")
    assert deleted.status == "deleted"
    assert deleted.chunks_deleted == result.chunks_created
    assert store.store == {}
    assert registry.get("default", "policy") is None
    assert manager.require_snapshot().chunk_count == 0
    with pytest.raises(DocumentNotFound):
        service.delete_document("policy")


def test_failed_index_rebuild_rolls_back_new_version(monkeypatch):
    service, store, manager, registry = _service()
    service.ingest_text("Original text remains available.", doc_id="policy")
    original_ids = {row_id for row_id in store.store}
    original_snapshot = manager.require_snapshot()
    original_content = {row["text"] for row in store.store.values()}

    def fail(*args, **kwargs):
        raise RuntimeError("simulated index failure")

    monkeypatch.setattr(manager, "rebuild", fail)
    with pytest.raises(IngestionError, match="previous version preserved"):
        service.ingest_text("Replacement text must not activate.", doc_id="policy")

    assert set(store.store) == original_ids
    assert {row["text"] for row in store.store.values()} == original_content
    assert all(row["metadata"]["version_state"] == "active" for row in store.store.values())
    record = registry.get("default", "policy")
    assert record.active_version == 1
    # The previous snapshot was never swapped because the rebuild failed.
    assert manager._snapshot is original_snapshot


def test_failed_staging_is_cleaned_up(monkeypatch):
    service, store, _, registry = _service()

    def fail_add(*args, **kwargs):
        raise RuntimeError("embedding provider failed")

    monkeypatch.setattr("src.ingestion.ingestion_service.add_with_embeddings", fail_add)
    with pytest.raises(IngestionError, match="staging failed"):
        service.ingest_text("This must not be indexed.", doc_id="policy")
    assert store.store == {}
    assert registry.get("default", "policy") is None


def test_validation_rejects_empty_and_oversized_documents(monkeypatch):
    service, _, _, _ = _service()
    with pytest.raises(IngestionError):
        service.ingest_text("   ", doc_id="blank")
    monkeypatch.setattr("src.ingestion.ingestion_service.settings.MAX_DOCUMENT_CHARS", 5)
    with pytest.raises(IngestionError, match="MAX_DOCUMENT_CHARS"):
        service.ingest_text("much longer than five characters", doc_id="large")


def test_concurrent_identical_ingest_is_idempotent():
    service, store, _, _ = _service()
    barrier = threading.Barrier(4)
    results = []

    def worker():
        barrier.wait()
        results.append(service.ingest_text("Shared policy content for concurrency.", doc_id="policy"))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(results) == 4
    assert sum(1 for result in results if result.status == "created") == 1
    assert sum(1 for result in results if result.duplicate) == 3
    assert len({row["metadata"]["document_version"] for row in store.store.values()}) == 1


def test_reserved_metadata_cannot_be_overridden():
    service, store, _, _ = _service()
    service.ingest_text(
        "Metadata protection.",
        doc_id="policy",
        tenant_id="t1",
        metadata={"tenant_id": "evil", "doc_id": "evil", "version_state": "active"},
    )
    assert all(row["metadata"]["tenant_id"] == "t1" for row in store.store.values())
    assert all(row["metadata"]["doc_id"] == "policy" for row in store.store.values())
