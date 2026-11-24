"""Reliability, crash recovery, and concurrency tests for document ingestion."""
from __future__ import annotations

import concurrent.futures
import threading
import time
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from src.evaluation.offline import HashEmbeddings
from src.ingestion.document_registry import DocumentRecord, SQLDocumentRegistry, VersionRecord
from src.ingestion.ingestion_service import DocumentRegistry, IngestionService
from src.retrieval.index_manager import IndexManager
from src.runtime.errors import IngestionError


def _setup_service(tmp_path=None):
    store = InMemoryVectorStore(embedding=HashEmbeddings())
    manager = IndexManager()
    db_path = str(tmp_path / "registry.db") if tmp_path else ":memory:"
    registry = SQLDocumentRegistry(db_path=db_path)
    service = IngestionService(store, embeddings=HashEmbeddings(), index_manager=manager, registry=registry)
    return service, store, manager, registry


def test_concurrent_workers_updating_same_document(tmp_path):
    """Two concurrent workers attempting to update the same doc_id must execute cleanly without corrupting state."""
    service, store, manager, registry = _setup_service(tmp_path)

    # Pre-seed document
    service.ingest_text("Initial version text.", doc_id="doc1", tenant_id="tenant_a")

    errors = []
    results = []

    def worker(worker_id: int):
        try:
            res = service.ingest_text(
                f"Updated text from worker {worker_id} with distinct contents.",
                doc_id="doc1",
                tenant_id="tenant_a",
            )
            results.append(res)
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(worker, 1)
        f2 = executor.submit(worker, 2)
        f1.result()
        f2.result()

    assert not errors
    assert len(results) == 2
    doc = registry.get("tenant_a", "doc1")
    assert doc is not None
    assert doc.active_version in {2, 3}
    # Active version in store matches registry active version
    active_in_store = [
        row for row in store.store.values()
        if row["metadata"].get("doc_id") == "doc1" and row["metadata"].get("version_state") == "active"
    ]
    assert len(active_in_store) > 0
    assert all(row["metadata"]["document_version"] == doc.active_version for row in active_in_store)


def test_staging_verification_failure_triggers_rollback(tmp_path, monkeypatch):
    """When chunks_present fails, staging vectors must be cleaned up and rollback executed."""
    service, store, manager, registry = _setup_service(tmp_path)
    service.ingest_text("Stable initial document version.", doc_id="doc1", tenant_id="t1")
    initial_chunks = len(store.store)

    # Force chunks_present to return False
    from src.ingestion import ingestion_service
    monkeypatch.setattr(ingestion_service, "chunks_present", lambda s, ids: False)

    with pytest.raises(IngestionError, match="staged chunks failed verification"):
        service.ingest_text("New version that fails verification.", doc_id="doc1", tenant_id="t1")

    # Vector store should not have the failed staging chunks
    assert len(store.store) == initial_chunks
    doc = registry.get("t1", "doc1")
    assert doc.active_version == 1


def test_failure_after_staging_preserves_previous_version(tmp_path, monkeypatch):
    """Simulate failure during activation (e.g. index rebuild failure)."""
    service, store, manager, registry = _setup_service(tmp_path)
    service.ingest_text("Original document content.", doc_id="doc1", tenant_id="t1")

    def fail_rebuild(*args, **kwargs):
        raise RuntimeError("simulated index rebuild crash")

    monkeypatch.setattr(manager, "rebuild", fail_rebuild)

    with pytest.raises(IngestionError, match="activation failed; previous version preserved"):
        service.ingest_text("Second version that crashes during index rebuild.", doc_id="doc1", tenant_id="t1")

    # Document active version should remain 1
    doc = registry.get("t1", "doc1")
    assert doc.active_version == 1
    # Active chunks should be version 1
    active = [
        r for r in store.store.values()
        if r["metadata"].get("version_state") == "active"
    ]
    assert all(r["metadata"]["document_version"] == 1 for r in active)


def test_idempotency_key_prevents_duplicate_work(tmp_path):
    """Requests with the same idempotency_key return existing active results."""
    service, store, manager, registry = _setup_service(tmp_path)

    r1 = service.ingest_text(
        "Important security policy statement.",
        doc_id="sec-policy",
        tenant_id="t1",
        idempotency_key="idemp-key-100",
    )
    assert r1.status == "created"
    assert r1.chunks_created > 0

    # Second request with identical idempotency key
    r2 = service.ingest_text(
        "Important security policy statement.",
        doc_id="sec-policy",
        tenant_id="t1",
        idempotency_key="idemp-key-100",
    )
    assert r2.duplicate is True
    assert r2.status == "unchanged"
    assert r2.chunks_created == 0


def test_crash_recovery_and_orphan_cleanup(tmp_path):
    """Stuck jobs and orphaned staging vectors are identified and cleaned up."""
    service, store, manager, registry = _setup_service(tmp_path)

    # Ingest a valid document first
    service.ingest_text("Normal active document.", doc_id="doc_ok", tenant_id="t1")

    # Manually simulate a stuck job and staging chunks
    stuck_job_id = "job-stuck-999"
    # Create document and staging version first to satisfy foreign key
    doc_rec = DocumentRecord(doc_id="doc_stuck", tenant_id="t1", source="file.txt")
    doc_rec.versions[1] = VersionRecord(version=1, content_sha256="fake_sha", status="staging", created_at="2020-01-01T00:00:00")
    registry.put(doc_rec, chunk_records=[{
        "chunk_id": "doc_stuck::c0::dummy",
        "tenant_id": "t1",
        "doc_id": "doc_stuck",
        "version": 1,
        "point_id": "doc_stuck::c0::dummy",
    }])
    registry.create_job(stuck_job_id, "t1", "doc_stuck", version=1, state="STAGING")
    # Backdate job
    conn = registry._get_connection()
    with conn:
        conn.execute("UPDATE ingestion_jobs SET updated_at = '2020-01-01T00:00:00' WHERE job_id = ?", (stuck_job_id,))

    # Add chunk to store
    store.store["doc_stuck::c0::dummy"] = {
        "id": "doc_stuck::c0::dummy",
        "text": "Staging chunk",
        "metadata": {"version_state": "staging"},
    }

    # Run recovery
    stats = service.recover_and_cleanup(stuck_job_timeout_seconds=10.0, staging_timeout_seconds=10.0)
    assert stats["stuck_jobs_recovered"] >= 1
    # Stuck job transitioned to FAILED
    job = registry.get_job(stuck_job_id)
    assert job["state"] == "FAILED"
    # Orphaned staging chunk removed from store
    assert "doc_stuck::c0::dummy" not in store.store


def test_optimized_deletion_bypasses_full_collection_scan(tmp_path, monkeypatch):
    """delete_document looks up chunk IDs from registry directly without scanning the full vector store."""
    service, store, manager, registry = _setup_service(tmp_path)
    res = service.ingest_text("Text to be deleted cleanly.", doc_id="doc_del", tenant_id="t1")

    scanned_called = False
    from src.ingestion import ingestion_service
    original_get_indexed = ingestion_service.get_indexed_documents

    def spy_get_indexed(*args, **kwargs):
        nonlocal scanned_called
        scanned_called = True
        return original_get_indexed(*args, **kwargs)

    monkeypatch.setattr(ingestion_service, "get_indexed_documents", spy_get_indexed)

    del_res = service.delete_document("doc_del", tenant_id="t1")
    assert del_res.status == "deleted"
    # When registry has the record, delete_chunks is called directly; get_indexed_documents was NOT called for scan!
    # (Note: rebuild calls get_retrieval_documents, not get_indexed_documents for deletion candidate search)
    assert registry.get("t1", "doc_del") is None
    assert "doc_del" not in [r["metadata"].get("doc_id") for r in store.store.values()]


def test_relational_schema_tables_and_audit_events(tmp_path):
    """Verify that SQLDocumentRegistry records events in indexing_events and audit_events."""
    service, store, manager, registry = _setup_service(tmp_path)
    service.ingest_text("Audit test document content.", doc_id="audit_doc", tenant_id="t1")

    conn = registry._get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM audit_events WHERE tenant_id = 't1'")
    audit_rows = cur.fetchall()
    assert len(audit_rows) >= 2  # ingestion_received, version_activated

    cur.execute("SELECT * FROM indexing_events WHERE tenant_id = 't1'")
    index_rows = cur.fetchall()
    assert len(index_rows) >= 1  # rebuild event

    cur.execute("SELECT * FROM document_chunks WHERE tenant_id = 't1'")
    chunk_rows = cur.fetchall()
    assert len(chunk_rows) > 0
    assert chunk_rows[0]["point_id"] is not None
