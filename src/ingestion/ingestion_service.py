"""Versioned ingestion with staging, verification, activation and rollback.

Lifecycle per document version:
    staging -> active -> retired -> deleted

State machine:
    RECEIVED -> VALIDATING -> EXTRACTING -> CHUNKING -> EMBEDDING -> STAGING -> VERIFYING -> ACTIVATING -> ACTIVE
    Failure paths:
    -> FAILED
    -> CANCELLED
    -> ROLLBACK_REQUIRED

New chunks are written as `staging` (never returned by retrieval filters),
verified by exact chunk and point ID match, then activated. The previous version
is retired, the immutable retrieval snapshot is rebuilt, and only then are retired
vectors deleted. Any failure before snapshot activation restores the previous
version and removes the partial write.
"""
from __future__ import annotations

import datetime
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.documents import Document

from src.config import settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.document_registry import DocumentRecord, SQLDocumentRegistry, VersionRecord
from src.ingestion.fingerprint import content_sha256, derive_document_id, normalize_text
from src.ingestion.loaders import load_document
from src.runtime.errors import DocumentNotFound, DocumentTooLarge, IngestionError
from src.retrieval.index_manager import IndexManager
from src.retrieval.retrieval_runtime import get_default_index_manager
from src.retrieval.vector_store import (
    add_with_embeddings,
    chunks_present,
    delete_chunks,
    get_indexed_documents,
    get_retrieval_documents,
    set_chunk_state,
)


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class DocumentRegistry:
    """Document registry adapter supporting SQLite, PostgreSQL, and JSON compatibility."""

    def __init__(self, path: Optional[Union[str, Path]] = None, db_path: Optional[str] = None):
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        if db_path:
            self._sql = SQLDocumentRegistry(db_path=db_path)
        elif self.path is None:
            self._sql = SQLDocumentRegistry(db_path=":memory:")
        elif str(self.path).endswith(".db"):
            self._sql = SQLDocumentRegistry(db_path=str(self.path))
        else:
            p_str = str(self.path)
            db_target = p_str.replace(".json", ".db") if p_str.endswith(".json") else f"{p_str}.db"
            self._sql = SQLDocumentRegistry(db_path=db_target)
        self._load_legacy_json()

    def _load_legacy_json(self) -> None:
        if not self.path or not self.path.exists() or not str(self.path).endswith(".json"):
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            for item in payload.get("documents", []):
                record = DocumentRecord.from_dict(item)
                if not self._sql.get(record.tenant_id, record.doc_id):
                    self._sql.put(record)
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(
                f"document registry is corrupt at {self.path}: {type(exc).__name__}", component="ingestion"
            ) from exc

    def _sync_json(self) -> None:
        if not self.path or not str(self.path).endswith(".json"):
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = self._sql._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT tenant_id FROM documents")
            tenants = [r["tenant_id"] for r in cur.fetchall()]
            docs = []
            for t in tenants:
                docs.extend([r.as_dict() for r in self._sql.list(t, include_deleted=True)])
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"schema_version": 1, "documents": docs}, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:
            pass

    def get(self, tenant_id: str, doc_id: str) -> Optional[DocumentRecord]:
        return self._sql.get(tenant_id, doc_id)

    def put(self, record: DocumentRecord, chunk_records: Optional[List[Dict[str, Any]]] = None) -> None:
        self._sql.put(record, chunk_records=chunk_records)
        self._sync_json()

    def remove(self, tenant_id: str, doc_id: str) -> Optional[DocumentRecord]:
        rec = self._sql.remove(tenant_id, doc_id)
        self._sync_json()
        return rec

    def list(self, tenant_id: str, include_deleted: bool = False) -> List[DocumentRecord]:
        return self._sql.list(tenant_id, include_deleted=include_deleted)

    def get_chunk_ids(self, tenant_id: str, doc_id: str, version: Optional[int] = None) -> List[str]:
        return self._sql.get_chunk_ids(tenant_id, doc_id, version=version)

    def create_job(self, *args, **kwargs):
        return self._sql.create_job(*args, **kwargs)

    def update_job(self, *args, **kwargs):
        return self._sql.update_job(*args, **kwargs)

    def get_job(self, *args, **kwargs):
        return self._sql.get_job(*args, **kwargs)

    def get_job_by_idempotency(self, *args, **kwargs):
        return self._sql.get_job_by_idempotency(*args, **kwargs)

    def get_stuck_jobs(self, *args, **kwargs):
        return self._sql.get_stuck_jobs(*args, **kwargs)

    def get_staged_versions(self, *args, **kwargs):
        return self._sql.get_staged_versions(*args, **kwargs)

    def get_retired_versions(self, *args, **kwargs):
        return self._sql.get_retired_versions(*args, **kwargs)

    def record_indexing_event(self, *args, **kwargs):
        return self._sql.record_indexing_event(*args, **kwargs)

    def record_audit_event(self, *args, **kwargs):
        return self._sql.record_audit_event(*args, **kwargs)

    def reset(self) -> None:
        self._sql.reset()
        if self.path and self.path.exists():
            self.path.unlink(missing_ok=True)


@dataclass
class IngestResult:
    doc_id: str
    version: int
    chunks_created: int
    status: str
    content_sha256: str
    duplicate: bool = False
    activated: bool = False
    index_version: Optional[int] = None
    warnings: List[str] = field(default_factory=list)
    tenant_id: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeleteResult:
    doc_id: str
    tenant_id: str
    chunks_deleted: int
    status: str
    index_refreshed: bool
    warnings: List[str] = field(default_factory=list)


class IngestionService:
    def __init__(
        self,
        store,
        *,
        embeddings=None,
        index_manager: Optional[IndexManager] = None,
        registry: Optional[Union[DocumentRegistry, SQLDocumentRegistry]] = None,
        embedding_model_id: Optional[str] = None,
    ):
        self.store = store
        self.embeddings = embeddings
        self.index_manager = index_manager or get_default_index_manager()
        if registry is not None:
            self.registry = registry
        else:
            from langchain_core.vectorstores import InMemoryVectorStore

            path = None if isinstance(store, InMemoryVectorStore) else settings.INGESTION_REGISTRY_PATH
            self.registry = DocumentRegistry(path)
        self.embedding_model_id = embedding_model_id or settings.EMBEDDING_MODEL
        self._doc_locks: Dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    def _get_doc_lock(self, tenant_id: str, doc_id: str) -> threading.Lock:
        key = f"{tenant_id}\x00{doc_id}"
        with self._locks_lock:
            if key not in self._doc_locks:
                self._doc_locks[key] = threading.Lock()
            return self._doc_locks[key]

    def _embedder(self):
        if self.embeddings is not None:
            return self.embeddings
        from src.runtime.lifecycle import get_model_registry

        self.embeddings = get_model_registry().get_embeddings()
        return self.embeddings

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def ingest_text(
        self,
        text: str,
        *,
        doc_id: Optional[str] = None,
        source: str = "api_text",
        tenant_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> IngestResult:
        if not tenant_id or not tenant_id.strip():
            raise IngestionError("tenant_id is required", component="ingestion")
        normalized = normalize_text(text)
        if not normalized:
            raise IngestionError("document contains no readable text", component="ingestion")
        if len(normalized) > settings.MAX_DOCUMENT_CHARS:
            raise DocumentTooLarge(
                f"document exceeds MAX_DOCUMENT_CHARS={settings.MAX_DOCUMENT_CHARS}", component="ingestion"
            )
        resolved_id = derive_document_id(tenant_id, source, doc_id)

        doc = Document(
            page_content=normalized,
            metadata={
                **(metadata or {}),
                "source": source,
                "doc_id": resolved_id,
                "tenant_id": tenant_id,
            },
        )
        return self._ingest_documents(
            [doc],
            doc_id=resolved_id,
            source=source,
            tenant_id=tenant_id,
            metadata=metadata or {},
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
            idempotency_key=idempotency_key,
        )

    def ingest_file(
        self,
        path: str,
        *,
        filename: str,
        doc_id: Optional[str] = None,
        tenant_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> IngestResult:
        docs = load_document(path)
        if not docs:
            raise IngestionError(f"no readable text extracted from {filename}", component="loader")
        has_content = any(normalize_text(d.page_content) for d in docs)
        if not has_content:
            raise IngestionError(f"no readable text extracted from {filename}", component="loader")

        resolved_id = derive_document_id(tenant_id, filename, doc_id)
        enriched = dict(metadata or {})
        enriched.setdefault("file_name", filename)
        if len(docs) > 1 or docs[0].metadata.get("page") is not None:
            enriched.setdefault("page_count", len(docs))

        # Preserve individual page documents and provenance
        raw_docs: List[Document] = []
        for doc in docs:
            norm = normalize_text(doc.page_content)
            if not norm:
                continue
            meta = {
                **enriched,
                "source": filename,
                "doc_id": resolved_id,
                "tenant_id": tenant_id,
            }
            if doc.metadata.get("page") is not None:
                meta["page"] = doc.metadata["page"]
            raw_docs.append(Document(page_content=norm, metadata=meta))

        return self._ingest_documents(
            raw_docs,
            doc_id=resolved_id,
            source=filename,
            tenant_id=tenant_id,
            metadata=enriched,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
            idempotency_key=idempotency_key,
        )

    def _ingest_documents(
        self,
        raw_docs: List[Document],
        *,
        doc_id: str,
        source: str,
        tenant_id: str,
        metadata: Dict[str, Any],
        chunk_size: Optional[int],
        chunk_overlap: Optional[int],
        strategy: Optional[str],
        idempotency_key: Optional[str] = None,
    ) -> IngestResult:
        full_content = "\n\n".join(d.page_content for d in raw_docs)
        digest = content_sha256(full_content)
        job_id = str(uuid.uuid4())

        with self._get_doc_lock(tenant_id, doc_id):
            # 1. State: RECEIVED
            if idempotency_key and hasattr(self.registry, "get_job_by_idempotency"):
                prior_job = self.registry.get_job_by_idempotency(tenant_id, idempotency_key)
                if prior_job and prior_job.get("state") == "ACTIVE":
                    existing = self.registry.get(tenant_id, doc_id)
                    if existing and existing.active_version is not None:
                        active = existing.versions.get(existing.active_version)
                        if active and not existing.deleted:
                            return IngestResult(
                                doc_id=doc_id,
                                version=active.version,
                                chunks_created=0,
                                status="unchanged",
                                content_sha256=active.content_sha256,
                                duplicate=True,
                                activated=True,
                                tenant_id=tenant_id,
                            )

            if hasattr(self.registry, "create_job"):
                self.registry.create_job(
                    job_id=job_id,
                    tenant_id=tenant_id,
                    doc_id=doc_id,
                    version=0,
                    idempotency_key=idempotency_key,
                    state="RECEIVED",
                )
            if hasattr(self.registry, "record_audit_event"):
                self.registry.record_audit_event(
                    id=str(uuid.uuid4()),
                    tenant_id=tenant_id,
                    doc_id=doc_id,
                    action="ingestion_received",
                    details=json.dumps({"job_id": job_id, "source": source}),
                )

            # 2. State: VALIDATING
            if hasattr(self.registry, "update_job"):
                self.registry.update_job(job_id, state="VALIDATING")

            existing = self.registry.get(tenant_id, doc_id)
            if existing and existing.active_version is not None:
                active = existing.versions.get(existing.active_version)
                if active and active.content_sha256 == digest and not existing.deleted:
                    if hasattr(self.registry, "update_job"):
                        self.registry.update_job(job_id, state="ACTIVE")
                    return IngestResult(
                        doc_id=doc_id,
                        version=active.version,
                        chunks_created=0,
                        status="unchanged",
                        content_sha256=digest,
                        duplicate=True,
                        activated=True,
                        tenant_id=tenant_id,
                    )

            previous = existing.clone() if existing else None
            version = (max(previous.versions) + 1) if previous and previous.versions else 1

            # 3. State: EXTRACTING / PREPARING
            if hasattr(self.registry, "update_job"):
                self.registry.update_job(job_id, state="EXTRACTING")

            safe_user_metadata = {
                key: value
                for key, value in (metadata or {}).items()
                if key
                not in {
                    "doc_id",
                    "tenant_id",
                    "chunk_id",
                    "chunk_index",
                    "document_version",
                    "version_state",
                    "content_sha256",
                    "source",
                }
            }

            prepared_docs = []
            for d in raw_docs:
                d_meta = {
                    **safe_user_metadata,
                    **d.metadata,
                    "doc_id": doc_id,
                    "source": source,
                    "tenant_id": tenant_id,
                    "document_version": version,
                }
                prepared_docs.append(Document(page_content=d.page_content, metadata=d_meta))

            # 4. State: CHUNKING
            if hasattr(self.registry, "update_job"):
                self.registry.update_job(job_id, state="CHUNKING")

            chunks = chunk_documents(
                prepared_docs,
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                document_version=version,
            )
            if not chunks:
                if hasattr(self.registry, "update_job"):
                    self.registry.update_job(job_id, state="FAILED", error="document could not be chunked")
                raise IngestionError("document could not be chunked into valid pieces", component="ingestion")

            ingestion_id = job_id
            for chunk in chunks:
                chunk.metadata.update(
                    version_state="staging",
                    content_sha256=digest,
                    ingestion_id=ingestion_id,
                )
            chunk_ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
            if len(set(chunk_ids)) != len(chunk_ids):
                if hasattr(self.registry, "update_job"):
                    self.registry.update_job(job_id, state="FAILED", error="chunk identity collision")
                raise IngestionError("chunk identity collision within document", component="ingestion")

            version_record = VersionRecord(
                version=version,
                content_sha256=digest,
                chunk_ids=chunk_ids,
                chunk_count=len(chunks),
                status="staging",
                embedding_model=self.embedding_model_id,
                chunk_size=chunk_size or settings.CHUNK_SIZE,
                chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP,
                chunk_strategy=strategy or settings.CHUNKER_STRATEGY,
            )
            record = previous.clone() if previous else DocumentRecord(
                doc_id=doc_id, tenant_id=tenant_id, source=source
            )
            record.deleted = False
            record.source = source
            record.versions[version] = version_record

            old_ids: List[str] = []
            if previous and previous.active_version is not None:
                old = previous.versions.get(previous.active_version)
                old_ids = list(old.chunk_ids) if old else []

            # 5. State: EMBEDDING & 6. State: STAGING
            if hasattr(self.registry, "update_job"):
                self.registry.update_job(job_id, state="EMBEDDING")

            try:
                if hasattr(self.registry, "update_job"):
                    self.registry.update_job(job_id, state="STAGING")
                add_with_embeddings(self.store, chunks, chunk_ids, self._embedder())
            except Exception as exc:  # noqa: BLE001
                if hasattr(self.registry, "update_job"):
                    self.registry.update_job(job_id, state="ROLLBACK_REQUIRED", error=str(exc))
                self._cleanup_ids(chunk_ids)
                if hasattr(self.registry, "update_job"):
                    self.registry.update_job(job_id, state="FAILED", error=str(exc))
                raise IngestionError(
                    f"embedding/staging failed: {type(exc).__name__}: {exc}", component="ingestion"
                ) from exc

            # 7. State: VERIFYING
            if hasattr(self.registry, "update_job"):
                self.registry.update_job(job_id, state="VERIFYING")

            if not chunks_present(self.store, chunk_ids):
                if hasattr(self.registry, "update_job"):
                    self.registry.update_job(job_id, state="ROLLBACK_REQUIRED", error="missing vectors in verification")
                self._cleanup_ids(chunk_ids)
                if hasattr(self.registry, "update_job"):
                    self.registry.update_job(job_id, state="FAILED", error="verification failed")
                raise IngestionError("staged chunks failed verification (missing vectors)", component="ingestion")

            # 8. State: ACTIVATING
            if hasattr(self.registry, "update_job"):
                self.registry.update_job(job_id, state="ACTIVATING")

            try:
                if old_ids:
                    set_chunk_state(self.store, old_ids, "retired")
                set_chunk_state(self.store, chunk_ids, "active")
                record.active_version = version
                record.updated_at = _utcnow()
                version_record.status = "active"
                version_record.activated_at = _utcnow()
                for old_version in record.versions.values():
                    if old_version.version != version and old_version.status == "active":
                        old_version.status = "retired"
                        old_version.retired_at = _utcnow()

                chunk_records = [
                    {
                        "chunk_id": c.metadata["chunk_id"],
                        "tenant_id": tenant_id,
                        "doc_id": doc_id,
                        "version": version,
                        "page": c.metadata.get("page"),
                        "section_path": c.metadata.get("section_path"),
                        "point_id": c.metadata["chunk_id"],
                    }
                    for c in chunks
                ]
                self.registry.put(record, chunk_records=chunk_records)
                snapshot = self.index_manager.rebuild(
                    get_retrieval_documents(self.store), store=self.store, tenant_id=tenant_id
                )
                if hasattr(self.registry, "record_indexing_event"):
                    self.registry.record_indexing_event(
                        event_id=str(uuid.uuid4()),
                        tenant_id=tenant_id,
                        event_type="rebuild",
                        version=snapshot.version,
                        details=f"Activated version {version} for doc {doc_id}",
                    )
            except Exception as exc:  # noqa: BLE001
                if hasattr(self.registry, "update_job"):
                    self.registry.update_job(job_id, state="ROLLBACK_REQUIRED", error=str(exc))
                self._rollback(previous, record, old_ids, chunk_ids)
                if hasattr(self.registry, "update_job"):
                    self.registry.update_job(job_id, state="FAILED", error=str(exc))
                raise IngestionError(
                    f"activation failed; previous version preserved: {type(exc).__name__}: {exc}",
                    component="ingestion",
                ) from exc

            # 9. State: ACTIVE
            if hasattr(self.registry, "update_job"):
                self.registry.update_job(job_id, state="ACTIVE")
            if hasattr(self.registry, "record_audit_event"):
                self.registry.record_audit_event(
                    id=str(uuid.uuid4()),
                    tenant_id=tenant_id,
                    doc_id=doc_id,
                    action="version_activated",
                    details=json.dumps({"version": version, "chunk_count": len(chunks)}),
                )
            try:
                from src.observability.metrics import metrics
                active_count = len([d for d in self.registry.list(tenant_id=tenant_id) if d.active_version is not None])
                metrics.set_gauge("rag_active_document_versions", float(active_count), {"tenant_id": tenant_id})
            except Exception:
                pass

            warnings: List[str] = []
            if old_ids:
                try:
                    delete_chunks(self.store, old_ids)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(
                        f"retired vectors could not be deleted ({type(exc).__name__}); they are excluded by state filter"
                    )

            return IngestResult(
                doc_id=doc_id,
                version=version,
                chunks_created=len(chunks),
                status="updated" if previous else "created",
                content_sha256=digest,
                duplicate=False,
                activated=True,
                index_version=snapshot.version,
                warnings=warnings,
                tenant_id=tenant_id,
            )

    def _cleanup_ids(self, ids: Sequence[str]) -> None:
        try:
            delete_chunks(self.store, list(ids))
        except Exception:  # noqa: BLE001
            self.index_manager.invalidate("failed to remove partially staged chunks")

    def _rollback(
        self,
        previous: Optional[DocumentRecord],
        current: DocumentRecord,
        old_ids: Sequence[str],
        new_ids: Sequence[str],
    ) -> None:
        try:
            delete_chunks(self.store, list(new_ids))
            if old_ids:
                set_chunk_state(self.store, list(old_ids), "active")
            if previous is not None:
                self.registry.put(previous)
            else:
                self.registry.remove(current.tenant_id, current.doc_id)
        except Exception:  # noqa: BLE001
            self.index_manager.invalidate("ingestion rollback failed", tenant_id=current.tenant_id)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    def delete_document(self, doc_id: str, *, tenant_id: str = "default") -> DeleteResult:
        with self._get_doc_lock(tenant_id, doc_id):
            record = self.registry.get(tenant_id, doc_id)
            ids: set[str] = set()
            if record:
                for version in record.versions.values():
                    ids.update(version.chunk_ids)
            if hasattr(self.registry, "get_chunk_ids"):
                ids.update(self.registry.get_chunk_ids(tenant_id, doc_id))

            # Performance optimization: only scan collection if registry record is absent
            if not ids and record is None:
                scanned = [
                    str(doc.metadata.get("chunk_id"))
                    for doc in get_indexed_documents(self.store)
                    if doc.metadata.get("doc_id") == doc_id and doc.metadata.get("tenant_id", "default") == tenant_id
                ]
                ids.update(scanned)

            if not ids and record is None:
                raise DocumentNotFound(f"document '{doc_id}' was not found for this tenant", component="ingestion")

            deleted = 0
            if ids:
                try:
                    deleted = delete_chunks(self.store, list(ids))
                except Exception as exc:  # noqa: BLE001
                    self.index_manager.invalidate("document deletion could not be applied to the vector store", tenant_id=tenant_id)
                    raise IngestionError(
                        f"deletion failed: {type(exc).__name__}: {exc}", component="ingestion"
                    ) from exc

            warnings: List[str] = []
            index_refreshed = True
            try:
                self.index_manager.rebuild(get_retrieval_documents(self.store), store=self.store, tenant_id=tenant_id)
            except Exception as exc:  # noqa: BLE001
                self.index_manager.invalidate("index refresh after deletion failed", tenant_id=tenant_id)
                index_refreshed = False
                warnings.append(f"index refresh failed: {type(exc).__name__}")

            self.registry.remove(tenant_id, doc_id)
            if hasattr(self.registry, "record_audit_event"):
                self.registry.record_audit_event(
                    id=str(uuid.uuid4()),
                    tenant_id=tenant_id,
                    doc_id=doc_id,
                    action="document_deleted",
                    details=json.dumps({"chunks_deleted": deleted}),
                )
            try:
                from src.observability.metrics import metrics
                active_count = len([d for d in self.registry.list(tenant_id=tenant_id) if d.active_version is not None])
                metrics.set_gauge("rag_active_document_versions", float(active_count), {"tenant_id": tenant_id})
            except Exception:
                pass

            return DeleteResult(
                doc_id=doc_id,
                tenant_id=tenant_id,
                chunks_deleted=deleted,
                status="deleted" if index_refreshed else "deleted_index_invalidated",
                index_refreshed=index_refreshed,
                warnings=warnings,
            )

    # ------------------------------------------------------------------
    # Crash Recovery & Orphan Cleanup
    # ------------------------------------------------------------------
    def recover_and_cleanup(
        self,
        *,
        staging_timeout_seconds: float = 300.0,
        stuck_job_timeout_seconds: float = 300.0,
        retired_retention_seconds: float = 86400.0,
    ) -> Dict[str, int]:
        """Scans for stuck jobs, orphaned staging vectors, and old retired vectors."""
        recovered_jobs = 0
        cleaned_staged = 0
        purged_retired = 0

        # 1. Recover stuck jobs
        if hasattr(self.registry, "get_stuck_jobs"):
            stuck_jobs = self.registry.get_stuck_jobs(timeout_seconds=stuck_job_timeout_seconds)
            for job in stuck_jobs:
                tenant_id = job["tenant_id"]
                doc_id = job["doc_id"]
                job_id = job["job_id"]
                if hasattr(self.registry, "get_chunk_ids"):
                    chunk_ids = self.registry.get_chunk_ids(tenant_id, doc_id, version=job.get("version"))
                    if chunk_ids:
                        self._cleanup_ids(chunk_ids)
                if hasattr(self.registry, "update_job"):
                    self.registry.update_job(job_id, state="FAILED", error="recovered from stuck state after timeout")
                recovered_jobs += 1

        # 2. Clean up orphaned staging versions
        if hasattr(self.registry, "get_staged_versions"):
            staged_vers = self.registry.get_staged_versions(timeout_seconds=staging_timeout_seconds)
            for sv in staged_vers:
                tenant_id = sv["tenant_id"]
                doc_id = sv["doc_id"]
                version = sv["version"]
                if hasattr(self.registry, "get_chunk_ids"):
                    chunk_ids = self.registry.get_chunk_ids(tenant_id, doc_id, version=version)
                    if chunk_ids:
                        self._cleanup_ids(chunk_ids)
                rec = self.registry.get(tenant_id, doc_id)
                if rec and version in rec.versions:
                    rec.versions[version].status = "failed"
                    rec.versions[version].error = "staged version timed out and was cleaned up"
                    self.registry.put(rec)
                cleaned_staged += 1

        # 3. Purge old retired vectors
        if hasattr(self.registry, "get_retired_versions"):
            retired_vers = self.registry.get_retired_versions(retention_seconds=retired_retention_seconds)
            for rv in retired_vers:
                tenant_id = rv["tenant_id"]
                doc_id = rv["doc_id"]
                version = rv["version"]
                if hasattr(self.registry, "get_chunk_ids"):
                    chunk_ids = self.registry.get_chunk_ids(tenant_id, doc_id, version=version)
                    if chunk_ids:
                        try:
                            delete_chunks(self.store, chunk_ids)
                        except Exception:
                            pass
                purged_retired += 1

        return {
            "stuck_jobs_recovered": recovered_jobs,
            "staged_versions_cleaned": cleaned_staged,
            "retired_versions_purged": purged_retired,
        }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_document(self, doc_id: str, *, tenant_id: str = "default") -> Optional[Dict[str, Any]]:
        record = self.registry.get(tenant_id, doc_id)
        if not record or record.deleted:
            return None
        active = record.versions.get(record.active_version) if record.active_version else None
        return {
            "doc_id": record.doc_id,
            "tenant_id": record.tenant_id,
            "source": record.source,
            "active_version": record.active_version,
            "status": active.status if active else "unknown",
            "chunk_count": active.chunk_count if active else 0,
            "content_sha256": active.content_sha256 if active else None,
            "versions": [
                {
                    "version": version.version,
                    "status": version.status,
                    "chunks": version.chunk_count,
                    "created_at": version.created_at,
                    "activated_at": version.activated_at,
                    "content_sha256": version.content_sha256,
                }
                for version in sorted(record.versions.values(), key=lambda v: v.version)
            ],
        }

    def list_documents(self, *, tenant_id: str = "default") -> List[Dict[str, Any]]:
        result = []
        for record in sorted(self.registry.list(tenant_id), key=lambda r: r.doc_id):
            active = record.versions.get(record.active_version) if record.active_version else None
            result.append(
                {
                    "doc_id": record.doc_id,
                    "tenant_id": record.tenant_id,
                    "source": record.source,
                    "active_version": record.active_version,
                    "status": active.status if active else "unknown",
                    "chunk_count": active.chunk_count if active else 0,
                    "content_sha256": active.content_sha256 if active else None,
                    "versions": [
                        {
                            "version": version.version,
                            "status": version.status,
                            "chunks": version.chunk_count,
                            "created_at": version.created_at,
                            "activated_at": version.activated_at,
                            "content_sha256": version.content_sha256,
                        }
                        for version in sorted(record.versions.values(), key=lambda v: v.version)
                    ],
                }
            )
        return result
