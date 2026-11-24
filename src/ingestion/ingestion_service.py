"""Versioned ingestion with staging, verification, activation and rollback.

Lifecycle per document version:
    staging -> active -> retired -> deleted

New chunks are written as `staging` (never returned by retrieval filters),
verified to exist, then activated. The previous version is retired in the same
lock, the immutable retrieval snapshot is rebuilt, and only then are retired
vectors deleted. Any failure before snapshot activation restores the previous
version and removes the partial write. A post-activation deletion failure cannot
resurface retired rows because they are already filtered out.
"""
from __future__ import annotations

import datetime
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document

from src.config import settings
from src.ingestion.chunker import chunk_documents
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


@dataclass
class VersionRecord:
    version: int
    content_sha256: str
    chunk_ids: List[str] = field(default_factory=list)
    chunk_count: int = 0
    status: str = "staging"
    created_at: str = field(default_factory=_utcnow)
    activated_at: Optional[str] = None
    retired_at: Optional[str] = None
    embedding_model: str = ""
    chunk_size: int = 0
    chunk_overlap: int = 0
    chunk_strategy: str = ""
    error: Optional[str] = None


@dataclass
class DocumentRecord:
    doc_id: str
    tenant_id: str
    source: str
    active_version: Optional[int] = None
    versions: Dict[int, VersionRecord] = field(default_factory=dict)
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    deleted: bool = False

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["versions"] = {str(key): value for key, value in data["versions"].items()}
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentRecord":
        versions = {int(key): VersionRecord(**value) for key, value in data.get("versions", {}).items()}
        return cls(
            doc_id=data["doc_id"],
            tenant_id=data["tenant_id"],
            source=data["source"],
            active_version=data.get("active_version"),
            versions=versions,
            created_at=data.get("created_at", _utcnow()),
            updated_at=data.get("updated_at", _utcnow()),
            deleted=bool(data.get("deleted", False)),
        )

    def clone(self) -> "DocumentRecord":
        return DocumentRecord.from_dict(self.as_dict())


class DocumentRegistry:
    """Small JSON registry. One process owns the file; a production multi-writer
    deployment should replace it with a transactional store (documented)."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self._docs: Dict[str, DocumentRecord] = {}
        self._load()

    @staticmethod
    def _key(tenant_id: str, doc_id: str) -> str:
        return f"{tenant_id}\x00{doc_id}"

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            for item in payload.get("documents", []):
                record = DocumentRecord.from_dict(item)
                self._docs[self._key(record.tenant_id, record.doc_id)] = record
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(
                f"document registry is corrupt at {self.path}: {type(exc).__name__}", component="ingestion"
            ) from exc

    def _persist_locked(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"schema_version": 1, "documents": [r.as_dict() for r in self._docs.values()]}, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def get(self, tenant_id: str, doc_id: str) -> Optional[DocumentRecord]:
        with self._lock:
            record = self._docs.get(self._key(tenant_id, doc_id))
            return record.clone() if record else None

    def put(self, record: DocumentRecord) -> None:
        with self._lock:
            record.updated_at = _utcnow()
            self._docs[self._key(record.tenant_id, record.doc_id)] = record.clone()
            self._persist_locked()

    def remove(self, tenant_id: str, doc_id: str) -> Optional[DocumentRecord]:
        with self._lock:
            record = self._docs.pop(self._key(tenant_id, doc_id), None)
            self._persist_locked()
            return record

    def list(self, tenant_id: str, include_deleted: bool = False) -> List[DocumentRecord]:
        with self._lock:
            return [
                record.clone()
                for record in self._docs.values()
                if record.tenant_id == tenant_id and (include_deleted or not record.deleted)
            ]

    def reset(self) -> None:
        with self._lock:
            self._docs.clear()
            if self.path and self.path.exists():
                self.path.unlink()


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
        registry: Optional[DocumentRegistry] = None,
        embedding_model_id: Optional[str] = None,
    ):
        self.store = store
        self.embeddings = embeddings
        self.index_manager = index_manager or get_default_index_manager()
        if registry is not None:
            self.registry = registry
        else:
            from langchain_core.vectorstores import InMemoryVectorStore

            # A volatile vector store must not be paired with a persisted registry
            # (a restart would claim documents that no longer exist).
            path = None if isinstance(store, InMemoryVectorStore) else settings.INGESTION_REGISTRY_PATH
            self.registry = DocumentRegistry(path)
        self.embedding_model_id = embedding_model_id or settings.EMBEDDING_MODEL
        self._write_lock = threading.Lock()

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
        return self._ingest_normalized(
            normalized,
            doc_id=resolved_id,
            source=source,
            tenant_id=tenant_id,
            metadata=metadata or {},
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
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
    ) -> IngestResult:
        docs = load_document(path)
        text = "\n\n".join(doc.page_content for doc in docs)
        pages = {idx + 1: doc.metadata.get("page") for idx, doc in enumerate(docs)}
        if not normalize_text(text):
            raise IngestionError(f"no readable text extracted from {filename}", component="loader")
        resolved_id = derive_document_id(tenant_id, filename, doc_id)
        enriched = dict(metadata or {})
        enriched.setdefault("file_name", filename)
        if any(value for value in pages.values()):
            enriched.setdefault("page_count", len(docs))
        return self.ingest_text(
            text,
            doc_id=resolved_id,
            source=filename,
            tenant_id=tenant_id,
            metadata=enriched,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
        )

    def _ingest_normalized(
        self,
        normalized: str,
        *,
        doc_id: str,
        source: str,
        tenant_id: str,
        metadata: Dict[str, Any],
        chunk_size: Optional[int],
        chunk_overlap: Optional[int],
        strategy: Optional[str],
    ) -> IngestResult:
        digest = content_sha256(normalized)
        with self._write_lock:
            existing = self.registry.get(tenant_id, doc_id)
            if existing and existing.active_version is not None:
                active = existing.versions.get(existing.active_version)
                if active and active.content_sha256 == digest and not existing.deleted:
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

            # Reserved metadata cannot be overridden by callers.
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
            raw = Document(
                page_content=normalized,
                metadata={
                    **safe_user_metadata,
                    "doc_id": doc_id,
                    "source": source,
                    "tenant_id": tenant_id,
                    "document_version": version,
                },
            )
            chunks = chunk_documents(
                [raw],
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                document_version=version,
            )
            if not chunks:
                raise IngestionError("document could not be chunked into valid pieces", component="ingestion")
            ingestion_id = str(uuid.uuid4())
            for chunk in chunks:
                chunk.metadata.update(
                    version_state="staging",
                    content_sha256=digest,
                    ingestion_id=ingestion_id,
                )
            chunk_ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
            if len(set(chunk_ids)) != len(chunk_ids):
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

            # ---- Stage ------------------------------------------------
            try:
                add_with_embeddings(self.store, chunks, chunk_ids, self._embedder())
            except Exception as exc:  # noqa: BLE001
                self._cleanup_ids(chunk_ids)
                raise IngestionError(
                    f"embedding/staging failed: {type(exc).__name__}: {exc}", component="ingestion"
                ) from exc

            # ---- Verify -----------------------------------------------
            if not chunks_present(self.store, chunk_ids):
                self._cleanup_ids(chunk_ids)
                raise IngestionError("staged chunks failed verification (missing vectors)", component="ingestion")

            # ---- Activate + retire ------------------------------------
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
                self.registry.put(record)
                snapshot = self.index_manager.rebuild(get_retrieval_documents(self.store), store=self.store)
            except Exception as exc:  # noqa: BLE001
                self._rollback(previous, record, old_ids, chunk_ids)
                raise IngestionError(
                    f"activation failed; previous version preserved: {type(exc).__name__}: {exc}",
                    component="ingestion",
                ) from exc

            warnings: List[str] = []
            # ---- Delete retired vectors ------------------------------
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
            # Never keep serving a snapshot whose corpus no longer matches the store.
            self.index_manager.invalidate("ingestion rollback failed")

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    def delete_document(self, doc_id: str, *, tenant_id: str = "default") -> DeleteResult:
        with self._write_lock:
            record = self.registry.get(tenant_id, doc_id)
            scanned = [
                str(doc.metadata.get("chunk_id"))
                for doc in get_indexed_documents(self.store)
                if doc.metadata.get("doc_id") == doc_id and doc.metadata.get("tenant_id", "default") == tenant_id
            ]
            if record is None and not scanned:
                raise DocumentNotFound(f"document '{doc_id}' was not found for this tenant", component="ingestion")
            ids = set(scanned)
            if record:
                for version in record.versions.values():
                    ids.update(version.chunk_ids)
            deleted = 0
            try:
                deleted = delete_chunks(self.store, list(ids))
            except Exception as exc:  # noqa: BLE001
                self.index_manager.invalidate("document deletion could not be applied to the vector store")
                raise IngestionError(
                    f"deletion failed: {type(exc).__name__}: {exc}", component="ingestion"
                ) from exc

            warnings: List[str] = []
            index_refreshed = True
            try:
                self.index_manager.rebuild(get_retrieval_documents(self.store), store=self.store)
            except Exception as exc:  # noqa: BLE001
                # Fail closed: a snapshot that may still contain deleted chunks must not serve queries.
                self.index_manager.invalidate("index refresh after deletion failed")
                index_refreshed = False
                warnings.append(f"index refresh failed: {type(exc).__name__}")

            self.registry.remove(tenant_id, doc_id)
            return DeleteResult(
                doc_id=doc_id,
                tenant_id=tenant_id,
                chunks_deleted=deleted,
                status="deleted" if index_refreshed else "deleted_index_invalidated",
                index_refreshed=index_refreshed,
                warnings=warnings,
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
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
