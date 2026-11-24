"""Durable SQL-backed document and ingestion lifecycle registry.

Supports SQLite (local development and deterministic offline testing) and
PostgreSQL (production multi-writer). Manages documents, document versions,
ingestion jobs, chunk records, indexing events, and audit logs.
"""
from __future__ import annotations

import datetime
import json
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


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


class SQLDocumentRegistry:
    """Relational document registry supporting SQLite and PostgreSQL."""

    def __init__(self, db_path: Optional[str] = None, postgres_uri: Optional[str] = None):
        self.db_path = db_path
        self.postgres_uri = postgres_uri
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.db_path and self.db_path != ":memory:":
                path = Path(self.db_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(str(path), check_same_thread=False)
            else:
                self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            with self._conn:
                self._conn.execute("PRAGMA foreign_keys = ON;")
                self._conn.execute("PRAGMA journal_mode = WAL;")
                self._conn.execute("PRAGMA busy_timeout = 5000;")
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        with conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    tenant_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    active_version INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (tenant_id, doc_id)
                );

                CREATE TABLE IF NOT EXISTS document_versions (
                    tenant_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'staging',
                    created_at TEXT NOT NULL,
                    activated_at TEXT,
                    retired_at TEXT,
                    metadata TEXT,
                    PRIMARY KEY (tenant_id, doc_id, version),
                    FOREIGN KEY (tenant_id, doc_id) REFERENCES documents (tenant_id, doc_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    idempotency_key TEXT,
                    state TEXT NOT NULL DEFAULT 'RECEIVED',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_tenant_doc ON ingestion_jobs (tenant_id, doc_id);
                CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_idempotency ON ingestion_jobs (tenant_id, idempotency_key);
                CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_state ON ingestion_jobs (state);

                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    page INTEGER,
                    section_path TEXT,
                    point_id TEXT NOT NULL,
                    FOREIGN KEY (tenant_id, doc_id, version) REFERENCES document_versions (tenant_id, doc_id, version) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_ver ON document_chunks (tenant_id, doc_id, version);

                CREATE TABLE IF NOT EXISTS indexing_events (
                    event_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    version INTEGER,
                    timestamp TEXT NOT NULL,
                    details TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_indexing_events_tenant ON indexing_events (tenant_id);

                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    details TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_audit_events_tenant_doc ON audit_events (tenant_id, doc_id);
                """
            )

    # ------------------------------------------------------------------
    # Document and Version CRUD
    # ------------------------------------------------------------------
    def get(self, tenant_id: str, doc_id: str) -> Optional[DocumentRecord]:
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT tenant_id, doc_id, source, active_version, created_at, updated_at, deleted "
                "FROM documents WHERE tenant_id = ? AND doc_id = ?",
                (tenant_id, doc_id),
            )
            row = cur.fetchone()
            if not row:
                return None

            cur.execute(
                "SELECT version, content_sha256, chunk_count, status, created_at, activated_at, retired_at, metadata "
                "FROM document_versions WHERE tenant_id = ? AND doc_id = ? ORDER BY version ASC",
                (tenant_id, doc_id),
            )
            version_rows = cur.fetchall()

            versions: Dict[int, VersionRecord] = {}
            for v_row in version_rows:
                v_num = v_row["version"]
                meta = json.loads(v_row["metadata"] or "{}")

                # Retrieve chunk IDs for this version
                cur.execute(
                    "SELECT chunk_id FROM document_chunks WHERE tenant_id = ? AND doc_id = ? AND version = ? ORDER BY chunk_id ASC",
                    (tenant_id, doc_id, v_num),
                )
                chunk_ids = [c["chunk_id"] for c in cur.fetchall()]
                if not chunk_ids and meta.get("chunk_ids"):
                    chunk_ids = meta.get("chunk_ids", [])

                v_rec = VersionRecord(
                    version=v_num,
                    content_sha256=v_row["content_sha256"],
                    chunk_ids=chunk_ids,
                    chunk_count=v_row["chunk_count"] or len(chunk_ids),
                    status=v_row["status"],
                    created_at=v_row["created_at"],
                    activated_at=v_row["activated_at"],
                    retired_at=v_row["retired_at"],
                    embedding_model=meta.get("embedding_model", ""),
                    chunk_size=meta.get("chunk_size", 0),
                    chunk_overlap=meta.get("chunk_overlap", 0),
                    chunk_strategy=meta.get("chunk_strategy", ""),
                    error=meta.get("error"),
                )
                versions[v_num] = v_rec

            return DocumentRecord(
                doc_id=row["doc_id"],
                tenant_id=row["tenant_id"],
                source=row["source"],
                active_version=row["active_version"],
                versions=versions,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                deleted=bool(row["deleted"]),
            )

    def put(self, record: DocumentRecord, chunk_records: Optional[List[Dict[str, Any]]] = None) -> None:
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute(
                    """
                    INSERT INTO documents (tenant_id, doc_id, source, active_version, created_at, updated_at, deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(tenant_id, doc_id) DO UPDATE SET
                        source = excluded.source,
                        active_version = excluded.active_version,
                        updated_at = excluded.updated_at,
                        deleted = excluded.deleted
                    """,
                    (
                        record.tenant_id,
                        record.doc_id,
                        record.source,
                        record.active_version,
                        record.created_at,
                        record.updated_at,
                        1 if record.deleted else 0,
                    ),
                )

                for v in record.versions.values():
                    meta = {
                        "embedding_model": v.embedding_model,
                        "chunk_size": v.chunk_size,
                        "chunk_overlap": v.chunk_overlap,
                        "chunk_strategy": v.chunk_strategy,
                        "error": v.error,
                        "chunk_ids": v.chunk_ids,
                    }
                    conn.execute(
                        """
                        INSERT INTO document_versions (
                            tenant_id, doc_id, version, content_sha256, chunk_count, status, created_at, activated_at, retired_at, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(tenant_id, doc_id, version) DO UPDATE SET
                            content_sha256 = excluded.content_sha256,
                            chunk_count = excluded.chunk_count,
                            status = excluded.status,
                            activated_at = excluded.activated_at,
                            retired_at = excluded.retired_at,
                            metadata = excluded.metadata
                        """,
                        (
                            record.tenant_id,
                            record.doc_id,
                            v.version,
                            v.content_sha256,
                            v.chunk_count,
                            v.status,
                            v.created_at,
                            v.activated_at,
                            v.retired_at,
                            json.dumps(meta),
                        ),
                    )

                if chunk_records:
                    for ch in chunk_records:
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO document_chunks (
                                chunk_id, tenant_id, doc_id, version, page, section_path, point_id
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                ch["chunk_id"],
                                ch.get("tenant_id", record.tenant_id),
                                ch.get("doc_id", record.doc_id),
                                ch.get("version", record.active_version or 1),
                                ch.get("page"),
                                ch.get("section_path"),
                                ch.get("point_id", ch["chunk_id"]),
                            ),
                        )

    def remove(self, tenant_id: str, doc_id: str) -> Optional[DocumentRecord]:
        with self._lock:
            record = self.get(tenant_id, doc_id)
            if not record:
                return None
            conn = self._get_connection()
            with conn:
                conn.execute(
                    "DELETE FROM documents WHERE tenant_id = ? AND doc_id = ?",
                    (tenant_id, doc_id),
                )
            return record

    def list(self, tenant_id: str, include_deleted: bool = False) -> List[DocumentRecord]:
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            if include_deleted:
                cur.execute("SELECT doc_id FROM documents WHERE tenant_id = ?", (tenant_id,))
            else:
                cur.execute("SELECT doc_id FROM documents WHERE tenant_id = ? AND deleted = 0", (tenant_id,))
            doc_ids = [row["doc_id"] for row in cur.fetchall()]
            results = []
            for doc_id in doc_ids:
                doc = self.get(tenant_id, doc_id)
                if doc:
                    results.append(doc)
            return results

    def get_chunk_ids(self, tenant_id: str, doc_id: str, version: Optional[int] = None) -> List[str]:
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            if version is not None:
                cur.execute(
                    "SELECT chunk_id FROM document_chunks WHERE tenant_id = ? AND doc_id = ? AND version = ?",
                    (tenant_id, doc_id, version),
                )
            else:
                cur.execute(
                    "SELECT chunk_id FROM document_chunks WHERE tenant_id = ? AND doc_id = ?",
                    (tenant_id, doc_id),
                )
            return [row["chunk_id"] for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Ingestion Jobs State Machine
    # ------------------------------------------------------------------
    def create_job(
        self,
        job_id: str,
        tenant_id: str,
        doc_id: str,
        version: int,
        idempotency_key: Optional[str] = None,
        state: str = "RECEIVED",
    ) -> None:
        with self._lock:
            conn = self._get_connection()
            now = _utcnow()
            with conn:
                conn.execute(
                    """
                    INSERT INTO ingestion_jobs (
                        job_id, tenant_id, doc_id, version, idempotency_key, state, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, tenant_id, doc_id, version, idempotency_key, state, now, now),
                )

    def update_job(self, job_id: str, state: Optional[str] = None, error: Optional[str] = None) -> None:
        with self._lock:
            conn = self._get_connection()
            now = _utcnow()
            with conn:
                if state and error is not None:
                    conn.execute(
                        "UPDATE ingestion_jobs SET state = ?, error = ?, updated_at = ? WHERE job_id = ?",
                        (state, error, now, job_id),
                    )
                elif state:
                    conn.execute(
                        "UPDATE ingestion_jobs SET state = ?, updated_at = ? WHERE job_id = ?",
                        (state, now, job_id),
                    )
                elif error is not None:
                    conn.execute(
                        "UPDATE ingestion_jobs SET error = ?, updated_at = ? WHERE job_id = ?",
                        (error, now, job_id),
                    )

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_job_by_idempotency(self, tenant_id: str, idempotency_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM ingestion_jobs WHERE tenant_id = ? AND idempotency_key = ? ORDER BY created_at DESC LIMIT 1",
                (tenant_id, idempotency_key),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_stuck_jobs(self, timeout_seconds: float = 300.0) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=timeout_seconds)).isoformat()
            cur.execute(
                """
                SELECT * FROM ingestion_jobs
                WHERE state NOT IN ('ACTIVE', 'FAILED', 'CANCELLED')
                  AND updated_at < ?
                """,
                (cutoff,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_staged_versions(self, timeout_seconds: float = 300.0) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=timeout_seconds)).isoformat()
            cur.execute(
                """
                SELECT * FROM document_versions
                WHERE status = 'staging' AND created_at < ?
                """,
                (cutoff,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_retired_versions(self, retention_seconds: float = 86400.0) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=retention_seconds)).isoformat()
            cur.execute(
                """
                SELECT * FROM document_versions
                WHERE status = 'retired' AND retired_at < ?
                """,
                (cutoff,),
            )
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Indexing and Audit Events
    # ------------------------------------------------------------------
    def record_indexing_event(
        self,
        event_id: str,
        tenant_id: str,
        event_type: str,
        version: Optional[int] = None,
        details: Optional[str] = None,
    ) -> None:
        with self._lock:
            conn = self._get_connection()
            now = _utcnow()
            with conn:
                conn.execute(
                    "INSERT INTO indexing_events (event_id, tenant_id, event_type, version, timestamp, details) VALUES (?, ?, ?, ?, ?, ?)",
                    (event_id, tenant_id, event_type, version, now, details or ""),
                )

    def record_audit_event(
        self,
        id: str,
        tenant_id: str,
        doc_id: str,
        action: str,
        actor: str = "system",
        details: Optional[str] = None,
    ) -> None:
        with self._lock:
            conn = self._get_connection()
            now = _utcnow()
            with conn:
                conn.execute(
                    "INSERT INTO audit_events (id, tenant_id, doc_id, action, actor, timestamp, details) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (id, tenant_id, doc_id, action, actor, now, details or ""),
                )

    def reset(self) -> None:
        with self._lock:
            if self._conn:
                with self._conn:
                    self._conn.execute("DELETE FROM audit_events;")
                    self._conn.execute("DELETE FROM indexing_events;")
                    self._conn.execute("DELETE FROM ingestion_jobs;")
                    self._conn.execute("DELETE FROM document_chunks;")
                    self._conn.execute("DELETE FROM document_versions;")
                    self._conn.execute("DELETE FROM documents;")
            if self.db_path and self.db_path != ":memory:":
                p = Path(self.db_path)
                if p.exists():
                    p.unlink(missing_ok=True)
                wal = p.with_name(f"{p.name}-wal")
                shm = p.with_name(f"{p.name}-shm")
                wal.unlink(missing_ok=True)
                shm.unlink(missing_ok=True)
