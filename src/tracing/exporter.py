"""Bounded, asynchronous trace export.

The previous exporter opened a PostgreSQL connection *and* wrote SQLite for
every span, synchronously, on the request path, with no queue, timeout policy
or failure accounting. This version writes to exactly one configured sink via a
bounded queue and batch writer. A full queue drops records and increments a
metric instead of blocking the request; dropped records are never reported as
persisted.
"""
from __future__ import annotations

import datetime
import queue
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.config import settings

_EXTRA_COLUMNS = {
    "usage_source": "TEXT DEFAULT 'unavailable'",
    "index_version": "INTEGER",
    "candidate_count": "INTEGER",
    "rerank_status": "TEXT",
    "error_category": "TEXT",
    "status": "TEXT DEFAULT 'ok'",
    "cost_source": "TEXT DEFAULT 'estimated_static_pricing'",
    "tenant_id": "TEXT DEFAULT 'default'",
    "subject_id": "TEXT",
    "request_id": "TEXT",
    "credential_id": "TEXT",
    "auth_method": "TEXT",
}


class TraceRecord(BaseModel):
    query_id: str
    stage: str
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model_name: str = "n/a"
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    usage_source: str = "unavailable"
    index_version: Optional[int] = None
    candidate_count: Optional[int] = None
    rerank_status: Optional[str] = None
    error_category: Optional[str] = None
    status: str = "ok"
    cost_source: str = "estimated_static_pricing"
    tenant_id: str = "default"
    subject_id: Optional[str] = None
    request_id: Optional[str] = None
    credential_id: Optional[str] = None
    auth_method: Optional[str] = None


class TraceExporter:
    def __init__(
        self,
        postgres_uri: Optional[str] = None,
        *,
        sink: Optional[str] = None,
        db_path: Optional[str] = None,
        queue_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        flush_interval: Optional[float] = None,
    ):
        self.sink = (sink or settings.TRACE_SINK).lower()
        self.postgres_uri = postgres_uri or settings.POSTGRES_URI
        self.sqlite_db_path = db_path or settings.TRACE_DB_PATH
        self.queue_size = queue_size or settings.TRACE_QUEUE_SIZE
        self.batch_size = batch_size or settings.TRACE_BATCH_SIZE
        self.flush_interval = flush_interval or settings.TRACE_FLUSH_INTERVAL_SECONDS
        self._queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._stop = threading.Event()
        self._drained = threading.Event()
        self._drained.set()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._pg_connection = None
        self.written = 0
        self.dropped = 0
        self.failed = 0
        self.last_error: Optional[str] = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def export(self, record: TraceRecord) -> bool:
        if self.sink == "none":
            return False
        self._ensure_worker()
        try:
            self._queue.put_nowait(record)
            self._drained.clear()
            return True
        except queue.Full:
            with self._lock:
                self.dropped += 1
            try:
                from src.observability.metrics import metrics
                metrics.inc("rag_trace_queue_dropped_total", value=1.0)
                metrics.inc("rag_traces_dropped_total", {"reason": "queue_full"}, value=1.0)
            except Exception:
                pass
            return False

    def flush(self, timeout: float = 5.0) -> bool:
        if self._thread is None:
            return True
        if self._drained.wait(timeout):
            return True
        return self._queue.empty()

    def close(self, timeout: float = 5.0) -> None:
        """Flushes buffered traces, stops the background worker, and cleans up connections."""
        self.flush(timeout=timeout)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

        remaining_count = self._queue.qsize()
        if remaining_count > 0:
            try:
                self._drain_remaining()
            except Exception as exc:
                lost = self._queue.qsize()
                if lost > 0:
                    try:
                        from src.observability.metrics import metrics
                        metrics.inc("rag_traces_lost_on_shutdown_total", value=float(lost))
                        metrics.inc("rag_traces_dropped_total", {"reason": "shutdown_timeout"}, value=float(lost))
                    except Exception:
                        pass
        if self._pg_connection is not None:
            try:
                self._pg_connection.close()
            except Exception:  # noqa: BLE001
                pass
            self._pg_connection = None

    def status(self) -> Dict[str, Any]:
        depth = self._queue.qsize()
        drops = self.dropped
        health = "healthy"
        warning = None
        if drops > 0:
            health = "degraded"
            warning = f"Exporter dropped {drops} traces due to queue overflow."
        elif depth > (self.queue_size * 0.8):
            health = "warning"
            warning = f"Exporter queue is nearing capacity ({depth}/{self.queue_size})."
        return {
            "sink": self.sink,
            "status": health,
            "warning": warning,
            "queue_depth": depth,
            "queue_capacity": self.queue_size,
            "written": self.written,
            "dropped": drops,
            "failed": self.failed,
            "worker_alive": bool(self._thread and self._thread.is_alive()),
            "last_error": self.last_error,
            "persistent": self.sink in {"sqlite", "postgres"},
        }

    def get_recent_traces(self, limit: int = 50, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        self.flush(timeout=2.0)
        try:
            if self.sink == "postgres":
                results = self._read_postgres(limit)
            elif self.sink == "sqlite":
                results = self._read_sqlite(limit)
            else:
                results = []
            if tenant_id:
                results = [r for r in results if r.get("tenant_id") == tenant_id]
            return results
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
        return []

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _ensure_worker(self) -> None:
        if self._thread is not None or self.sink == "none":
            return
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="trace-exporter", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        try:
            try:
                self._initialize_sink()
            except Exception as exc:  # noqa: BLE001 - keep the worker alive to account for drops
                with self._lock:
                    self.last_error = f"sink initialization failed: {type(exc).__name__}: {exc}"
                self._update_metrics()
            while not self._stop.is_set():
                batch = self._collect_batch()
                if batch:
                    self._write_batch(batch)
                if self._queue.empty():
                    self._drained.set()
        finally:
            self._drained.set()

    def _collect_batch(self) -> List[TraceRecord]:
        batch: List[TraceRecord] = []
        try:
            batch.append(self._queue.get(timeout=self.flush_interval))
        except queue.Empty:
            return batch
        while len(batch) < self.batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _drain_remaining(self) -> None:
        remaining = []
        while True:
            try:
                remaining.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if remaining:
            self._write_batch(remaining)

    def _initialize_sink(self) -> None:
        if self._initialized:
            return
        if self.sink == "sqlite":
            self._init_sqlite()
            self._prune_sqlite()
        elif self.sink == "postgres":
            self._init_postgres()
        self._initialized = True

    def _write_batch(self, batch: List[TraceRecord]) -> None:
        if not batch:
            return
        try:
            if self.sink == "sqlite":
                self._write_sqlite(batch)
            elif self.sink == "postgres":
                self._write_postgres(batch)
            else:
                return
            with self._lock:
                self.written += len(batch)
            try:
                from src.observability.metrics import metrics
                metrics.inc("rag_traces_exported_total", {"status": "success"}, value=float(len(batch)))
            except Exception:
                pass
            self._update_metrics()
        except Exception as exc:  # noqa: BLE001 - counted and surfaced, never retried forever
            with self._lock:
                self.failed += len(batch)
                self.last_error = f"{type(exc).__name__}: {exc}"
            try:
                from src.observability.metrics import metrics
                metrics.inc("rag_traces_exported_total", {"status": "failure"}, value=float(len(batch)))
            except Exception:
                pass
            self._update_metrics()

    def _update_metrics(self) -> None:
        try:
            from src.observability.metrics import metrics

            metrics.set_gauge("rag_trace_queue_depth", self._queue.qsize())
            metrics.set_gauge("rag_trace_dropped_total", self.dropped)
            metrics.set_gauge("rag_trace_failed_total", self.failed)
            metrics.set_gauge("rag_trace_written_total", self.written)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # SQLite
    # ------------------------------------------------------------------
    def _sqlite_connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_db_path, timeout=5.0)
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _init_sqlite(self) -> None:
        connection = self._sqlite_connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    tokens_in INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0.0,
                    model_name TEXT DEFAULT 'n/a',
                    timestamp TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL, timestamp TEXT NOT NULL,
                    recall_at_k REAL, precision_at_k REAL, mrr REAL, ndcg_at_k REAL,
                    faithfulness REAL, relevancy REAL, context_precision REAL, status TEXT NOT NULL
                )
                """
            )
            existing = {row[1] for row in cursor.execute("PRAGMA table_info(traces)")}
            for name, ddl in _EXTRA_COLUMNS.items():
                if name not in existing:
                    cursor.execute(f"ALTER TABLE traces ADD COLUMN {name} {ddl}")
            connection.commit()
        finally:
            connection.close()

    def _prune_sqlite(self) -> None:
        if settings.TRACE_RETENTION_DAYS <= 0:
            return
        cutoff = (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=settings.TRACE_RETENTION_DAYS)
        ).isoformat()
        connection = self._sqlite_connect()
        try:
            connection.execute("DELETE FROM traces WHERE timestamp < ?", (cutoff,))
            connection.commit()
        finally:
            connection.close()

    def _write_sqlite(self, batch: List[TraceRecord]) -> None:
        connection = self._sqlite_connect()
        try:
            connection.executemany(
                """
                INSERT INTO traces (
                    query_id, stage, latency_ms, tokens_in, tokens_out, cost_usd, model_name, timestamp,
                    usage_source, index_version, candidate_count, rerank_status, error_category, status, cost_source,
                    tenant_id, subject_id, request_id, credential_id, auth_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.query_id, r.stage, r.latency_ms, r.tokens_in, r.tokens_out, r.cost_usd,
                        r.model_name, r.timestamp, r.usage_source, r.index_version, r.candidate_count,
                        r.rerank_status, r.error_category, r.status, r.cost_source,
                        r.tenant_id, r.subject_id, r.request_id, r.credential_id, r.auth_method,
                    )
                    for r in batch
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def _read_sqlite(self, limit: int) -> List[Dict[str, Any]]:
        connection = self._sqlite_connect()
        try:
            rows = connection.execute(
                """
                SELECT query_id, stage, latency_ms, tokens_in, tokens_out, cost_usd, model_name, timestamp,
                       usage_source, index_version, candidate_count, rerank_status, error_category, status, cost_source,
                       tenant_id, subject_id, request_id
                FROM traces ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            connection.close()
        keys = [
            "query_id", "stage", "latency_ms", "tokens_in", "tokens_out", "cost_usd", "model_name",
            "timestamp", "usage_source", "index_version", "candidate_count", "rerank_status",
            "error_category", "status", "cost_source", "tenant_id", "subject_id", "request_id",
        ]
        return [dict(zip(keys, row)) for row in rows]

    # ------------------------------------------------------------------
    # PostgreSQL (single sink; no duplicate SQLite write)
    # ------------------------------------------------------------------
    def _init_postgres(self) -> None:
        import psycopg2  # noqa: F401

        connection = psycopg2.connect(self.postgres_uri, connect_timeout=3)
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                id SERIAL PRIMARY KEY, query_id VARCHAR(100) NOT NULL, stage VARCHAR(50) NOT NULL,
                latency_ms DOUBLE PRECISION NOT NULL, tokens_in INTEGER DEFAULT 0, tokens_out INTEGER DEFAULT 0,
                cost_usd DOUBLE PRECISION DEFAULT 0.0, model_name VARCHAR(100) DEFAULT 'n/a',
                timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP, usage_source TEXT DEFAULT 'unavailable',
                index_version INTEGER, candidate_count INTEGER, rerank_status TEXT, error_category TEXT,
                status TEXT DEFAULT 'ok', cost_source TEXT DEFAULT 'estimated_static_pricing'
            )
            """
        )
        for name, ddl in _EXTRA_COLUMNS.items():
            cursor.execute(f"ALTER TABLE traces ADD COLUMN IF NOT EXISTS {name} {ddl}")
        connection.commit()
        self._pg_connection = connection

    def _write_postgres(self, batch: List[TraceRecord]) -> None:
        if self._pg_connection is None:
            self._init_postgres()
        cursor = self._pg_connection.cursor()
        cursor.executemany(
            """
            INSERT INTO traces (query_id, stage, latency_ms, tokens_in, tokens_out, cost_usd, model_name, timestamp,
                usage_source, index_version, candidate_count, rerank_status, error_category, status, cost_source,
                tenant_id, subject_id, request_id, credential_id, auth_method)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            [
                (
                    r.query_id, r.stage, r.latency_ms, r.tokens_in, r.tokens_out, r.cost_usd,
                    r.model_name, r.timestamp, r.usage_source, r.index_version, r.candidate_count,
                    r.rerank_status, r.error_category, r.status, r.cost_source,
                    r.tenant_id, r.subject_id, r.request_id, r.credential_id, r.auth_method,
                )
                for r in batch
            ],
        )
        self._pg_connection.commit()

    def _read_postgres(self, limit: int) -> List[Dict[str, Any]]:
        if self._pg_connection is None:
            self._init_postgres()
        cursor = self._pg_connection.cursor()
        cursor.execute(
            """
            SELECT query_id, stage, latency_ms, tokens_in, tokens_out, cost_usd, model_name, timestamp,
                   usage_source, index_version, candidate_count, rerank_status, error_category, status, cost_source
            FROM traces ORDER BY id DESC LIMIT %s
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        keys = [
            "query_id", "stage", "latency_ms", "tokens_in", "tokens_out", "cost_usd", "model_name",
            "timestamp", "usage_source", "index_version", "candidate_count", "rerank_status",
            "error_category", "status", "cost_source",
        ]
        return [dict(zip(keys, row)) for row in rows]


_exporter_instance: Optional[TraceExporter] = None
_exporter_lock = threading.Lock()


def get_exporter() -> TraceExporter:
    global _exporter_instance
    if _exporter_instance is None:
        with _exporter_lock:
            if _exporter_instance is None:
                _exporter_instance = TraceExporter()
    return _exporter_instance


def reset_exporter(exporter: Optional[TraceExporter] = None) -> None:
    global _exporter_instance
    with _exporter_lock:
        if _exporter_instance is not None and exporter is None:
            _exporter_instance.close()
        _exporter_instance = exporter
