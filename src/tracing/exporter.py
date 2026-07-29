import sqlite3
import datetime
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.config import settings

class TraceRecord(BaseModel):
    query_id: str
    stage: str
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model_name: str = "n/a"
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

class TraceExporter:
    def __init__(self, postgres_uri: Optional[str] = None):
        self.postgres_uri = postgres_uri or settings.POSTGRES_URI
        self.sqlite_db_path = "traces_local.db"
        self._init_sqlite()

    def _init_sqlite(self):
        try:
            conn = sqlite3.connect(self.sqlite_db_path)
            cursor = conn.cursor()
            cursor.execute("""
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
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    recall_at_k REAL,
                    precision_at_k REAL,
                    mrr REAL,
                    ndcg_at_k REAL,
                    faithfulness REAL,
                    relevancy REAL,
                    context_precision REAL,
                    status TEXT NOT NULL
                );
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[TraceExporter] Error initializing local SQLite trace DB: {e}")

    def export(self, record: TraceRecord) -> None:
        # First try Postgres
        pg_success = False
        try:
            import psycopg2
            conn = psycopg2.connect(self.postgres_uri, connect_timeout=1)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO traces (query_id, stage, latency_ms, tokens_in, tokens_out, cost_usd, model_name, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """, (record.query_id, record.stage, record.latency_ms, record.tokens_in, record.tokens_out, record.cost_usd, record.model_name, record.timestamp))
            conn.commit()
            conn.close()
            pg_success = True
        except Exception:
            pass

        # Always duplicate / write to SQLite fallback
        try:
            conn = sqlite3.connect(self.sqlite_db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO traces (query_id, stage, latency_ms, tokens_in, tokens_out, cost_usd, model_name, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (record.query_id, record.stage, record.latency_ms, record.tokens_in, record.tokens_out, record.cost_usd, record.model_name, record.timestamp))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[TraceExporter] Local SQLite trace write failed: {e}")

    def get_recent_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        # Try Postgres first
        try:
            import psycopg2
            conn = psycopg2.connect(self.postgres_uri, connect_timeout=1)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT query_id, stage, latency_ms, tokens_in, tokens_out, cost_usd, model_name, timestamp
                FROM traces ORDER BY id DESC LIMIT %s;
            """, (limit,))
            rows = cursor.fetchall()
            conn.close()
            return [
                {
                    "query_id": r[0], "stage": r[1], "latency_ms": r[2],
                    "tokens_in": r[3], "tokens_out": r[4], "cost_usd": r[5],
                    "model_name": r[6], "timestamp": r[7]
                }
                for r in rows
            ]
        except Exception:
            pass

        # Fallback SQLite
        try:
            conn = sqlite3.connect(self.sqlite_db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT query_id, stage, latency_ms, tokens_in, tokens_out, cost_usd, model_name, timestamp
                FROM traces ORDER BY id DESC LIMIT ?;
            """, (limit,))
            rows = cursor.fetchall()
            conn.close()
            return [
                {
                    "query_id": r[0], "stage": r[1], "latency_ms": r[2],
                    "tokens_in": r[3], "tokens_out": r[4], "cost_usd": r[5],
                    "model_name": r[6], "timestamp": r[7]
                }
                for r in rows
            ]
        except Exception as e:
            print(f"[TraceExporter] Local SQLite trace fetch failed: {e}")
            return []

_exporter_instance: Optional[TraceExporter] = None

def get_exporter() -> TraceExporter:
    global _exporter_instance
    if _exporter_instance is None:
        _exporter_instance = TraceExporter()
    return _exporter_instance
