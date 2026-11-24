-- Migration 001: Transactional Document Lifecycle Schema
-- Compatible with PostgreSQL and SQLite

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
