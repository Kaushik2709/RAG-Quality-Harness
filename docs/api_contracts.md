# REST API Contracts & Schema Specification

The RAG Platform exposes a versioned, authenticated REST API conforming to OpenAPI 3.1 specifications.

---

## 1. Authentication & Common Headers

| Header | Type | Required | Description |
|---|---|---|---|
| `X-API-Key` | string | Optional (Configurable) | API secret token. Required when `API_KEY` is configured in runtime settings. |
| `X-Tenant-ID` | string | Optional | Tenant partition identifier. Defaults to `"default"`. Enforces tenant isolation. |
| `X-Request-ID` | string | Optional | Correlation identifier. Auto-generated via UUIDv4 if omitted. |

---

## 2. Standard Structured Error Format

All `4xx` and `5xx` responses return a standardized structured JSON error payload conforming to `ErrorBody`:

```json
{
  "code": "invalid_credentials",
  "message": "Invalid API key provided",
  "component": "authentication",
  "retryable": false,
  "request_id": "req-9a8b7c6d-5e4f"
}
```

### Error Code Taxonomy
- `invalid_credentials` (401): Missing or invalid `X-API-Key`.
- `permission_denied` (403): Request role insufficient for requested resource (e.g. reader attempting ingest).
- `tenant_mismatch` (403): Cross-tenant access attempt detected.
- `not_found` (404): Requested document or resource does not exist.
- `validation_error` (422): Request schema or parameters failed Pydantic validation.
- `document_too_large` (413): Document exceeds maximum character limit (`MAX_DOCUMENT_CHARS`).
- `unsupported_format` (415): File extension or MIME type not in allowed parser list.
- `rate_limit_exceeded` (429): Token bucket limit exhausted; returns `Retry-After` header.
- `service_unavailable` (503): Critical dependency (e.g. Qdrant or required model) not ready.

---

## 3. Endpoints

### 3.1 Query Execution

`POST /query`

Executes the grounded RAG retrieval, synthesis, and citation verification pipeline.

#### Request Schema
```json
{
  "query": "What is the Nova service authentication timeout?",
  "top_k": 5,
  "top_k_rerank": 3,
  "use_hybrid": true,
  "retrieval_mode": "hybrid"
}
```

#### Response Schema (200 OK)
```json
{
  "answer": "The Nova service authentication timeout is 45 seconds for all production environments.",
  "citations": [
    {
      "chunk_id": "nova-policy:v1:chunk-0000",
      "quote": "authentication timeout is 45 seconds for all production environments"
    }
  ],
  "abstained": false,
  "abstention_reason": null,
  "retrieved_chunks": [
    {
      "chunk_id": "nova-policy:v1:chunk-0000",
      "doc_id": "nova-policy",
      "content": "The Nova service authentication timeout is 45 seconds...",
      "metadata": {
        "tenant_id": "default",
        "version_state": "active"
      }
    }
  ],
  "retrieval_metadata": {
    "mode": "hybrid",
    "index_version": 4,
    "candidate_count": 10,
    "stage_latency_ms": {"dense": 12.4, "bm25": 4.1, "fusion": 0.8}
  },
  "citation_validation": {
    "status": "validated",
    "citation_precision": 1.0,
    "citation_recall": 1.0,
    "rejected": []
  },
  "usage": {
    "input_tokens": 142,
    "output_tokens": 28,
    "source": "provider"
  },
  "request_id": "req-12345"
}
```

---

### 3.2 Document Ingestion

`POST /ingest`

Ingests raw text with automatic chunking, staging, verification, and atomic index refresh.

#### Request Schema
```json
{
  "text": "Detailed operational guide...",
  "doc_id": "atlas-guide-01",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "metadata": {"author": "infra-team"}
}
```

#### Response Schema (200 OK)
```json
{
  "doc_id": "atlas-guide-01",
  "version": 1,
  "chunks_created": 3,
  "status": "created",
  "content_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "duplicate": false,
  "activated": true,
  "index_version": 5,
  "warnings": [],
  "tenant_id": "default"
}
```

---

### 3.3 File Upload Ingestion

`POST /ingest/file`

Multipart form upload supporting `.txt`, `.md`, `.pdf`, and `.html`.

- **Form Fields:** `file` (binary), `doc_id` (string, optional), `chunk_size` (int, optional).
- **Enforces:** File size limits, temporary file cleanup, parser sandboxing.

---

### 3.4 Document Management & Lifecycle

- `GET /documents`: List active registered documents and versions for current tenant.
- `DELETE /documents/{id}`: Atomically deletes document, removes all associated vector/BM25 chunks, and invalidates/refreshes the index.

---

### 3.5 Operational & Health Endpoints

- `GET /ready`: Deep readiness probe returning HTTP 200 when all required models and vector store are ready; returns HTTP 503 when critical components fail.
- `GET /runtime/status`: Live diagnostic dashboard metrics: index version, active chunk count, store backend, model lifecycle states, and telemetry queue metrics.
- `GET /metrics`: Prometheus-compatible exposition format metrics.
- `POST /index/refresh`: Triggers an atomic background rebuild and hot-swap of the retrieval index.
- `GET /traces`: Inspect historical query waterfall traces and stage latencies.
