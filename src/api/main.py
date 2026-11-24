"""FastAPI application: ingestion, query, runtime status, telemetry and evals.

Backward-compatible routes are preserved (`/`, `/ingest`, `/ingest/file`,
`/query`, `/traces`, `/evals/reports`, `/evals/run`). New routes expose index
lifecycle, document lifecycle, readiness and Prometheus metrics.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import ValidationError

from src.api.dependencies import get_identity, get_limiter
from src.api.schemas import (
    CitationResponse,
    ContextChunkResponse,
    DeleteResponse,
    DocumentSummary,
    IngestResponse,
    IngestTextRequest,
    IndexRefreshResponse,
    QueryRequest,
    QueryResponse,
)
from src.config import settings
from src.ingestion.ingestion_service import IngestionService
from src.observability.metrics import metrics
from src.pipeline.graph import compile_rag_graph
from src.retrieval.retrieval_runtime import get_retrieval_runtime
from src.retrieval.vector_store import get_vector_store
from src.runtime import lifecycle
from src.runtime.errors import (
    AuthorizationError,
    DocumentNotFound,
    DocumentTooLarge,
    IngestionError,
    RuntimeFailure,
    UnsupportedDocument,
)
from src.runtime.readiness import collect_readiness
from src.security.authentication import Identity
from src.security.authorization import require_role, require_tenant
from src.tracing.exporter import get_exporter

logger = logging.getLogger("rag.api")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    lifecycle.startup()
    try:
        yield
    finally:
        lifecycle.shutdown()


app = FastAPI(
    title="RAG Reliability, Evaluation & Optimization Platform API",
    version=settings.SERVICE_VERSION,
    description="Versioned retrieval, grounded generation, evaluation and observability for RAG systems.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Tenant-ID", "X-Request-ID"],
)

rag_graph = compile_rag_graph()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ingestion_services: Dict[int, IngestionService] = {}


def _ingestion_service() -> IngestionService:
    store = get_vector_store()
    cached = _ingestion_services.get(id(store))
    if cached is not None and cached.store is store:
        return cached
    embeddings = getattr(store, "embedding", None) or getattr(store, "embeddings", None)
    service = IngestionService(store, embeddings=embeddings, index_manager=get_retrieval_runtime().index_manager)
    _ingestion_services[id(store)] = service
    return service


# ----------------------------------------------------------------------
# Error contract
# ----------------------------------------------------------------------
def _error_response(request: Request, exc: RuntimeFailure) -> JSONResponse:
    return JSONResponse(
        status_code=exc.http_status,
        content={
            "error": {
                "code": exc.code,
                "message": str(exc),
                "component": exc.component,
                "retryable": exc.retryable,
                "request_id": getattr(request.state, "request_id", None),
            }
        },
    )


@app.exception_handler(RuntimeFailure)
async def runtime_failure_handler(request: Request, exc: RuntimeFailure):
    return _error_response(request, exc)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "request_validation_error",
                "message": "request validation failed",
                "component": "api",
                "retryable": False,
                "details": exc.errors(),
                "request_id": getattr(request.state, "request_id", None),
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    logger.exception("unhandled request failure")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "internal error; see server logs with the request id",
                "component": "api",
                "retryable": False,
                "request_id": getattr(request.state, "request_id", None),
            }
        },
    )


# ----------------------------------------------------------------------
# Request middleware: correlation id, bounded rate limit, HTTP metrics
# ----------------------------------------------------------------------
@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4()}"
    request.state.request_id = request_id
    path = request.url.path
    exempt = path in {"/health", "/ready", "/metrics", "/openapi.json"} or path.startswith("/docs")
    if not exempt and settings.RATE_LIMIT_PER_MINUTE > 0:
        key = request.headers.get("X-API-Key") or (request.client.host if request.client else "unknown")
        allowed, retry_after = get_limiter().allow(key)
        if not allowed:
            metrics.inc("rag_rate_limited_total", {"path": path})
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "rate_limited",
                        "message": "request rate limit exceeded",
                        "component": "api",
                        "retryable": True,
                        "request_id": request_id,
                    }
                },
                headers={"Retry-After": f"{max(1, int(retry_after))}"},
            )
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    metrics.inc("rag_http_requests_total", {"method": request.method, "route": path, "status": str(response.status_code)})
    metrics.observe("rag_http_request_duration_ms", elapsed_ms, {"method": request.method, "route": path})
    return response


# ----------------------------------------------------------------------
# Health / metadata
# ----------------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "RAG Reliability, Evaluation & Optimization Platform",
        "version": settings.SERVICE_VERSION,
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker_model": settings.RERANKER_MODEL,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.GEMINI_MODEL if settings.LLM_PROVIDER == "gemini" else settings.CLAUDE_MODEL,
        "authentication": "enabled" if settings.API_KEY else "disabled",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Liveness only: this must not depend on external services."""
    return {"status": "ok", "version": settings.SERVICE_VERSION}


@app.get("/ready")
def ready(deep: bool = Query(default=False, description="Probe models/index; may initialise resources")):
    report = collect_readiness(deep=deep)
    status_code = 200 if report["ready"] else 503
    return JSONResponse(status_code=status_code, content=report)


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


# ----------------------------------------------------------------------
# Ingestion & documents
# ----------------------------------------------------------------------
@app.post("/ingest", response_model=IngestResponse)
def ingest_text(req: IngestTextRequest, identity: Identity = Depends(get_identity)):
    require_role(identity, "ingestor")
    service = _ingestion_service()
    try:
        result = service.ingest_text(
            req.text,
            doc_id=req.doc_id,
            source=req.source or "api_text",
            tenant_id=identity.tenant_id,
            metadata=req.metadata,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            strategy=req.strategy,
        )
    except RuntimeFailure:
        metrics.inc("rag_ingestion_total", {"status": "failure"})
        raise
    except Exception as exc:  # noqa: BLE001
        metrics.inc("rag_ingestion_total", {"status": "failure"})
        raise IngestionError(f"ingestion failed: {type(exc).__name__}: {exc}", component="ingestion") from exc
    metrics.inc("rag_ingestion_total", {"status": "duplicate" if result.duplicate else "success"})
    if result.index_version is not None:
        metrics.set_gauge("rag_index_version", result.index_version)
        metrics.set_gauge("rag_index_chunks", get_retrieval_runtime().status().get("chunk_count") or 0)
    return IngestResponse(
        doc_id=result.doc_id,
        chunks_created=result.chunks_created,
        status=result.status,
        version=result.version,
        content_sha256=result.content_sha256,
        duplicate=result.duplicate,
        activated=result.activated,
        index_version=result.index_version,
        tenant_id=result.tenant_id,
        warnings=result.warnings,
    )


_TEXT_CONTENT_TYPES = {
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {"text/markdown", "text/plain", "application/octet-stream"},
    ".markdown": {"text/markdown", "text/plain", "application/octet-stream"},
    ".html": {"text/html", "application/xhtml+xml", "application/octet-stream"},
    ".htm": {"text/html", "application/xhtml+xml", "application/octet-stream"},
    ".pdf": {"application/pdf", "application/octet-stream"},
}


def _validate_upload(filename: str, content_type: Optional[str]) -> str:
    filename = os.path.basename(filename or "upload.txt")[:255]
    ext = os.path.splitext(filename)[1].lower()
    allowed = {item.strip().lower() for item in settings.ALLOWED_UPLOAD_EXTENSIONS.split(",") if item.strip()}
    if ext not in allowed:
        raise UnsupportedDocument(f"extension '{ext}' is not allowed", component="upload")
    if content_type:
        base = content_type.split(";")[0].strip().lower()
        if base not in _TEXT_CONTENT_TYPES.get(ext, set()):
            raise UnsupportedDocument(
                f"content-type '{base}' is not allowed for {ext} files", component="upload"
            )
    return ext


async def _write_upload_bounded(file: UploadFile, ext: str) -> str:
    total = 0
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    path = handle.name
    try:
        with handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > settings.MAX_UPLOAD_BYTES:
                    raise DocumentTooLarge(
                        f"upload exceeds MAX_UPLOAD_BYTES={settings.MAX_UPLOAD_BYTES}", component="upload"
                    )
                handle.write(chunk)
        if total == 0:
            raise IngestionError("uploaded file is empty", component="upload")
        if ext == ".pdf":
            with open(path, "rb") as probe:
                if probe.read(5) != b"%PDF-":
                    raise UnsupportedDocument("file does not have a PDF signature", component="upload")
        return path
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        raise


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(..., description="PDF, TXT, MD, or HTML document"),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    strategy: Optional[str] = Form(None),
    doc_id: Optional[str] = Form(None),
    identity: Identity = Depends(get_identity),
):
    require_role(identity, "ingestor")
    filename = os.path.basename(file.filename or "uploaded_file")[:255]
    ext = _validate_upload(filename, file.content_type)
    tmp_path = None
    try:
        async with asyncio.timeout(settings.UPLOAD_TIMEOUT_SECONDS):
            tmp_path = await _write_upload_bounded(file, ext)
        result = _ingestion_service().ingest_file(
            tmp_path,
            filename=filename,
            doc_id=doc_id,
            tenant_id=identity.tenant_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
        )
    except TimeoutError as exc:
        metrics.inc("rag_ingestion_total", {"status": "timeout"})
        raise IngestionError("upload exceeded the configured timeout", component="upload") from exc
    except RuntimeFailure:
        metrics.inc("rag_ingestion_total", {"status": "failure"})
        raise
    except Exception as exc:  # noqa: BLE001
        metrics.inc("rag_ingestion_total", {"status": "failure"})
        raise IngestionError(f"file ingestion failed: {type(exc).__name__}: {exc}", component="ingestion") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                logger.warning("failed to remove temporary upload")
    metrics.inc("rag_ingestion_total", {"status": "duplicate" if result.duplicate else "success"})
    return IngestResponse(
        doc_id=result.doc_id,
        chunks_created=result.chunks_created,
        status=result.status,
        version=result.version,
        content_sha256=result.content_sha256,
        duplicate=result.duplicate,
        activated=result.activated,
        index_version=result.index_version,
        tenant_id=result.tenant_id,
        warnings=result.warnings,
    )


@app.get("/documents", response_model=List[DocumentSummary])
def list_documents(identity: Identity = Depends(get_identity)):
    require_role(identity, "reader")
    return _ingestion_service().list_documents(tenant_id=identity.tenant_id)


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str, identity: Identity = Depends(get_identity)):
    require_role(identity, "admin")
    result = _ingestion_service().delete_document(doc_id, tenant_id=identity.tenant_id)
    metrics.inc("rag_document_deletions_total", {"status": result.status})
    if result.index_refreshed:
        status = get_retrieval_runtime().status()
        metrics.set_gauge("rag_index_chunks", status.get("chunk_count") or 0)
    return DeleteResponse(**result.__dict__)


# ----------------------------------------------------------------------
# Query
# ----------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest, identity: Identity = Depends(get_identity)):
    require_role(identity, "reader")
    query_id = f"q-{uuid.uuid4()}"
    started = time.perf_counter()
    initial_state = {
        "query": req.query,
        "query_id": query_id,
        "tenant_id": identity.tenant_id,
        "documents": [],
        "reranked_documents": [],
        "generation": "",
        "top_k": req.top_k,
        "top_k_rerank": req.top_k_rerank,
        "use_hybrid": req.use_hybrid,
        "retrieval_mode": req.retrieval_mode,
    }
    try:
        final_state = rag_graph.invoke(initial_state)
    except RuntimeFailure:
        metrics.inc("rag_query_total", {"status": "error"})
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("query pipeline failed query_id=%s", query_id)
        metrics.inc("rag_query_total", {"status": "error"})
        raise IngestionError(f"query pipeline failed: {type(exc).__name__}", component="pipeline") from exc
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    metrics.inc("rag_query_total", {"status": final_state.get("generation_status", "unknown")})

    scores: Dict[str, float] = {}
    retrieval = final_state.get("retrieval") or {}
    for key in ("fused", "scores"):
        pass
    if final_state.get("rerank", {}).get("status") == "applied":
        scores.update(final_state.get("rerank", {}).get("scores", {}))
    context_docs = final_state.get("reranked_documents", [])
    context_resp = [
        ContextChunkResponse(
            chunk_id=str(doc.metadata.get("chunk_id", f"{doc.metadata.get('doc_id', 'doc')}_c{idx}")),
            doc_id=str(doc.metadata.get("doc_id", "unknown")),
            text=doc.page_content,
            score=scores.get(str(doc.metadata.get("chunk_id", ""))),
            document_version=doc.metadata.get("document_version"),
            section_path=doc.metadata.get("section_path") or None,
            page=doc.metadata.get("page"),
        )
        for idx, doc in enumerate(context_docs)
    ]
    structured = final_state.get("structured_answer") or {}
    citations = [CitationResponse(**item) for item in structured.get("citations", [])]
    return QueryResponse(
        query_id=query_id,
        query=req.query,
        answer=final_state.get("generation", ""),
        retrieved_context=context_resp,
        total_latency_ms=elapsed_ms,
        guardrail_status=final_state.get("guardrail_status"),
        guardrail_blocked=final_state.get("guardrail_blocked"),
        generation_status=final_state.get("generation_status"),
        retrieval=final_state.get("retrieval"),
        rerank=final_state.get("rerank"),
        context=final_state.get("context"),
        citations=citations,
        citation_validation=final_state.get("citation_validation"),
        structured_answer=final_state.get("structured_answer"),
        usage=final_state.get("usage"),
        tenant_id=identity.tenant_id,
    )


# ----------------------------------------------------------------------
# Runtime / index
# ----------------------------------------------------------------------
@app.get("/runtime/status")
def runtime_status():
    return {
        "readiness": collect_readiness(deep=False),
        "config": {
            "retrieval_mode": settings.RETRIEVAL_MODE,
            "top_k_retrieval": settings.TOP_K_RETRIEVAL,
            "top_k_rerank": settings.TOP_K_RERANK,
            "rerank_policy": settings.RERANK_POLICY,
            "model_init_mode": settings.MODEL_INIT_MODE,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "embedding_model": settings.EMBEDDING_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "llm_provider": settings.LLM_PROVIDER,
            "store_backend": settings.STORE_BACKEND,
        },
        "telemetry": get_exporter().status(),
        "metrics_enabled": settings.METRICS_ENABLED,
    }


@app.post("/index/refresh", response_model=IndexRefreshResponse)
def refresh_index(identity: Identity = Depends(get_identity)):
    require_role(identity, "admin")
    runtime = get_retrieval_runtime()
    try:
        snapshot = runtime.refresh_from_store(get_vector_store())
        metrics.inc("rag_index_refresh_total", {"status": "success"})
        metrics.set_gauge("rag_index_chunks", snapshot.chunk_count)
        metrics.set_gauge("rag_index_version", snapshot.version)
    except RuntimeFailure:
        metrics.inc("rag_index_refresh_total", {"status": "failure"})
        raise
    status = runtime.status()
    return IndexRefreshResponse(status="refreshed", index=status)


# ----------------------------------------------------------------------
# Telemetry
# ----------------------------------------------------------------------
@app.get("/traces")
def get_traces(limit: int = Query(default=50, ge=1, le=500)):
    return get_exporter().get_recent_traces(limit=limit)


@app.get("/traces/{query_id}")
def get_trace_timeline(query_id: str, identity: Identity = Depends(get_identity)):
    require_role(identity, "reader")
    spans = [trace for trace in get_exporter().get_recent_traces(limit=2000) if trace.get("query_id") == query_id]
    spans.sort(key=lambda item: item.get("timestamp") or "")
    return {"query_id": query_id, "spans": spans, "span_count": len(spans)}


# ----------------------------------------------------------------------
# Evaluation artifacts (persisted files only; no invented values)
# ----------------------------------------------------------------------
@app.get("/evals/reports")
def get_eval_reports():
    import glob

    reports = []
    eval_dir = os.path.join(BASE_DIR, "eval")
    baseline_path = os.path.join(eval_dir, "baseline_metrics.json")
    reports_dir = os.path.join(eval_dir, "reports")

    def add_report(data: Dict[str, Any], label: str):
        m = data.get("metrics", {})
        reports.append(
            {
                "run": label,
                "status": data.get("status", "legacy-unverified"),
                "judge_status": data.get("judge_status", "unknown"),
                "config": data.get("config", {}),
                "manifest": data.get("manifest"),
                "timestamp": data.get("timestamp", ""),
                "recall_at_5": m.get("recall_at_5", m.get("recall")),
                "precision_at_5": m.get("precision_at_5", m.get("precision")),
                "mrr": m.get("mrr"),
                "ndcg_at_5": m.get("ndcg_at_5"),
                "map_at_5": m.get("map_at_5"),
                "faithfulness": m.get("mean_faithfulness", m.get("faithfulness")),
                "relevancy": m.get("mean_relevancy", m.get("relevancy")),
                "context_precision": m.get("mean_context_precision"),
            }
        )

    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, "r", encoding="utf-8") as handle:
                add_report(json.load(handle), "Baseline Snapshot")
        except Exception as exc:  # noqa: BLE001
            logger.warning("error reading baseline_metrics.json: %s", exc)

    for idx, filepath in enumerate(sorted(glob.glob(os.path.join(reports_dir, "*.json"))), start=1):
        try:
            with open(filepath, "r", encoding="utf-8") as handle:
                add_report(json.load(handle), f"Run {idx} ({os.path.basename(filepath)})")
        except Exception as exc:  # noqa: BLE001
            logger.warning("error reading report %s: %s", filepath, exc)
    return reports


_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="eval-runner")


def _execute_eval_task(accept_baseline: bool):
    try:
        from src.evaluation.regression_runner import run_evaluation

        dataset_path = os.path.join(BASE_DIR, "eval", "fixtures", "queries.jsonl")
        corpus_dir = os.path.join(BASE_DIR, "eval", "fixtures", "corpus")
        if not (os.path.exists(dataset_path) and os.path.exists(corpus_dir)):
            dataset_path = os.path.join(BASE_DIR, "eval", "golden_dataset.jsonl")
            corpus_dir = None
        results = run_evaluation(dataset_path=dataset_path, corpus_dir=corpus_dir)
        if accept_baseline and results.get("status") == "completed":
            baseline_path = os.path.join(BASE_DIR, "eval", "baseline_metrics.json")
            with open(baseline_path, "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2)
        logger.info("background evaluation completed status=%s", results.get("status"))
    except Exception:  # noqa: BLE001
        logger.exception("background evaluation failed")


@app.post("/evals/run")
def trigger_eval_run(
    accept_baseline: bool = Query(default=False, description="Explicitly accept the new run as baseline"),
    identity: Identity = Depends(get_identity),
):
    require_role(identity, "admin")
    _executor.submit(_execute_eval_task, accept_baseline)
    return {
        "status": "processing",
        "message": "Evaluation launched in a bounded background executor; inspect /evals/reports.",
        "accept_baseline": accept_baseline,
    }
