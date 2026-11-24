"""Public API schemas. Kept separate from route wiring so contract changes are
reviewable and testable without importing FastAPI."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2_000_000)
    doc_id: Optional[str] = Field(default=None, max_length=200)
    source: Optional[str] = Field(default="api_text", max_length=300)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_size: Optional[int] = Field(default=None, ge=1, le=100_000)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=50_000)
    strategy: Optional[str] = None

    @field_validator("strategy")
    def known_strategy(cls, value):
        if value is not None and value not in {"fixed", "recursive"}:
            raise ValueError("strategy must be 'fixed' or 'recursive'")
        return value

    @field_validator("source")
    def source_not_blank(cls, value):
        if value is not None and not value.strip():
            raise ValueError("source cannot be blank")
        return value


class IngestResponse(BaseModel):
    doc_id: str
    chunks_created: int
    status: str = "success"
    version: Optional[int] = None
    content_sha256: Optional[str] = None
    duplicate: bool = False
    activated: bool = False
    index_version: Optional[int] = None
    tenant_id: str = "default"
    warnings: List[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=20_000)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    top_k_rerank: Optional[int] = Field(default=None, ge=1, le=100)
    use_hybrid: Optional[bool] = None
    retrieval_mode: Optional[str] = None

    @field_validator("retrieval_mode")
    def known_mode(cls, value):
        if value is not None and value not in {"dense", "bm25", "hybrid"}:
            raise ValueError("retrieval_mode must be dense, bm25 or hybrid")
        return value


class ContextChunkResponse(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: Optional[float] = None
    document_version: Optional[int] = None
    section_path: Optional[str] = None
    page: Optional[int] = None


class CitationResponse(BaseModel):
    chunk_id: str
    quote: str


class QueryResponse(BaseModel):
    query_id: str
    query: str
    answer: str
    retrieved_context: List[ContextChunkResponse]
    total_latency_ms: float
    guardrail_status: Optional[Dict[str, Any]] = None
    guardrail_blocked: Optional[bool] = None
    generation_status: Optional[str] = None
    retrieval: Optional[Dict[str, Any]] = None
    rerank: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    citations: List[CitationResponse] = Field(default_factory=list)
    citation_validation: Optional[Dict[str, Any]] = None
    structured_answer: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None
    tenant_id: str = "default"


class DocumentSummary(BaseModel):
    doc_id: str
    tenant_id: str
    source: str
    active_version: Optional[int] = None
    status: str
    chunk_count: int
    content_sha256: Optional[str] = None
    versions: List[Dict[str, Any]] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    doc_id: str
    tenant_id: str
    chunks_deleted: int
    status: str
    index_refreshed: bool
    warnings: List[str] = Field(default_factory=list)


class IndexRefreshResponse(BaseModel):
    status: str
    index: Dict[str, Any]


class ErrorBody(BaseModel):
    code: str
    message: str
    component: Optional[str] = None
    retryable: bool = False
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorBody
