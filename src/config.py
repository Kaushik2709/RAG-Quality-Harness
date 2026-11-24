from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    APP_ENV: Literal["development", "test", "staging", "production"] = Field(default="development")
    SERVICE_VERSION: str = Field(default="3.0.0")

    # Ingestion & Chunking
    CHUNK_SIZE: int = Field(default=500, description="Target chunk length in characters")
    CHUNK_OVERLAP: int = Field(default=50, description="Overlap between consecutive chunks")
    CHUNKER_STRATEGY: Literal["fixed", "recursive"] = Field(default="recursive", description="Chunking algorithm strategy")
    MAX_DOCUMENT_CHARS: int = Field(default=2_000_000, description="Reject documents larger than this")
    MAX_CHUNKS_PER_DOCUMENT: int = Field(default=20_000, description="Hard cap on chunks per document version")
    MAX_PDF_PAGES: int = Field(default=500, description="Hard cap on pages parsed from a PDF")
    INGESTION_REGISTRY_PATH: str = Field(default="data/ingestion/registry.json", description="Persistent document/version registry")

    # Embedder & Reranker Models
    EMBEDDING_MODEL: str = Field(default="models/gemini-embedding-001", description="Google Generative AI embedding model")
    RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-base", description="CrossEncoder reranker model")

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    MODEL_INIT_MODE: Literal["strict", "degraded"] = Field(
        default="degraded",
        description="strict: fail requests when a required model is unavailable; degraded: skip only explicitly optional stages and report status",
    )
    RERANK_POLICY: Literal["strict", "degraded", "disabled"] = Field(
        default="degraded",
        description="How to behave when the cross-encoder is unavailable. Degraded skips reranking and reports it; it never claims reranking succeeded.",
    )
    MODEL_INIT_MAX_ATTEMPTS: int = Field(default=2, ge=1, le=10)
    MODEL_INIT_BACKOFF_SECONDS: float = Field(default=0.25, ge=0)
    MODEL_INIT_TIMEOUT_SECONDS: float = Field(default=120.0, gt=0, description="Bounded initialization timeout per model")
    MODEL_WARMUP_ENABLED: bool = Field(default=False, description="Warm up embeddings/reranker during startup")
    WARMUP_LLM: bool = Field(default=False, description="Issue a paid generation call during warmup")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    TOP_K_RETRIEVAL: int = Field(default=10, description="Number of initial candidates to retrieve")
    TOP_K_RERANK: int = Field(default=3, description="Number of top candidates after cross-encoder reranking")
    USE_HYBRID_SEARCH: bool = Field(default=True, description="Enable dense + BM25 hybrid retrieval")
    RETRIEVAL_MODE: Literal["dense", "bm25", "hybrid"] = Field(default="hybrid")
    FUSION_DENSE_WEIGHT: float = Field(default=0.6, ge=0)
    FUSION_SPARSE_WEIGHT: float = Field(default=0.4, ge=0)
    FUSION_RRF_K: int = Field(default=60, gt=0)
    MAX_INDEX_CHUNKS: int = Field(default=250_000, description="Refuse to build an index larger than this in-process")

    # Vector Store & Postgres
    STORE_BACKEND: Literal["auto", "memory", "qdrant"] = Field(
        default="auto",
        description="auto tries Qdrant; memory is the explicit, dev-only in-process backend",
    )
    ALLOW_IN_MEMORY_FALLBACK: bool = Field(
        default=False,
        description="Explicitly allow Qdrant failure to fall back to an in-memory store. Off by default so degraded retrieval is never silent.",
    )
    VECTOR_STORE_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0)
    QDRANT_HOST: str = Field(default="localhost", description="Qdrant host address")
    QDRANT_PORT: int = Field(default=6333, description="Qdrant HTTP port")
    QDRANT_GRPC_PORT: int = Field(default=6334, description="Qdrant gRPC port")
    QDRANT_PREFER_GRPC: bool = Field(default=True, description="Prefer gRPC over HTTP for Qdrant client")
    QDRANT_URL: str = Field(default="", description="Qdrant Cloud endpoint URL")
    QDRANT_API_KEY: str = Field(default="", description="Qdrant API Key")
    QDRANT_COLLECTION: str = Field(default="rag_documents", description="Qdrant collection name")
    POSTGRES_URI: str = Field(default="postgresql://rag_user:rag_password@localhost:5432/rag_traces", description="Postgres connection URI")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    LLM_PROVIDER: Literal["gemini", "claude"] = Field(default="gemini", description="LLM provider for response generation")
    GEMINI_API_KEY: str = Field(default="", description="Gemini API Key")
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash", description="Gemini model name")
    CLAUDE_API_KEY: str = Field(default="", description="Claude API Key")
    CLAUDE_MODEL: str = Field(default="claude-3-5-sonnet-20240620", description="Claude model name")
    LLM_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0)
    LLM_MAX_RETRIES: int = Field(default=0, ge=0, le=5, description="Implicit provider retries; kept at 0 to avoid duplicate spend")
    GENERATION_MAX_TOKENS: int = Field(default=1024, gt=0)
    MAX_CONTEXT_TOKENS: int = Field(default=3000, gt=0)
    CONTEXT_MAX_CHUNKS: int = Field(default=12, gt=0)
    CONTEXT_DIVERSIFY_SOURCES: bool = Field(default=True)
    MIN_CITATION_QUOTE_CHARS: int = Field(default=12, ge=0)
    CITATION_VALIDATION_POLICY: Literal["strict", "review", "disabled"] = Field(
        default="strict",
        description="strict: reject complete answer if any required citation fails; review: return marked partial result; disabled: allow without verification",
    )

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------
    ENABLE_INPUT_GUARDRAILS: bool = Field(default=True, description="Enable regex-based input guardrails & prompt injection detection")
    ENABLE_OUTPUT_GUARDRAILS: bool = Field(default=True, description="Enable output guardrails & secret/PII redaction")
    GUARDRAIL_BLOCK_PROMPT_INJECTION: bool = Field(default=True, description="Block execution if prompt injection is detected")
    GUARDRAIL_STRICT_MODE: bool = Field(default=False, description="Reject queries with any PII or sensitive data")
    RETRIEVED_DOCUMENT_SECURITY_POLICY: Literal["allow", "mark_untrusted", "exclude", "quarantine"] = Field(
        default="mark_untrusted",
        description="allow: include evidence; mark_untrusted: isolate and wrap in untrusted data delimiters; exclude: drop high-risk evidence; quarantine: store in quarantine registry and drop",
    )
    API_KEY: str = Field(default="", description="Static API key. Empty disables authentication (development only).")
    DEV_AUTH_ENABLED: bool = Field(default=False, description="Explicitly enable unauthenticated development access")
    DEV_TENANT_ID: str = Field(default="development", description="Fixed tenant used by the development bypass")
    API_KEY_HEADER: str = Field(default="X-API-Key")
    TENANT_IDS: str = Field(default="", description="Comma-separated tenant allowlist; empty allows any X-Tenant-ID")
    CORS_ALLOW_ORIGINS: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated CORS allowlist. Wildcards with credentials are not permitted.",
    )
    RATE_LIMIT_PER_MINUTE: int = Field(default=120, ge=0, description="0 disables the in-process limiter")
    RATE_LIMIT_BURST: int = Field(default=20, ge=0)
    MAX_UPLOAD_BYTES: int = Field(default=10 * 1024 * 1024, gt=0)
    ALLOWED_UPLOAD_EXTENSIONS: str = Field(default=".txt,.md,.markdown,.html,.htm,.pdf")
    UPLOAD_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    TRACE_SINK: Literal["sqlite", "postgres", "none"] = Field(default="sqlite")
    TRACE_DB_PATH: str = Field(default="traces_local.db")
    TRACE_QUEUE_SIZE: int = Field(default=10_000, gt=0)
    TRACE_BATCH_SIZE: int = Field(default=100, gt=0)
    TRACE_FLUSH_INTERVAL_SECONDS: float = Field(default=1.0, gt=0)
    TRACE_RETENTION_DAYS: int = Field(default=30, ge=0)
    METRICS_ENABLED: bool = Field(default=True)

    # ------------------------------------------------------------------
    # Regression & Evaluation Thresholds
    # ------------------------------------------------------------------
    MIN_RECALL_THRESHOLD: float = Field(default=0.75, description="Minimum acceptable Recall@K ratio")
    MIN_FAITHFULNESS_THRESHOLD: float = Field(default=0.80, description="Minimum acceptable LLM-as-Judge Faithfulness score")
    MAX_RECALL_DROP_TOLERANCE: float = Field(default=0.05, description="Maximum allowable recall drop vs baseline")
    EVAL_BOOTSTRAP_SAMPLES: int = Field(default=2000, ge=100)
    EVAL_BOOTSTRAP_SEED: int = Field(default=12345)


settings = Settings()
