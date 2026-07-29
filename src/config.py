from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Ingestion & Chunking
    CHUNK_SIZE: int = Field(default=500, description="Target chunk length in characters")
    CHUNK_OVERLAP: int = Field(default=50, description="Overlap between consecutive chunks")
    CHUNKER_STRATEGY: Literal["fixed", "recursive"] = Field(default="recursive", description="Chunking algorithm strategy")

    # Embedder & Reranker Models
    EMBEDDING_MODEL: str = Field(default="models/gemini-embedding-001", description="Google Generative AI embedding model")
    RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-base", description="CrossEncoder reranker model")

    # Retrieval
    TOP_K_RETRIEVAL: int = Field(default=10, description="Number of initial candidates to retrieve")
    TOP_K_RERANK: int = Field(default=3, description="Number of top candidates after cross-encoder reranking")
    USE_HYBRID_SEARCH: bool = Field(default=True, description="Enable dense + BM25 hybrid retrieval")

    # Vector Store & Postgres
    QDRANT_HOST: str = Field(default="localhost", description="Qdrant host address")
    QDRANT_PORT: int = Field(default=6333, description="Qdrant HTTP port")
    QDRANT_URL: str = Field(default="", description="Qdrant Cloud endpoint URL")
    QDRANT_API_KEY: str = Field(default="", description="Qdrant API Key")
    QDRANT_COLLECTION: str = Field(default="rag_documents", description="Qdrant collection name")
    POSTGRES_URI: str = Field(default="postgresql://rag_user:rag_password@localhost:5432/rag_traces", description="Postgres connection URI")

    # LLM Provider & Generation
    LLM_PROVIDER: Literal["gemini", "claude"] = Field(default="gemini", description="LLM provider for response generation")
    GEMINI_API_KEY: str = Field(default="", description="Gemini API Key")
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash", description="Gemini model name")
    CLAUDE_API_KEY: str = Field(default="", description="Claude API Key")
    CLAUDE_MODEL: str = Field(default="claude-3-5-sonnet-20240620", description="Claude model name")

    # Regression & Evaluation Thresholds
    MIN_RECALL_THRESHOLD: float = Field(default=0.75, description="Minimum acceptable Recall@K ratio")
    MIN_FAITHFULNESS_THRESHOLD: float = Field(default=0.80, description="Minimum acceptable LLM-as-Judge Faithfulness score")
    MAX_RECALL_DROP_TOLERANCE: float = Field(default=0.05, description="Maximum allowable recall drop vs baseline")

settings = Settings()
