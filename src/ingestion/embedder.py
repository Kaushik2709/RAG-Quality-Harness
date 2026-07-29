import os
from typing import Optional
from langchain_core.embeddings import Embeddings
from src.config import settings

def get_embeddings_model(model_name: Optional[str] = None, api_key: Optional[str] = None) -> Embeddings:
    """
    Returns a standard production LangChain Embeddings instance (GoogleGenerativeAIEmbeddings or HuggingFaceEmbeddings).
    Raises an exception if embedding model initialization fails.
    """
    m_name = model_name or settings.EMBEDDING_MODEL
    key = api_key or settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")

    if key:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model=m_name,
                google_api_key=key
            )
        except Exception as e:
            raise RuntimeError(f"[Embedder Error] Failed to initialize GoogleGenerativeAIEmbeddings ({m_name}): {e}")
    else:
        # Use real local HuggingFace embeddings model when no Google API key is provided
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        except Exception as e:
            raise ValueError(
                f"[Embedder Error] GEMINI_API_KEY is missing and HuggingFaceEmbeddings could not be loaded: {e}. "
                "Please configure GEMINI_API_KEY in your .env file."
            )

