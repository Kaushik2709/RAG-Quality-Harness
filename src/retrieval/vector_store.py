from typing import List, Optional
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore
from langchain_core.documents import Document
from src.ingestion.embedder import get_embeddings_model
from src.config import settings

def get_vector_store(
    embeddings=None,
    use_in_memory: bool = False,
    collection_name: Optional[str] = None
) -> VectorStore:
    """
    Returns a standard LangChain VectorStore object (Qdrant or InMemoryVectorStore).
    """
    emb = embeddings or get_embeddings_model()
    col_name = collection_name or settings.QDRANT_COLLECTION

    if use_in_memory:
        return InMemoryVectorStore(embedding=emb)

    try:
        from qdrant_client import QdrantClient

        url = (settings.QDRANT_URL or "").strip()
        api_key = (settings.QDRANT_API_KEY or "").strip()

        # Detect swapped URL / API key in environment
        if url.startswith("eyJ") and api_key.startswith("http"):
            url, api_key = api_key, url

        if url:
            if not url.startswith("http://") and not url.startswith("https://"):
                url = f"https://{url}"
            client = QdrantClient(
                url=url,
                api_key=api_key or None,
                check_compatibility=False,
                timeout=60.0
            )
        else:
            client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=api_key or None,
                check_compatibility=False,
                timeout=30.0
            )

        # Check/create collection if needed
        try:
            collections = [c.name for c in client.get_collections().collections]
            if col_name not in collections:
                from qdrant_client.models import VectorParams, Distance
                # Default dimension for text-embedding-004 is 768
                sample_vec = emb.embed_query("test")
                dim = len(sample_vec) if sample_vec else 768
                client.create_collection(
                    collection_name=col_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
                )
        except Exception as col_err:
            print(f"[VectorStore] Notice checking/creating collection '{col_name}': {col_err}")

        try:
            from langchain_qdrant import QdrantVectorStore
            return QdrantVectorStore(
                client=client,
                collection_name=col_name,
                embedding=emb
            )
        except Exception as q_err:
            print(f"[VectorStore Error] QdrantVectorStore failed: {q_err}")
            try:
                from langchain_community.vectorstores import Qdrant
                return Qdrant(
                    client=client,
                    collection_name=col_name,
                    embeddings=emb
                )
            except Exception as ex:
                print(f"[VectorStore Warning] Could not wrap QdrantClient into LangChain VectorStore (Primary error: {q_err} | Fallback error: {ex}). Using InMemoryVectorStore.")
                return InMemoryVectorStore(embedding=emb)
    except Exception as e:
        print(f"[VectorStore Warning] Qdrant connection error ({e}). Falling back to LangChain InMemoryVectorStore.")
        return InMemoryVectorStore(embedding=emb)
