"""Vector-store construction with explicit backend selection.

The previous implementation printed a warning and silently replaced a failed
Qdrant connection with an in-memory store. That turns an infrastructure failure
into an apparently healthy service with an empty corpus. The default is now to
raise `DependencyUnavailable`; an in-memory fallback exists only when the
operator explicitly asks for it (`STORE_BACKEND=memory` or
`ALLOW_IN_MEMORY_FALLBACK=true`).
"""
from __future__ import annotations

import math
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from src.config import settings
from src.ingestion.embedder import get_embeddings_model
from src.runtime.errors import DependencyUnavailable

_MEMORY_BACKEND = "memory"
_QDRANT_BACKEND = "qdrant"


def _create_qdrant_client():
    from qdrant_client import QdrantClient

    url = (settings.QDRANT_URL or "").strip()
    api_key = (settings.QDRANT_API_KEY or "").strip()

    # Detect swapped URL / API key in environment (observed real-world misconfiguration).
    if url.startswith("eyJ") and api_key.startswith("http"):
        url, api_key = api_key, url

    if url:
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"
        return QdrantClient(
            url=url,
            api_key=api_key or None,
            check_compatibility=False,
            timeout=settings.VECTOR_STORE_TIMEOUT_SECONDS,
        )
    return QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        grpc_port=settings.QDRANT_GRPC_PORT,
        prefer_grpc=settings.QDRANT_PREFER_GRPC,
        api_key=api_key or None,
        check_compatibility=False,
        timeout=settings.VECTOR_STORE_TIMEOUT_SECONDS,
    )


def _ensure_collection(client, collection_name: str, embeddings) -> None:
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        from qdrant_client.models import Distance, ScalarQuantization, ScalarQuantizationConfig, ScalarType, VectorParams

        sample = embeddings.embed_query("dimension probe")
        if not sample:
            raise DependencyUnavailable("embedding model returned an empty vector", component="embedding")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(sample), distance=Distance.COSINE),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
            ),
        )
    try:
        from qdrant_client.models import PayloadSchemaType

        for field in (
            "tenant_id",
            "version_state",
            "doc_id",
            "document_id",
            "metadata.tenant_id",
            "metadata.version_state",
            "metadata.doc_id",
            "metadata.document_id",
        ):
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass
    except Exception:
        pass


def _create_vector_store(
    embeddings=None,
    use_in_memory: bool = False,
    collection_name: Optional[str] = None,
) -> VectorStore:
    """Create a vector store, failing explicitly when the backend is unavailable."""
    emb = embeddings or get_embeddings_model()
    col_name = collection_name or settings.QDRANT_COLLECTION
    backend = "memory" if use_in_memory else settings.STORE_BACKEND

    if backend == "memory":
        return InMemoryVectorStore(embedding=emb)

    if backend not in {"auto", "qdrant"}:
        raise ValueError(f"Unknown STORE_BACKEND: {backend}")

    try:
        client = _create_qdrant_client()
        _ensure_collection(client, col_name, emb)
        try:
            from langchain_qdrant import QdrantVectorStore

            store = QdrantVectorStore(client=client, collection_name=col_name, embedding=emb)
        except ImportError:
            from langchain_community.vectorstores import Qdrant

            store = Qdrant(client=client, collection_name=col_name, embeddings=emb)
        try:
            setattr(store, "_rag_backend", _QDRANT_BACKEND)
        except Exception:
            pass
        return store
    except Exception as exc:  # noqa: BLE001 - converted to an explicit contract error
        if settings.ALLOW_IN_MEMORY_FALLBACK:
            return InMemoryVectorStore(embedding=emb)
        raise DependencyUnavailable(
            f"vector store backend '{backend}' is unavailable: {type(exc).__name__}: {exc}",
            component="vector_store",
            retryable=True,
        ) from exc


from functools import lru_cache


@lru_cache(maxsize=1)
def _default_vector_store():
    """One process-local default store, shared by ingestion and queries."""
    return _create_vector_store()


def get_vector_store(embeddings=None, use_in_memory=False, collection_name=None):
    if embeddings is None and not use_in_memory and collection_name is None:
        # The application store is cached. Creation still raises explicitly on failure.
        return _default_vector_store()
    return _create_vector_store(embeddings, use_in_memory, collection_name)


def reset_default_vector_store() -> None:
    # A test may replace the cached function with a plain callable.
    cache_clear = getattr(_default_vector_store, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()


def get_store_backend(store: Any) -> str:
    if isinstance(store, InMemoryVectorStore):
        return _MEMORY_BACKEND
    return getattr(store, "_rag_backend", _QDRANT_BACKEND)


def get_indexed_documents(store) -> List[Document]:
    """Read the complete collection. Only index builds may call this."""
    if isinstance(store, InMemoryVectorStore):
        return [Document(page_content=row["text"], metadata=dict(row["metadata"])) for row in store.store.values()]
    documents: List[Document] = []
    offset = None
    while True:
        points, offset = store.client.scroll(
            collection_name=store.collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            meta = dict(payload.get("metadata", {}))
            if "version_state" in payload and "version_state" not in meta:
                meta["version_state"] = payload["version_state"]
            documents.append(
                Document(page_content=payload.get("page_content", ""), metadata=meta)
            )
        if offset is None:
            return documents


def get_retrieval_documents(store) -> List[Document]:
    """Documents eligible for retrieval.

    Retired staging versions remain in the physical collection until their
    deletion succeeds. They must never enter a snapshot: including them would
    make index chunk counts disagree with the searchable corpus and, if a later
    delete failed, keep stale text in memory. Legacy documents without
    `version_state` are treated as active for backward compatibility.
    """
    return [
        doc
        for doc in get_indexed_documents(store)
        if doc.metadata.get("version_state", "active") == "active"
    ]


def count_documents(store) -> int:
    if isinstance(store, InMemoryVectorStore):
        return len(store.store)
    try:
        return int(store.client.count(store.collection_name, exact=True).count)
    except Exception:  # noqa: BLE001 - diagnostics must not invent a count
        return len(get_indexed_documents(store))


def _validate_vectors(vectors: Sequence[Sequence[float]], expected: int) -> None:
    if len(vectors) != expected:
        raise ValueError(f"embedding count {len(vectors)} does not match chunk count {expected}")
    if not vectors:
        raise ValueError("no embeddings were produced")
    dims = {len(v) for v in vectors}
    if len(dims) != 1 or 0 in dims:
        raise ValueError("embedding vectors have inconsistent or empty dimensions")
    for vector in vectors:
        if any(not isinstance(x, (int, float)) or not math.isfinite(float(x)) for x in vector):
            raise ValueError("embedding vector contains non-finite values")


def to_qdrant_point_id(chunk_id: str) -> str:
    """Qdrant requires point IDs to be unsigned integers or valid UUID strings.
    Deterministically map arbitrary chunk identifiers to stable UUIDv5."""
    try:
        uuid.UUID(str(chunk_id))
        return str(chunk_id)
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"urn:rag:chunk:{chunk_id}"))


def _format_point_ids(store, ids: Sequence[str]) -> List[str]:
    if isinstance(store, InMemoryVectorStore):
        return list(ids)
    return [to_qdrant_point_id(cid) for cid in ids]


def add_with_embeddings(store, documents: Sequence[Document], ids: Sequence[str], embeddings) -> None:
    """Embed once, validate, then write. Avoids double provider billing while
    still validating vector count/dimension/finiteness before activation."""
    texts = [doc.page_content for doc in documents]
    vectors = embeddings.embed_documents(texts)
    _validate_vectors(vectors, len(documents))

    if isinstance(store, InMemoryVectorStore):
        for doc, doc_id, vector in zip(documents, ids, vectors):
            store.store[doc_id] = {
                "id": doc_id,
                "vector": list(vector),
                "text": doc.page_content,
                "metadata": dict(doc.metadata),
            }
        return

    point_ids = _format_point_ids(store, ids)
    add_embeddings = getattr(store, "add_embeddings", None)
    if callable(add_embeddings):
        add_embeddings(text_embeddings=list(zip(texts, vectors)), metadatas=[d.metadata for d in documents], ids=point_ids)
        return
    # Last resort: the backend has no embedding-level write path. This embeds
    # again, which is recorded as a limitation rather than hidden.
    store.add_documents(list(documents), ids=point_ids)


def delete_chunks(store, ids: Sequence[str]) -> int:
    ids = [chunk_id for chunk_id in ids if chunk_id]
    if not ids:
        return 0
    if isinstance(store, InMemoryVectorStore):
        removed = 0
        for chunk_id in ids:
            if store.store.pop(chunk_id, None) is not None:
                removed += 1
        return removed
    point_ids = _format_point_ids(store, ids)
    store.delete(ids=point_ids)
    return len(ids)


def set_chunk_state(store, ids: Sequence[str], state: str) -> int:
    """Change the lifecycle state of chunks without re-embedding them."""
    ids = [chunk_id for chunk_id in ids if chunk_id]
    if not ids:
        return 0
    if isinstance(store, InMemoryVectorStore):
        changed = 0
        for chunk_id in ids:
            row = store.store.get(chunk_id)
            if row is not None:
                row["metadata"]["version_state"] = state
                changed += 1
        return changed
    point_ids = _format_point_ids(store, ids)
    try:
        points = store.client.retrieve(collection_name=store.collection_name, ids=point_ids, with_payload=True)
        for pt in points:
            payload = dict(pt.payload or {})
            meta = dict(payload.get("metadata", {}))
            meta["version_state"] = state
            payload["metadata"] = meta
            payload["version_state"] = state
            store.client.overwrite_payload(collection_name=store.collection_name, payload=payload, points=[pt.id])
    except Exception:
        set_payload = getattr(store.client, "set_payload", None)
        if not callable(set_payload):
            raise DependencyUnavailable("vector store cannot update payloads", component="vector_store")
        set_payload(collection_name=store.collection_name, payload={"version_state": state}, points=point_ids)
    return len(ids)


def chunks_present(store, ids: Sequence[str]) -> bool:
    expected = [chunk_id for chunk_id in ids if chunk_id]
    if not expected:
        return False
    if isinstance(store, InMemoryVectorStore):
        return all(chunk_id in store.store for chunk_id in expected)
    try:
        point_ids = _format_point_ids(store, expected)
        points = store.client.retrieve(collection_name=store.collection_name, ids=point_ids, with_payload=False)
        return len(points) == len(expected)
    except Exception:  # noqa: BLE001
        return False


def delete_by_metadata(store, filters: Dict[str, Any]) -> int:
    """Delete by exact metadata equality. Used for document-version retirement."""
    documents = get_indexed_documents(store)
    ids = [
        str(doc.metadata.get("chunk_id"))
        for doc in documents
        if all(doc.metadata.get(key) == value for key, value in filters.items())
    ]
    return delete_chunks(store, ids)


def _build_filter(filters: Dict[str, Any]):
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    conditions = []
    for key, value in filters.items():
        k = f"metadata.{key}" if not key.startswith("metadata.") else key
        conditions.append(FieldCondition(key=k, match=MatchValue(value=value)))
    return Filter(must=conditions)


def search_dense(
    store,
    query: str,
    k: int,
    *,
    metadata_filters: Optional[Dict[str, Any]] = None,
    predicate: Optional[Callable[[Document], bool]] = None,
) -> List[tuple[Document, float]]:
    """Dense search with pre-filtering on metadata.

    In-memory stores receive a callable filter (evaluated before ranking by
    langchain-core); Qdrant receives a native payload filter. Tenant isolation
    is therefore enforced by the store query, not by post-filtering a response.
    """
    if k <= 0:
        return []
    metadata_filters = metadata_filters or {}
    if isinstance(store, InMemoryVectorStore):
        def combined(doc: Document) -> bool:
            if any(doc.metadata.get(key) != value for key, value in metadata_filters.items()):
                return False
            return predicate(doc) if predicate else True

        return list(store.similarity_search_with_score(query, k=k, filter=combined))
    qdrant_filter = _build_filter(metadata_filters) if metadata_filters else None
    return list(store.similarity_search_with_score(query, k=k, filter=qdrant_filter))


def store_readiness(store) -> Dict[str, Any]:
    backend = get_store_backend(store)
    if backend == _MEMORY_BACKEND:
        return {"status": "ready", "backend": backend, "documents": len(store.store), "persistent": False}
    try:
        collections = [c.name for c in store.client.get_collections().collections]
        return {
            "status": "ready" if store.collection_name in collections else "collection_missing",
            "backend": backend,
            "collection": store.collection_name,
            "documents": count_documents(store),
            "persistent": True,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "backend": backend, "error": f"{type(exc).__name__}: {exc}"}
