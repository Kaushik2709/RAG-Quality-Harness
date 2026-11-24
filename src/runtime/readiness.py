"""Readiness that reflects real dependencies instead of a static 200."""
from __future__ import annotations

from typing import Any, Dict

from src.config import settings
from src.observability.metrics import metrics
from src.runtime.lifecycle import get_model_registry, startup_state
from src.runtime.model_registry import ModelState


def _initialized_store():
    """Return the already-created default store without triggering creation."""
    from src.retrieval import vector_store as vs

    if not hasattr(vs._default_vector_store, "cache_info"):
        return None
    try:
        cache = vs._default_vector_store.cache_info()
    except Exception:  # noqa: BLE001
        return None
    if cache.currsize == 0:
        return None
    try:
        return vs._default_vector_store()
    except Exception:  # noqa: BLE001
        return None


def collect_readiness(*, deep: bool = False) -> Dict[str, Any]:
    reasons: list[str] = []
    components: Dict[str, Any] = {}

    # 1. Model lifecycle
    registry = get_model_registry()
    model_status = registry.status()
    components["models"] = model_status
    required_failures = [
        name
        for name, status in model_status["models"].items()
        if status["required"] and status["state"] == ModelState.FAILED.value
    ]
    if required_failures:
        reasons.append(f"required models failed: {','.join(required_failures)}")
    optional_failures = [
        name
        for name, status in model_status["models"].items()
        if not status["required"] and status["state"] == ModelState.FAILED.value
    ]

    # 2. Index and store
    from src.retrieval.retrieval_runtime import get_retrieval_runtime
    from src.retrieval.vector_store import store_readiness

    runtime = get_retrieval_runtime()
    components["index"] = runtime.status()
    store = _initialized_store()
    if store is None:
        if deep:
            try:
                from src.retrieval.vector_store import get_vector_store

                store = get_vector_store()
            except Exception as exc:  # noqa: BLE001
                components["vector_store"] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
                reasons.append("vector store unavailable")
        else:
            components["vector_store"] = {"status": "not_initialized"}
    if store is not None:
        components["vector_store"] = store_readiness(store)

    if deep and store is not None:
        if components["index"].get("status") != "ready":
            try:
                runtime.warm_index(store)
                components["index"] = runtime.status()
                metrics.inc("rag_index_refresh_total", {"status": "success"})
                metrics.set_gauge("rag_index_chunks", components["index"].get("chunk_count", 0))
                metrics.set_gauge("rag_index_version", components["index"].get("version") or 0)
            except Exception as exc:  # noqa: BLE001
                components["index"] = {**runtime.status(), "error": f"{type(exc).__name__}: {exc}"}
                metrics.inc("rag_index_refresh_total", {"status": "failure"})
                reasons.append("retrieval index unavailable")
    elif components["index"].get("status") == "invalidated":
        reasons.append("retrieval index invalidated")

    # 3. Security configuration
    auth_disabled = not (settings.API_KEY or "").strip()
    components["authentication"] = {
        "status": "disabled" if auth_disabled else "enabled",
        "tenant_allowlist": bool(settings.TENANT_IDS),
    }
    if auth_disabled and settings.APP_ENV == "production":
        reasons.append("API_KEY is required when APP_ENV=production")

    # 4. Telemetry exporter
    try:
        from src.tracing.exporter import get_exporter

        components["telemetry"] = get_exporter().status()
    except Exception as exc:  # noqa: BLE001
        components["telemetry"] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    components["startup"] = startup_state()
    if components["startup"].get("error"):
        reasons.append("model warmup failed")

    if reasons:
        status = "not_ready"
    elif optional_failures:
        status = "degraded"
    else:
        status = "ready"
    return {
        "status": status,
        "ready": status in {"ready", "degraded"},
        "degraded": status == "degraded",
        "reasons": reasons,
        "optional_failures": optional_failures,
        "components": components,
    }
