"""Application lifecycle wiring: models, index, exporter, readiness."""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from src.config import settings
from src.runtime.model_registry import ModelRegistry, build_default_registry

_registry: Optional[ModelRegistry] = None
_lock = threading.Lock()
_startup_state: Dict[str, Any] = {"started": False, "warmup": None, "error": None}


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        with _lock:
            if _registry is None:
                _registry = build_default_registry()
    return _registry


def reset_model_registry(registry: Optional[ModelRegistry] = None) -> None:
    global _registry, _startup_state
    with _lock:
        if _registry is not None and registry is None:
            _registry.close()
        _registry = registry
        _startup_state = {"started": False, "warmup": None, "error": None}


def startup(warmup: Optional[bool] = None) -> Dict[str, Any]:
    """Prepare process-owned resources. Cold start never blocks HTTP startup on
    paid calls: warmup is opt-in via MODEL_WARMUP_ENABLED."""
    global _startup_state
    registry = get_model_registry()
    do_warmup = settings.MODEL_WARMUP_ENABLED if warmup is None else warmup
    warmup_result = None
    error = None
    if do_warmup:
        try:
            warmup_result = registry.warmup(llm=settings.WARMUP_LLM)
        except Exception as exc:  # noqa: BLE001 - surfaced by readiness
            error = f"{type(exc).__name__}: {exc}"
    _startup_state = {"started": True, "warmup": warmup_result, "error": error}
    return dict(_startup_state)


def shutdown() -> None:
    global _registry
    with _lock:
        if _registry is not None:
            _registry.close()
        _registry = None
    try:
        from src.tracing.exporter import get_exporter

        get_exporter().close()
    except Exception:  # noqa: BLE001 - shutdown is best effort but never silently claims success
        pass


def startup_state() -> Dict[str, Any]:
    return dict(_startup_state)
