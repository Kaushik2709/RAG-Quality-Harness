"""FastAPI dependency wiring for identity, tenant scope and rate limiting."""
from __future__ import annotations

from typing import Optional

from fastapi import Header

from src.config import settings
from src.security.authentication import Identity, authenticate
from src.security.ratelimit import TokenBucketLimiter

_limiter: Optional[TokenBucketLimiter] = None
_limiter_config: tuple[int, int] | None = None


def get_identity(
    api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
) -> Identity:
    return authenticate(api_key, tenant_id)


def get_limiter() -> TokenBucketLimiter:
    global _limiter, _limiter_config
    config = (settings.RATE_LIMIT_PER_MINUTE, settings.RATE_LIMIT_BURST)
    if _limiter is None or _limiter_config != config:
        _limiter = TokenBucketLimiter(per_minute=config[0], burst=config[1])
        _limiter_config = config
    return _limiter


def reset_limiter() -> None:
    global _limiter, _limiter_config
    if _limiter is not None:
        _limiter.reset()
    _limiter = None
    _limiter_config = None
