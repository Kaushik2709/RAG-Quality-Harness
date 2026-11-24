"""API authentication and tenant identity."""
from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Optional

from src.config import settings
from src.runtime.errors import AuthenticationError


@dataclass(frozen=True)
class Identity:
    subject: str
    tenant_id: str
    roles: tuple[str, ...] = ("reader",)
    auth_method: str = "disabled"
    authenticated: bool = False

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "tenant_id": self.tenant_id,
            "roles": list(self.roles),
            "auth_method": self.auth_method,
        }


def _tenant_allowlist() -> list[str]:
    return [t.strip() for t in settings.TENANT_IDS.split(",") if t.strip()]


def _resolve_tenant(requested: Optional[str]) -> str:
    tenant = (requested or "default").strip() or "default"
    allowed = _tenant_allowlist()
    if allowed and tenant not in allowed:
        raise AuthenticationError("tenant is not in the configured allowlist", component="authorization")
    return tenant


def authenticate(api_key: Optional[str], tenant_id: Optional[str] = None) -> Identity:
    configured = (settings.API_KEY or "").strip()
    if configured:
        supplied = (api_key or "").strip()
        # Constant-time comparison even for length-mismatched values.
        valid = hmac.compare_digest(supplied.encode("utf-8"), configured.encode("utf-8")) if supplied else False
        if not valid:
            raise AuthenticationError("missing or invalid API key", component="authentication")
        tenant = _resolve_tenant(tenant_id)
        return Identity(
            subject=f"api-key:{configured[:4]}…",
            tenant_id=tenant,
            roles=("reader", "ingestor", "admin"),
            auth_method="api_key",
            authenticated=True,
        )

    # Development mode: authentication is disabled by configuration, and this
    # fact is surfaced by /ready so it cannot be mistaken for a secured deploy.
    return Identity(
        subject="anonymous",
        tenant_id=_resolve_tenant(tenant_id),
        roles=("reader", "ingestor", "admin"),
        auth_method="disabled",
        authenticated=False,
    )
