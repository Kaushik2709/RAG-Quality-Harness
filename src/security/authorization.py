"""Authorization helpers. Tenant boundaries are checked before retrieval."""
from __future__ import annotations

from src.runtime.errors import AuthorizationError
from src.security.authentication import Identity


def require_role(identity: Identity, role: str) -> None:
    if role not in identity.roles:
        raise AuthorizationError(f"role '{role}' is required", component="authorization")


def require_tenant(identity: Identity, tenant_id: str) -> None:
    if identity.tenant_id != tenant_id:
        raise AuthorizationError("cross-tenant access is not permitted", component="authorization")


def tenant_scope(identity: Identity) -> str:
    if not identity.tenant_id:
        raise AuthorizationError("identity has no tenant scope", component="authorization")
    return identity.tenant_id
