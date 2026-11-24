"""Authorization helpers. Tenant boundaries and role/permission checks are enforced prior to resource access."""
from __future__ import annotations

from src.observability.metrics import metrics
from src.runtime.errors import AuthorizationError
from src.security.authentication import Identity


def require_role(identity: Identity, role: str) -> None:
    """Enforce role requirement. Admin role satisfies reader and ingestor checks."""
    if role not in identity.roles and "admin" not in identity.roles:
        metrics.inc("rag_auth_failures_total", {"reason": f"missing_role_{role}"})
        raise AuthorizationError(f"role '{role}' is required", component="authorization")


def require_permission(identity: Identity, permission: str) -> None:
    """Enforce fine-grained permission requirement."""
    if permission not in identity.permissions:
        metrics.inc("rag_auth_failures_total", {"reason": f"missing_permission_{permission}"})
        raise AuthorizationError(f"permission '{permission}' is required", component="authorization")


def require_tenant(identity: Identity, tenant_id: str) -> None:
    """Verify tenant isolation boundary."""
    if identity.tenant_id != tenant_id:
        metrics.inc("rag_cross_tenant_access_total", {"action": "tenant_boundary_check"})
        raise AuthorizationError("cross-tenant access is not permitted", component="authorization")


def tenant_scope(identity: Identity) -> str:
    """Extract and validate tenant scope from identity."""
    if not identity.tenant_id:
        raise AuthorizationError("identity has no tenant scope", component="authorization")
    return identity.tenant_id
