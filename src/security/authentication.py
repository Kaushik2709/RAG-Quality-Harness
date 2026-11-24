"""API authentication and tenant-scoped credential registry.

Implements:
1. Cryptographically secure API key generation.
2. Hashing before storage (SHA-256) - plaintext keys are never stored.
3. Constant-time verification (hmac.compare_digest).
4. Credential expiration and revocation.
5. Credential rotation.
6. Tenant-scoped credential ownership.
7. Role and permission assignment (Reader, Ingestor, Admin).
8. Administrative credential provisioning.
9. Authentication audit events.
10. Fail-closed production mode.
"""
from __future__ import annotations

import datetime
import hashlib
import hmac
import os
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config import settings
from src.observability.metrics import metrics
from src.runtime.errors import AuthenticationError, AuthorizationError

ROLE_PERMISSIONS: Dict[str, Tuple[str, ...]] = {
    "reader": ("query", "documents:read", "traces:read"),
    "ingestor": ("query", "documents:read", "traces:read", "documents:write"),
    "admin": (
        "query",
        "documents:read",
        "traces:read",
        "documents:write",
        "documents:delete",
        "index:refresh",
        "credentials:manage",
        "system:read",
        "traces:manage",
        "evals:run",
    ),
}


def hash_api_key(api_key: str) -> str:
    """Hash raw API key with SHA-256 for persistent storage and lookup."""
    clean = (api_key or "").strip()
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


def generate_api_key(prefix: str = "rag_live_") -> str:
    """Generate a cryptographically secure random API key."""
    return f"{prefix}{secrets.token_urlsafe(32)}"


@dataclass
class CredentialRecord:
    credential_id: str
    subject_id: str
    tenant_id: str
    key_hash: str
    key_prefix: str
    roles: Tuple[str, ...]
    permissions: Tuple[str, ...]
    created_at: str
    expires_at: Optional[str] = None
    revoked: bool = False
    revoked_at: Optional[str] = None
    rotated_from: Optional[str] = None
    description: Optional[str] = None

    def is_active(self) -> bool:
        if self.revoked:
            return False
        if self.expires_at:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            if now > self.expires_at:
                return False
        return True

    def status(self) -> str:
        if self.revoked:
            return "revoked"
        if self.expires_at:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            if now > self.expires_at:
                return "expired"
        return "active"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "credential_id": self.credential_id,
            "subject_id": self.subject_id,
            "tenant_id": self.tenant_id,
            "key_prefix": self.key_prefix,
            "roles": list(self.roles),
            "permissions": list(self.permissions),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "revoked": self.revoked,
            "revoked_at": self.revoked_at,
            "rotated_from": self.rotated_from,
            "description": self.description,
            "status": self.status(),
        }


@dataclass(frozen=True)
class Identity:
    subject_id: str = "anonymous"
    credential_id: str = "none"
    tenant_id: str = "default"
    roles: Tuple[str, ...] = ("reader",)
    permissions: Tuple[str, ...] = ("query", "documents:read")
    authentication_method: str = "disabled"
    credential_status: str = "active"
    authenticated: bool = False

    def __init__(
        self,
        subject_id: Optional[str] = None,
        credential_id: str = "none",
        tenant_id: str = "default",
        roles: Any = ("reader",),
        permissions: Any = None,
        authentication_method: Optional[str] = None,
        credential_status: str = "active",
        authenticated: bool = False,
        # Backward-compat params
        subject: Optional[str] = None,
        auth_method: Optional[str] = None,
    ):
        resolved_subj = subject_id or subject or "anonymous"
        resolved_auth = authentication_method or auth_method or ("api_key" if authenticated else "disabled")
        if isinstance(roles, (set, frozenset, list)):
            roles_tuple = tuple(sorted(roles))
        elif isinstance(roles, tuple):
            roles_tuple = roles
        else:
            roles_tuple = (str(roles),)

        if permissions is None:
            perms_set = set()
            for r in roles_tuple:
                perms_set.update(ROLE_PERMISSIONS.get(r, ()))
            perms_tuple = tuple(sorted(perms_set))
        elif isinstance(permissions, (set, frozenset, list)):
            perms_tuple = tuple(sorted(permissions))
        elif isinstance(permissions, tuple):
            perms_tuple = permissions
        else:
            perms_tuple = (str(permissions),)

        object.__setattr__(self, "subject_id", resolved_subj)
        object.__setattr__(self, "credential_id", credential_id)
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "roles", roles_tuple)
        object.__setattr__(self, "permissions", perms_tuple)
        object.__setattr__(self, "authentication_method", resolved_auth)
        object.__setattr__(self, "credential_status", credential_status)
        object.__setattr__(self, "authenticated", authenticated)

    @property
    def subject(self) -> str:
        return self.subject_id

    @property
    def auth_method(self) -> str:
        return self.authentication_method

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "subject": self.subject_id,
            "credential_id": self.credential_id,
            "tenant_id": self.tenant_id,
            "roles": list(self.roles),
            "permissions": list(self.permissions),
            "authentication_method": self.authentication_method,
            "auth_method": self.authentication_method,
            "credential_status": self.credential_status,
            "authenticated": self.authenticated,
        }


@dataclass
class AuthAuditEvent:
    event_type: str
    timestamp: str
    subject_id: Optional[str] = None
    tenant_id: Optional[str] = None
    credential_id: Optional[str] = None
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "subject_id": self.subject_id,
            "tenant_id": self.tenant_id,
            "credential_id": self.credential_id,
            "details": self.details,
        }


class CredentialRegistry:
    """Thread-safe credential registry storing only hashed API keys."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._credentials: Dict[str, CredentialRecord] = {}  # id -> record
        self._hash_to_id: Dict[str, str] = {}  # key_hash -> id
        self._audit_events: List[AuthAuditEvent] = []
        self._max_audit_events = 2000

    def _record_audit(
        self,
        event_type: str,
        subject_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        credential_id: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        event = AuthAuditEvent(
            event_type=event_type,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            subject_id=subject_id,
            tenant_id=tenant_id,
            credential_id=credential_id,
            details=details,
        )
        with self._lock:
            self._audit_events.append(event)
            if len(self._audit_events) > self._max_audit_events:
                self._audit_events = self._audit_events[-self._max_audit_events :]

    def provision(
        self,
        tenant_id: str,
        subject_id: str,
        roles: Sequence[str] = ("reader",),
        permissions: Optional[Sequence[str]] = None,
        expires_in_days: Optional[int] = None,
        description: Optional[str] = None,
        rotated_from: Optional[str] = None,
    ) -> Tuple[str, CredentialRecord]:
        raw_key = generate_api_key()
        k_hash = hash_api_key(raw_key)
        prefix = raw_key[:12] + "..."
        now = datetime.datetime.now(datetime.timezone.utc)
        created_at = now.isoformat()
        expires_at = None
        if expires_in_days is not None:
            expires_at = (now + datetime.timedelta(days=expires_in_days)).isoformat()

        roles_tuple = tuple(sorted(roles))
        if permissions is None:
            p_set = set()
            for r in roles_tuple:
                p_set.update(ROLE_PERMISSIONS.get(r, ()))
            perms_tuple = tuple(sorted(p_set))
        else:
            perms_tuple = tuple(sorted(permissions))

        cred_id = f"cred-{secrets.token_hex(8)}"
        record = CredentialRecord(
            credential_id=cred_id,
            subject_id=subject_id,
            tenant_id=tenant_id,
            key_hash=k_hash,
            key_prefix=prefix,
            roles=roles_tuple,
            permissions=perms_tuple,
            created_at=created_at,
            expires_at=expires_at,
            rotated_from=rotated_from,
            description=description,
        )

        with self._lock:
            self._credentials[cred_id] = record
            self._hash_to_id[k_hash] = cred_id
            self._record_audit(
                "credential_provisioned",
                subject_id=subject_id,
                tenant_id=tenant_id,
                credential_id=cred_id,
                details=f"roles={roles_tuple}",
            )

        return raw_key, record

    def verify_key(self, api_key: str, requested_tenant: Optional[str] = None) -> Identity:
        k_hash = hash_api_key(api_key)
        record: Optional[CredentialRecord] = None

        with self._lock:
            # Constant-time lookup across keys to avoid timing attacks
            for stored_hash, cred_id in self._hash_to_id.items():
                if hmac.compare_digest(k_hash, stored_hash):
                    record = self._credentials.get(cred_id)
                    break

        if not record:
            self._record_audit("auth_invalid_key", details="key_hash_not_found")
            metrics.inc("rag_auth_attempts_total", {"status": "failure", "tenant_id": requested_tenant or "unknown"})
            metrics.inc("rag_auth_failures_total", {"reason": "invalid_key", "tenant_id": requested_tenant or "unknown"})
            raise AuthenticationError("missing or invalid API key", component="authentication")

        if record.revoked:
            self._record_audit(
                "auth_revoked",
                subject_id=record.subject_id,
                tenant_id=record.tenant_id,
                credential_id=record.credential_id,
            )
            metrics.inc("rag_auth_attempts_total", {"status": "failure", "tenant_id": record.tenant_id})
            metrics.inc("rag_auth_failures_total", {"reason": "revoked", "tenant_id": record.tenant_id})
            raise AuthenticationError("credential has been revoked", component="authentication")

        if record.expires_at:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            if now > record.expires_at:
                self._record_audit(
                    "auth_expired",
                    subject_id=record.subject_id,
                    tenant_id=record.tenant_id,
                    credential_id=record.credential_id,
                )
                metrics.inc("rag_auth_attempts_total", {"status": "failure", "tenant_id": record.tenant_id})
                metrics.inc("rag_auth_failures_total", {"reason": "expired", "tenant_id": record.tenant_id})
                raise AuthenticationError("credential has expired", component="authentication")

        # Tenant boundary check: client-supplied tenant cannot override credential's tenant
        if requested_tenant and requested_tenant.strip():
            req_t = requested_tenant.strip()
            if req_t != record.tenant_id:
                self._record_audit(
                    "auth_tenant_mismatch",
                    subject_id=record.subject_id,
                    tenant_id=record.tenant_id,
                    credential_id=record.credential_id,
                    details=f"requested={req_t}, owned={record.tenant_id}",
                )
                metrics.inc("rag_auth_attempts_total", {"status": "failure", "tenant_id": record.tenant_id})
                metrics.inc("rag_auth_failures_total", {"reason": "tenant_mismatch", "tenant_id": record.tenant_id})
                metrics.inc("rag_cross_tenant_access_blocked_total", {"attempted_tenant": req_t, "actual_tenant": record.tenant_id})
                metrics.inc("rag_cross_tenant_access_total", {"action": "auth_header_override"})
                raise AuthorizationError(
                    f"credential belongs to tenant '{record.tenant_id}', cannot access requested tenant '{req_t}'",
                    component="authorization",
                )

        self._record_audit(
            "auth_success",
            subject_id=record.subject_id,
            tenant_id=record.tenant_id,
            credential_id=record.credential_id,
        )
        metrics.inc("rag_auth_attempts_total", {"status": "success", "tenant_id": record.tenant_id})
        metrics.inc("rag_auth_success_total", {"tenant": record.tenant_id})

        return Identity(
            subject_id=record.subject_id,
            credential_id=record.credential_id,
            tenant_id=record.tenant_id,
            roles=record.roles,
            permissions=record.permissions,
            authentication_method="api_key",
            credential_status="active",
            authenticated=True,
        )

    def has_key_hash(self, key_hash: str) -> bool:
        with self._lock:
            for stored_hash in self._hash_to_id:
                if hmac.compare_digest(key_hash, stored_hash):
                    return True
            return False

    def revoke(self, credential_id: str) -> bool:
        with self._lock:
            record = self._credentials.get(credential_id)
            if not record:
                return False
            record.revoked = True
            record.revoked_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self._record_audit(
                "credential_revoked",
                subject_id=record.subject_id,
                tenant_id=record.tenant_id,
                credential_id=record.credential_id,
            )
            try:
                metrics.inc("rag_credential_revocations_total", {"tenant_id": record.tenant_id})
            except Exception:
                pass
            return True

    def rotate(self, credential_id: str, expires_in_days: Optional[int] = None) -> Tuple[str, CredentialRecord]:
        with self._lock:
            old_record = self._credentials.get(credential_id)
            if not old_record:
                raise AuthenticationError(f"credential '{credential_id}' not found", component="authentication")
            # Revoke old credential
            old_record.revoked = True
            old_record.revoked_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

            # Provision new replacement credential
            raw_key, new_record = self.provision(
                tenant_id=old_record.tenant_id,
                subject_id=old_record.subject_id,
                roles=old_record.roles,
                permissions=old_record.permissions,
                expires_in_days=expires_in_days,
                description=f"Rotated from {old_record.credential_id}",
                rotated_from=old_record.credential_id,
            )
            self._record_audit(
                "credential_rotated",
                subject_id=old_record.subject_id,
                tenant_id=old_record.tenant_id,
                credential_id=new_record.credential_id,
                details=f"from={old_record.credential_id}",
            )
            return raw_key, new_record

    def list_credentials(self, tenant_id: Optional[str] = None) -> List[CredentialRecord]:
        with self._lock:
            items = list(self._credentials.values())
            if tenant_id:
                items = [item for item in items if item.tenant_id == tenant_id]
            return items

    def get_by_id(self, credential_id: str) -> Optional[CredentialRecord]:
        with self._lock:
            return self._credentials.get(credential_id)

    def get_audit_events(self, tenant_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            events = self._audit_events
            if tenant_id:
                events = [e for e in events if e.tenant_id == tenant_id]
            return [e.to_dict() for e in events[-limit:]]

    def clear(self) -> None:
        with self._lock:
            self._credentials.clear()
            self._hash_to_id.clear()
            self._audit_events.clear()


_registry = CredentialRegistry()


def get_credential_registry() -> CredentialRegistry:
    return _registry


def reset_credential_registry() -> None:
    _registry.clear()


def _tenant_allowlist() -> list[str]:
    return [t.strip() for t in settings.TENANT_IDS.split(",") if t.strip()]


def _resolve_tenant(requested: Optional[str]) -> str:
    tenant = (requested or "default").strip() or "default"
    allowed = _tenant_allowlist()
    if allowed and tenant not in allowed:
        raise AuthenticationError("tenant is not in the configured allowlist", component="authorization")
    return tenant


def authenticate(api_key: Optional[str], tenant_id: Optional[str] = None) -> Identity:
    """Authenticate API key against credential registry with tenant boundary checks.

    Fails closed in production if no valid key is provided.
    """
    supplied = (api_key or "").strip()
    is_prod = settings.APP_ENV == "production"

    if is_prod and not supplied:
        _registry._record_audit("auth_missing_key", details="production_fail_closed")
        metrics.inc("rag_auth_failures_total", {"reason": "missing_key_production"})
        raise AuthenticationError("missing API key; production mode requires authentication", component="authentication")

    if supplied:
        k_hash = hash_api_key(supplied)
        if _registry.has_key_hash(k_hash):
            return _registry.verify_key(supplied, requested_tenant=tenant_id)

        # 2. Backward compatibility with static global API_KEY if configured
        configured = (settings.API_KEY or "").strip()
        if configured:
            valid = hmac.compare_digest(supplied.encode("utf-8"), configured.encode("utf-8"))
            if valid:
                tenant = _resolve_tenant(tenant_id)
                _registry._record_audit(
                    "auth_success",
                    subject_id=f"api-key:{configured[:4]}…",
                    tenant_id=tenant,
                    credential_id="legacy-global",
                )
                metrics.inc("rag_auth_success_total", {"tenant": tenant})
                return Identity(
                    subject_id=f"api-key:{configured[:4]}…",
                    credential_id="legacy-global",
                    tenant_id=tenant,
                    roles=("reader", "ingestor", "admin"),
                    permissions=ROLE_PERMISSIONS["admin"],
                    authentication_method="api_key",
                    credential_status="active",
                    authenticated=True,
                )

        _registry._record_audit("auth_invalid_key", details="key_not_recognized")
        metrics.inc("rag_auth_failures_total", {"reason": "invalid_key"})
        raise AuthenticationError("missing or invalid API key", component="authentication")

    # If static API_KEY is set, non-prod also rejects unauthenticated requests
    configured = (settings.API_KEY or "").strip()
    if configured:
        _registry._record_audit("auth_missing_key", details="key_configured_but_missing")
        metrics.inc("rag_auth_failures_total", {"reason": "missing_key"})
        raise AuthenticationError("missing or invalid API key", component="authentication")

    # Anonymous development mode (only when APP_ENV != production and no API_KEY configured)
    resolved_tenant = _resolve_tenant(tenant_id)
    return Identity(
        subject_id="anonymous",
        credential_id="none",
        tenant_id=resolved_tenant,
        roles=("reader", "ingestor", "admin"),
        permissions=ROLE_PERMISSIONS["admin"],
        authentication_method="disabled",
        credential_status="active",
        authenticated=False,
    )
