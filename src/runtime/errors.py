"""Explicit, structured runtime failures.

These exceptions are part of the public API contract: callers can map them to
HTTP status codes without parsing error strings, and logs never need to expose
provider payloads.
"""
from __future__ import annotations


class RuntimeFailure(RuntimeError):
    """Base class for failures that are safe to surface to API clients."""

    code = "runtime_failure"
    http_status = 500

    def __init__(self, message: str, *, component: str | None = None, retryable: bool = False):
        super().__init__(message)
        self.component = component
        self.retryable = retryable

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": str(self),
            "component": self.component,
            "retryable": self.retryable,
        }


class DependencyUnavailable(RuntimeFailure):
    code = "dependency_unavailable"
    http_status = 503


class ModelUnavailable(DependencyUnavailable):
    code = "model_unavailable"


class IndexUnavailable(DependencyUnavailable):
    code = "index_unavailable"


class IndexBuildError(RuntimeFailure):
    code = "index_build_failed"


class IngestionError(RuntimeFailure):
    code = "ingestion_failed"
    http_status = 400


class DocumentNotFound(IngestionError):
    code = "document_not_found"
    http_status = 404


class UnsupportedDocument(IngestionError):
    code = "unsupported_document"
    http_status = 415


class DocumentTooLarge(IngestionError):
    code = "document_too_large"
    http_status = 413


class GenerationContractError(RuntimeFailure):
    code = "generation_contract_error"
    http_status = 502


class CitationValidationError(RuntimeFailure):
    code = "citation_validation_error"
    http_status = 422


class AuthorizationError(RuntimeFailure):
    code = "forbidden"
    http_status = 403


class AuthenticationError(RuntimeFailure):
    code = "unauthorized"
    http_status = 401
