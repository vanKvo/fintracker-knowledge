"""Exception contract shared by every feature of the Knowledge Service.

App doc 4.8 "Errors" and the error catalog (docs/fintracker-architecture/error-catalog.md), the same
contract as the AI assistant's and Analytics' `core/exceptions.py`:

- `AppError` is the base of every error a caller may see. Each subclass fixes its HTTP status, its
  stable `code` (listed in the catalog, never renamed) and a `user_message` that is safe to return.
  `core/error_handlers.py` turns any `AppError` into an RFC 9457 response; no handler builds one.
- `log_message` is for developers only. It goes to the logs, never to the caller. Neither message
  may contain query, chunk or document text (app doc 4.5).
- `DependencyError` is a call to AWS or the database that failed. It never reaches a caller as is:
  the service retries or raises an `AppError`, chaining the cause with `raise ... from exc`.

Every Knowledge route is internal (service to service), so `user_message` is for the calling
service's logs, not a person; the AI assistant treats every Knowledge error as "no docs" (KN-RET-07).

Feature-specific errors live in `<feature>/exceptions.py`. Only errors shared by more than one
feature belong here.

__author__ = "Van Vo"
"""

from __future__ import annotations


class AppError(Exception):
    """Base of every caller-facing error. Defaults to a 500 `internal_error`."""

    status: int = 500
    code: str = "internal_error"
    title: str = "Internal server error"
    user_message: str = "Something went wrong. The error has been recorded."
    retryable: bool = False

    def __init__(self, user_message: str | None = None, *, log_message: str | None = None) -> None:
        self.user_message = user_message or type(self).user_message
        self.log_message = log_message
        super().__init__(log_message or self.code)


class DependencyError(Exception):
    """A downstream call failed. Internal: its message is for logs, never for a caller."""


# --- 4xx: the caller can fix it ------------------------------------------------------------------


class InvalidArgumentError(AppError):
    status, code, title = 400, "invalid_argument", "Invalid request"
    user_message = "The request is not valid."

    @classmethod
    def for_field(cls, field: str) -> InvalidArgumentError:
        """Names the field at fault, never its value (it may be the user's query)."""
        return cls(f"The request is not valid: {field}.", log_message=f"invalid {field}")


class ScopeNotEnabledError(AppError):
    """KN-RET-02: `scope = user` before KN-UDOC ships."""

    status, code, title = 400, "scope_not_enabled", "Scope not enabled"
    user_message = "This scope is not available yet."


class UserContextMissingError(AppError):
    """KN-RET-01: a signed request without a forwarded user context."""

    status, code, title = 401, "user_context_missing", "User context missing"
    user_message = "The request has no verified user context."


class CallerNotAllowedError(AppError):
    """KN-RET-01: the signed caller is not on the allow-list (fails closed)."""

    status, code, title = 403, "caller_not_allowed", "Caller not allowed"
    user_message = "The caller is not allowed to do this."


# --- 5xx: our side -------------------------------------------------------------------------------


class ServiceUnavailableError(AppError):
    """A dependency is down; the same request may succeed shortly."""

    status, code, title = 503, "service_unavailable", "Service unavailable"
    user_message = "This is temporarily unavailable. Try again in a moment."
    retryable = True


class EmbeddingUnavailableError(ServiceUnavailableError):
    """KN-RET-03: Bedrock embedding failed after retries."""

    code, title = "embedding_unavailable", "Embedding unavailable"
    user_message = "Embedding is temporarily unavailable."


class DatabaseUnavailableError(ServiceUnavailableError):
    code, title = "database_unavailable", "Database unavailable"
    user_message = "The database is temporarily unavailable."


class DeadlineExceededError(AppError):
    """KN-RET-07: the request ran past its time budget."""

    status, code, title = 504, "timeout", "Request timed out"
    user_message = "The request took too long."
    retryable = True
