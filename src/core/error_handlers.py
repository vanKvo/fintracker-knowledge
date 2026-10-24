"""The one place errors become HTTP responses (app doc 4.8; error catalog).

Knowledge runs plain Lambda handlers (no web framework), so this is a function each handler calls
from one `except` block rather than registered handlers. The body is RFC 9457
`application/problem+json`, the same as every other service:

    {"type": ".../errors/<code>", "title", "status", "detail", "code", "retryable", "trace_id"}

- `trace_id` is what a caller quotes; `request_id` (API Gateway's) is logged, never returned.
- 4xx: logged at WARNING with the code and the developer `log_message`; no traceback.
- 5xx: logged at ERROR with the exception type and cause chain. Messages of exceptions that are
  not ours are redacted (they can carry query or chunk text, app doc 4.5).
- Exception text, SQL, request bodies and input values never reach the caller.

__author__ = "Van Vo"
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any

from .exceptions import AppError, DependencyError

TYPE_BASE = "https://fintracker.app/errors/"
PROBLEM_JSON = "application/problem+json"
_REDACTED = "<message redacted: may contain user data>"

logger = logging.getLogger(__name__)


def problem_response(error: AppError, trace_id: str) -> dict[str, Any]:
    body = {
        "type": TYPE_BASE + error.code.replace("_", "-"),
        "title": error.title,
        "status": error.status,
        "detail": error.user_message,
        "code": error.code,
        "retryable": error.retryable,
        "trace_id": trace_id,
    }
    return {"statusCode": error.status, "headers": {"content-type": PROBLEM_JSON}, "body": json.dumps(body)}


def _chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and all(current is not seen for seen in chain):
        chain.append(current)
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return chain[::-1]


def safe_traceback(exc: BaseException) -> str:
    """Python's traceback text, with every frame, minus messages we don't control."""
    parts: list[str] = []
    for index, error in enumerate(_chain(exc)):
        if index:
            parts.append("\nThe above exception led to the following exception:\n\n")
        parts.append("Traceback (most recent call last):\n")
        parts.extend(traceback.format_tb(error.__traceback__))
        message = str(error) if isinstance(error, (AppError, DependencyError)) else _REDACTED
        parts.append(f"{type(error).__module__}.{type(error).__qualname__}: {message}\n")
    return "".join(parts)


def handle_error(exc: Exception, *, trace_id: str, request_id: str, route: str) -> dict[str, Any]:
    """Logs the failure with both IDs and returns its problem+json response. Any exception that is
    not an `AppError` is a bug: a 500 `internal_error`."""
    error = exc if isinstance(exc, AppError) else AppError(log_message=f"unhandled {type(exc).__name__}")
    ids = {"trace_id": trace_id, "request_id": request_id, "code": error.code, "status": error.status,
           "route": route, "reason": error.log_message}
    if error.status >= 500:
        logger.error("Request failed (trace %s, request %s): %s", trace_id, request_id, error.code,
                     extra={**ids, "exception_name": type(exc).__name__,
                            "traceback": safe_traceback(exc)})
    else:
        logger.warning("Request rejected (trace %s, request %s): %s", trace_id, request_id, error.code,
                       extra=ids)
    return problem_response(error, trace_id)
