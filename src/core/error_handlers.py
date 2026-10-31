"""The one place errors become HTTP responses (app doc 4.8; error catalog).

Every failure is RFC 9457 `application/problem+json`, the same as every other service:

    {"type": ".../errors/<code>", "title", "status", "detail", "code", "retryable", "trace_id"}

- `trace_id` is what a caller quotes; `request_id` (API Gateway's) is logged, never returned.
- 4xx: logged at WARNING with the code and the developer `log_message`; no traceback.
- 5xx: logged at ERROR with the exception type and cause chain. Messages of exceptions that are
  not ours are redacted (they can carry query or chunk text, app doc 4.5).
- Exception text, SQL, request bodies and input values never reach the caller.
- Rejections of the internal-call checks (401/403/400 on the headers) are the shared library's handler.

__author__ = "Van Vo"
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import observability  # noqa: F401  (XS-ACCT-06: installs the card-number log mask)
from .exceptions import (
    AppError,
    DependencyError,
    InvalidArgumentError,
    MethodNotAllowedError,
    NotFoundError,
)
from .request_context import ids_for

TYPE_BASE = "https://fintracker.app/errors/"
PROBLEM_JSON = "application/problem+json"
_REDACTED = "<message redacted: may contain user data>"

logger = logging.getLogger(__name__)

_HTTP_STATUS_ERRORS: dict[int, type[AppError]] = {
    400: InvalidArgumentError,
    404: NotFoundError,
    405: MethodNotAllowedError,
}


def problem_response(error: AppError, trace_id: str, headers: dict[str, str] | None = None) -> JSONResponse:
    body = {
        "type": TYPE_BASE + error.code.replace("_", "-"),
        "title": error.title,
        "status": error.status,
        "detail": error.user_message,
        "code": error.code,
        "retryable": error.retryable,
        "trace_id": trace_id,
    }
    return JSONResponse(status_code=error.status, content=body, media_type=PROBLEM_JSON, headers=headers)


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


def _record(request: Request, error: AppError, cause: BaseException | None = None) -> None:
    trace_id, request_id = ids_for(request)
    route = getattr(request.scope.get("route"), "path", None)  # the pattern, never the raw path
    ids = {"trace_id": trace_id, "request_id": request_id, "code": error.code, "status": error.status,
           "method": request.method, "route": route, "reason": error.log_message}
    if error.status >= 500:
        failure = cause or error
        logger.error("Request failed (trace %s, request %s): %s", trace_id, request_id, error.code,
                     extra={**ids, "exception_name": type(failure).__name__, "traceback": safe_traceback(failure)})
    else:
        logger.warning("Request rejected (trace %s, request %s): %s", trace_id, request_id, error.code, extra=ids)


async def app_error_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, AppError)
    _record(request, exc, cause=exc)
    return problem_response(exc, ids_for(request)[0])


async def validation_error_handler(request: Request, exc: Exception) -> Response:
    """A schema failure names nothing of the input: the route validates its own body (KN-RET-02)."""
    error = InvalidArgumentError.for_field("request")
    _record(request, error)
    return problem_response(error, ids_for(request)[0])


async def http_exception_handler(request: Request, exc: Exception) -> Response:
    """Framework errors (unknown route, wrong method) in the same shape as our own, never echoing the path."""
    assert isinstance(exc, StarletteHTTPException)
    error_type = _HTTP_STATUS_ERRORS.get(exc.status_code)
    if error_type is None:
        error = AppError(log_message=f"unmapped HTTPException {exc.status_code}")
        error.status = exc.status_code if 400 <= exc.status_code < 600 else 500
    else:
        error = error_type(log_message=f"HTTPException {exc.status_code}")
    _record(request, error)
    allow = (exc.headers or {}).get("Allow")
    return problem_response(error, ids_for(request)[0], {"Allow": allow} if allow else None)


async def request_context_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Last line of defense: any exception that is not an `AppError` is a bug and a 500 `internal_error`.

    Caught here rather than with `exception_handler(Exception)`: Starlette runs that handler outside the middleware
    stack, where the request state is no longer the same.
    """
    ids_for(request)
    try:
        return await call_next(request)
    except Exception as exc:  # noqa: BLE001 - the one place every unexpected failure becomes a response
        error = AppError(log_message=f"unhandled {type(exc).__name__}")
        _record(request, error, cause=exc)
        return problem_response(error, ids_for(request)[0])


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.middleware("http")(request_context_middleware)
