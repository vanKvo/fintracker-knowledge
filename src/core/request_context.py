"""Per-request correlation IDs (app doc 4.7, 4.8).

- `trace_id` is what a caller sees and quotes: the request's trace ID in W3C form (32 hex), taken
  from `traceparent`, else the X-Ray header or environment, else a new random one.
- `request_id` is API Gateway's request ID (Mangum puts the event in the ASGI scope as `aws.event`), else a new one.
  It is logged, never returned.

Computed once per request and kept in the request state, so the route, the error handlers and the logs agree.

__author__ = "Van Vo"
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Any

from starlette.requests import Request

_TRACEPARENT = re.compile(r"^[0-9a-f]{2}-([0-9a-f]{32})-[0-9a-f]{16}-[0-9a-f]{2}$")
_XRAY_ROOT = re.compile(r"Root=1-([0-9a-f]{8})-([0-9a-f]{24})")
_ZERO_TRACE = "0" * 32


def _from_traceparent(value: str | None) -> str | None:
    match = _TRACEPARENT.match((value or "").strip().lower())
    return match.group(1) if match and match.group(1) != _ZERO_TRACE else None


def _from_xray(value: str | None) -> str | None:
    match = _XRAY_ROOT.search(value or "")
    return match.group(1) + match.group(2) if match else None


def _gateway_request_id(request: Request) -> str | None:
    event: Any = request.scope.get("aws.event")
    context = event.get("requestContext") if isinstance(event, dict) else None
    value = context.get("requestId") if isinstance(context, dict) else None
    return value if isinstance(value, str) and value else None


def ids_for(request: Request) -> tuple[str, str]:
    """(trace_id, request_id) of this request."""
    state = request.state
    if not hasattr(state, "trace_id"):
        state.trace_id = (
            _from_traceparent(request.headers.get("traceparent"))
            or _from_xray(request.headers.get("x-amzn-trace-id"))
            or _from_xray(os.environ.get("_X_AMZN_TRACE_ID"))
            or uuid.uuid4().hex
        )
        state.request_id = _gateway_request_id(request) or str(uuid.uuid4())
    return str(state.trace_id), str(state.request_id)
