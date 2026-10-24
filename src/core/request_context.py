"""Per-request correlation IDs (app doc 4.7, 4.8), for plain Lambda events.

- `trace_id` is what a caller sees and quotes: the request's trace ID in W3C form (32 hex), taken
  from `traceparent`, else the X-Ray header or Lambda environment, else a new random one.
- `request_id` is API Gateway's request ID (else a new one). It is logged, never returned.

Same rules as the AI assistant's `core/request_context.py`, without the Starlette dependency.

__author__ = "Van Vo"
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Mapping
from typing import Any

_TRACEPARENT = re.compile(r"^[0-9a-f]{2}-([0-9a-f]{32})-[0-9a-f]{16}-[0-9a-f]{2}$")
_XRAY_ROOT = re.compile(r"Root=1-([0-9a-f]{8})-([0-9a-f]{24})")
_ZERO_TRACE = "0" * 32


def _from_traceparent(value: str | None) -> str | None:
    match = _TRACEPARENT.match((value or "").strip().lower())
    return match.group(1) if match and match.group(1) != _ZERO_TRACE else None


def _from_xray(value: str | None) -> str | None:
    match = _XRAY_ROOT.search(value or "")
    return match.group(1) + match.group(2) if match else None


def lower_headers(event: Mapping[str, Any]) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in (event.get("headers") or {}).items()}


def ids_for(event: Mapping[str, Any]) -> tuple[str, str]:
    """(trace_id, request_id) for an API Gateway event."""
    headers = lower_headers(event)
    trace_id = (_from_traceparent(headers.get("traceparent")) or _from_xray(headers.get("x-amzn-trace-id"))
                or _from_xray(os.environ.get("_X_AMZN_TRACE_ID")) or uuid.uuid4().hex)
    request_id = str((event.get("requestContext") or {}).get("requestId") or uuid.uuid4())
    return trace_id, request_id
