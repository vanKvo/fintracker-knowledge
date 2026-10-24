"""POST /internal/retrieve on knowledge-internal (KN-RET-01, -07): Lambda handler.

API Gateway IAM auth rejects unsigned requests before this runs. Here the signed caller is checked
against the allow-list (fails closed) and the user is read from the forwarded verified context.
Every failure becomes an RFC 9457 response through `core/error_handlers.handle_error` (app doc 4.8).

__author__ = "Van Vo"
"""

from __future__ import annotations

import json
from collections.abc import Collection
from typing import Any

from src.core.error_handlers import handle_error
from src.core.exceptions import (
    CallerNotAllowedError,
    InvalidArgumentError,
    UserContextMissingError,
)
from src.core.request_context import ids_for, lower_headers
from src.retrieval.metrics import RequestMetrics, RetrievalMetrics
from src.retrieval.service import RetrievalService
from src.system_docs.access import authorize_caller

USER_ID_HEADER = "x-internal-user-id"
ROUTE = "POST /internal/retrieve"


class RetrieveHandler:
    def __init__(self, service: RetrievalService, allowed_caller_arns: Collection[str], metrics: RetrievalMetrics) -> None:
        self._service = service
        self._allowed = allowed_caller_arns
        self._metrics = metrics

    def handle(self, event: dict[str, Any]) -> dict[str, Any]:
        trace_id, request_id = ids_for(event)
        context = event.get("requestContext") or {}
        caller = ((context.get("authorizer") or {}).get("iam") or {}).get("userArn")
        try:
            user_id = self._authorize(event, caller)
            result = self._service.retrieve(request_id, caller, user_id, _json_body(event))
            return _response(200, result.to_dict())
        except (CallerNotAllowedError, UserContextMissingError) as error:
            self._count_refused(request_id, caller, error.code)
            return handle_error(error, trace_id=trace_id, request_id=request_id, route=ROUTE)
        except Exception as exc:  # noqa: BLE001 - the one place every failure becomes a response
            return handle_error(exc, trace_id=trace_id, request_id=request_id, route=ROUTE)

    def _authorize(self, event: dict[str, Any], caller: str | None) -> str:
        authorize_caller(caller, self._allowed)
        user_id = lower_headers(event).get(USER_ID_HEADER, "").strip()
        if not user_id:
            raise UserContextMissingError(log_message=f"no {USER_ID_HEADER} header")
        return user_id

    def _count_refused(self, request_id: str, caller: str | None, code: str) -> None:
        self._metrics.request(RequestMetrics(request_id, caller, None, None, 0, None, code, 0.0, 0.0, 0.0, 0.0))


def _json_body(event: dict[str, Any]) -> Any:
    try:
        return json.loads(event.get("body") or "")
    except ValueError:
        raise InvalidArgumentError.for_field("body") from None


def _response(status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {"statusCode": status, "headers": {"content-type": "application/json"}, "body": json.dumps(body)}
