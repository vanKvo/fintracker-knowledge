"""POST /internal/retrieve on knowledge-internal (KN-RET-01, -07): Lambda handler.

API Gateway IAM auth rejects unsigned requests before this runs. Here the signed caller is checked
against the allow-list (fails closed) and the user is read from the forwarded verified context.

__author__ = "Van Vo"
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Collection
from typing import Any

from src.retrieval.metrics import RequestMetrics, RetrievalMetrics
from src.retrieval.models import RetrievalError
from src.retrieval.service import RetrievalService
from src.system_docs.access import CallerNotAllowed, authorize_caller

logger = logging.getLogger(__name__)

USER_ID_HEADER = "x-internal-user-id"


class RetrieveHandler:
    def __init__(self, service: RetrievalService, allowed_caller_arns: Collection[str], metrics: RetrievalMetrics) -> None:
        self._service = service
        self._allowed = allowed_caller_arns
        self._metrics = metrics

    def handle(self, event: dict[str, Any]) -> dict[str, Any]:
        context = event.get("requestContext") or {}
        request_id = str(context.get("requestId") or uuid.uuid4())
        caller = ((context.get("authorizer") or {}).get("iam") or {}).get("userArn")
        try:
            user_id = self._authorize(event, caller)
            result = self._service.retrieve(request_id, caller, user_id, _json_body(event))
            return _response(200, result.to_dict())
        except RetrievalError as error:
            if error.code in {"caller_not_allowed", "user_context_missing"}:
                self._count_refused(request_id, caller, error.code)
            return _error(error, request_id)
        except Exception as exc:  # noqa: BLE001 - never surface exception text
            logger.error("Unhandled retrieve error (request %s): %s", request_id, type(exc).__name__)
            return _error(RetrievalError("internal_error"), request_id)

    def _authorize(self, event: dict[str, Any], caller: str | None) -> str:
        try:
            authorize_caller(caller, self._allowed)
        except CallerNotAllowed:
            raise RetrievalError("caller_not_allowed") from None
        headers = {str(k).lower(): v for k, v in (event.get("headers") or {}).items()}
        user_id = str(headers.get(USER_ID_HEADER) or "").strip()
        if not user_id:
            raise RetrievalError("user_context_missing")
        return user_id

    def _count_refused(self, request_id: str, caller: str | None, code: str) -> None:
        self._metrics.request(RequestMetrics(request_id, caller, None, None, 0, None, code, 0.0, 0.0, 0.0, 0.0))


def _json_body(event: dict[str, Any]) -> Any:
    try:
        return json.loads(event.get("body") or "")
    except ValueError:
        raise RetrievalError("invalid_argument", "body") from None


def _response(status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {"statusCode": status, "headers": {"content-type": "application/json"}, "body": json.dumps(body)}


def _error(error: RetrievalError, request_id: str) -> dict[str, Any]:
    return _response(
        error.status, {"error": {"code": error.code, "message": error.message, "request_id": request_id}}
    )
