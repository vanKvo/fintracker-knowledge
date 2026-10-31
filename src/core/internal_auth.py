"""KN-RET-01, KN-FAPI-02 / LB-IHTTP-09: who may call an internal route, and on whose behalf.

Both checks are the shared fintracker-internal-http library's FastAPI dependencies, in its order: the caller is decided
first (the ARN API Gateway verified, against INTERNAL_ALLOWED_CALLER_ARNS; unset or empty rejects every call), and only
then is the user read, from X-Internal-User-Id. No verified caller is 401 `unauthorized`; an unlisted caller is 403
`caller_not_allowed`; no user is 401 `user_context_missing`; a user that is not a UUID is 400 `invalid_argument`.
The rejections are rendered by the library's handler (installed by the app factory).

INTERNAL_AUTH_MODE=local (never set in AWS) reads the caller from a plain request header instead.

A refused request is still counted in the request metrics (KN-RET-08), so this wraps the library's dependencies
rather than replacing them.

__author__ = "Van Vo"
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Request
from fintracker_internal_http import InternalRejection, get_internal_user, require_internal_caller

from src.core.dependencies import get_metrics
from src.core.request_context import ids_for
from src.retrieval.metrics import RequestMetrics, RetrievalMetrics

_verified_user = get_internal_user(require=("user",))


@dataclass(frozen=True)
class InternalCall:
    caller: str
    user_id: str


async def internal_call(request: Request, metrics: RetrievalMetrics = Depends(get_metrics)) -> InternalCall:
    """The verified caller and the forwarded user, or the library's rejection (after counting it)."""
    request_id = ids_for(request)[1]
    try:
        caller = await require_internal_caller(request)
    except InternalRejection as rejection:
        _count_refused(metrics, request_id, None, rejection.code)
        raise
    try:
        user = await _verified_user(request, caller)
    except InternalRejection as rejection:
        _count_refused(metrics, request_id, caller, rejection.code)
        raise
    return InternalCall(caller, user.user_id)


def _count_refused(metrics: RetrievalMetrics, request_id: str, caller: str | None, code: str) -> None:
    metrics.request(RequestMetrics(request_id, caller, None, None, 0, None, code, 0.0, 0.0, 0.0, 0.0))
