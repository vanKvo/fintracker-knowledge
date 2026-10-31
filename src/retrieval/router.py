"""POST /internal/retrieve on knowledge-internal (KN-RET-01, -07; KN-FAPI-01, -02).

A thin route: the caller check and user context come first (`core/internal_auth.py`), then the body is parsed and
`RetrievalService` does the rest. Failures are `AppError`s that the app's handlers turn into RFC 9457 responses
(`core/error_handlers.py`, app doc 4.8).

__author__ = "Van Vo"
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from src.core.dependencies import get_retrieval_service
from src.core.exceptions import InvalidArgumentError
from src.core.internal_auth import InternalCall, internal_call
from src.core.request_context import ids_for
from src.retrieval.service import RetrievalService

MAX_BODY_BYTES = 64 * 1024  # a query is at most 1,000 characters (KN-RET-02)

router = APIRouter()


@router.post("/internal/retrieve")
async def retrieve(
    request: Request,
    call: InternalCall = Depends(internal_call),
    service: RetrievalService = Depends(get_retrieval_service),
) -> JSONResponse:
    body = await _json_body(request)
    # The service is blocking (database, Bedrock) and enforces its own 3 s budget in a worker thread.
    result = await run_in_threadpool(service.retrieve, ids_for(request)[1], call.caller, call.user_id, body)
    return JSONResponse(result.to_dict())


async def _json_body(request: Request) -> Any:
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise InvalidArgumentError.for_field("body")
    try:
        return json.loads(raw)
    except ValueError:
        raise InvalidArgumentError.for_field("body") from None
