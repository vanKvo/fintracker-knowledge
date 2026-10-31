"""KN-FAPI-08: `GET /health` (the process is up) and `GET /ready` (it can serve).

Neither carries user data and neither is routed by API Gateway, so neither needs a caller or a user. `/health` touches
nothing. `/ready` needs the database to answer as the retrieve role and the embedding client to have credentials.

__author__ = "Van Vo"
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from src.core.container import Container
from src.core.dependencies import get_container
from src.core.exceptions import EmbeddingUnavailableError

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(container: Container = Depends(get_container)) -> dict[str, str]:
    await run_in_threadpool(container.check_database)  # raises DatabaseUnavailableError: 503
    if not await run_in_threadpool(container.embedding_configured):
        raise EmbeddingUnavailableError(log_message="no AWS credentials for the embedding client")
    return {"status": "ready"}
