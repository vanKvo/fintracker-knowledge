"""KN-RET-02 to KN-RET-08: validate, embed, search, re-check ownership, shape the response.

The user always arrives as an argument from the verified context, never from the body. Every step
failure becomes one documented RetrievalError; exception text, query text and chunk text are never
logged or returned (only the exception type, IDs, counts, scores and timings).

__author__ = "Van Vo"
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Protocol

from src.config.knowledge_config import EmbeddingConfig
from src.embedding.models import EmbeddingInputTooLong, EmbeddingUnavailable
from src.retrieval.metrics import RequestMetrics, RetrievalMetrics
from src.retrieval.models import (
    DatabaseUnavailable,
    RetrievalError,
    RetrievedChunk,
    RetrieveRequest,
    RetrieveResponse,
    SearchHit,
    parse_request,
)
from src.retrieval.repository import SEARCH_POOL, ChunkSearch

logger = logging.getLogger(__name__)

TIME_BUDGET_SECONDS = 3.0
DB_RETRY_DELAY_SECONDS = 1.0
MAX_CHUNKS_PER_DOCUMENT = 3  # one long doc must not fill all k slots (open question 1)


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> tuple[float, ...]: ...


@dataclass
class _Progress:
    """Filled by the worker thread so a timeout or failure still reports what was reached."""

    request: RetrieveRequest | None = None
    result_count: int = 0
    top_score: float | None = None
    validate_ms: float = 0.0
    embed_ms: float = 0.0
    search_ms: float = 0.0


class RetrievalService:
    def __init__(
        self,
        embedder: QueryEmbedder,
        search: ChunkSearch,
        metrics: RetrievalMetrics,
        config: EmbeddingConfig = EmbeddingConfig(),
        *,
        budget_seconds: float = TIME_BUDGET_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._embedder = embedder
        self._search = search
        self._metrics = metrics
        self._config = config
        self._budget = budget_seconds
        self._sleep = sleep

    def retrieve(self, request_id: str, caller: str | None, user_id: str, body: Any) -> RetrieveResponse:
        started = time.monotonic()
        progress = _Progress()
        error: RetrievalError | None = None
        # A worker thread lets the call be abandoned at the budget (KN-RET-07); it can't be cancelled,
        # but its result is ignored and its own client timeouts bound how long it lingers.
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            return pool.submit(self._run, request_id, user_id, body, progress).result(timeout=self._budget)
        except FutureTimeout:
            error = RetrievalError("timeout")
            raise error from None
        except RetrievalError as exc:
            error = exc
            raise
        except Exception as exc:  # noqa: BLE001 - anything else is reported generically
            logger.error("Retrieval failed (request %s): %s", request_id, type(exc).__name__)
            error = RetrievalError("internal_error")
            raise error from None
        finally:
            pool.shutdown(wait=False)
            self._emit(request_id, caller, progress, error, (time.monotonic() - started) * 1000)

    def _run(self, request_id: str, user_id: str, body: Any, progress: _Progress) -> RetrieveResponse:
        step = time.monotonic()
        progress.request = request = parse_request(body)
        progress.validate_ms = _since(step)

        step = time.monotonic()
        vector = self._embed(request.query)
        progress.embed_ms = _since(step)

        step = time.monotonic()
        result = self._search_with_retry(user_id, vector, request.scope)
        progress.search_ms = _since(step)

        owned = [h for h in result.hits if self._owned(h, request.scope, user_id, request_id)]
        chunks = _top_chunks(owned, request.k)
        progress.result_count = len(chunks)
        progress.top_score = chunks[0].score if chunks else None
        return RetrieveResponse(chunks, self._config.active_embedding_model, result.index_commit or "unknown")

    def _embed(self, query: str) -> tuple[float, ...]:
        try:
            return self._embedder.embed_query(query)
        except EmbeddingUnavailable:
            raise RetrievalError("embedding_unavailable") from None
        except EmbeddingInputTooLong:
            raise RetrievalError("invalid_argument", "query") from None

    def _search_with_retry(self, user_id: str, vector: tuple[float, ...], scope: str):  # type: ignore[no-untyped-def]
        model = self._config.active_embedding_model
        try:
            return self._search.search(user_id, vector, scope, model, SEARCH_POOL)
        except DatabaseUnavailable:
            self._sleep(DB_RETRY_DELAY_SECONDS)  # also covers an Aurora resume from pause in dev
        try:
            return self._search.search(user_id, vector, scope, model, SEARCH_POOL)
        except DatabaseUnavailable:
            raise RetrievalError("database_unavailable") from None

    def _owned(self, hit: SearchHit, scope: str, user_id: str, request_id: str) -> bool:
        """KN-RET-05: second line of defence behind RLS. A mismatch is dropped and signalled."""
        ok = hit.scope == scope and (hit.user_id is None if scope == "system" else hit.user_id == user_id)
        if not ok:
            self._metrics.ownership_mismatch(hit.chunk_id, request_id)
        return ok

    def _emit(
        self, request_id: str, caller: str | None, progress: _Progress, error: RetrievalError | None, total_ms: float
    ) -> None:
        request = progress.request
        try:
            self._metrics.request(
                RequestMetrics(
                    request_id, caller, request.scope if request else None, request.k if request else None,
                    0 if error else progress.result_count, None if error else progress.top_score,
                    error.code if error else None, progress.validate_ms, progress.embed_ms, progress.search_ms,
                    total_ms,
                )
            )
        except Exception as exc:  # noqa: BLE001 - metrics must never fail a request
            logger.error("Metrics emit failed (request %s): %s", request_id, type(exc).__name__)


def _since(start: float) -> float:
    return (time.monotonic() - start) * 1000


def _top_chunks(hits: list[SearchHit], k: int) -> list[RetrievedChunk]:
    per_document: dict[str, int] = {}
    chunks: list[RetrievedChunk] = []
    for h in sorted(hits, key=lambda hit: hit.distance):
        if per_document.get(h.document_id, 0) >= MAX_CHUNKS_PER_DOCUMENT:
            continue
        per_document[h.document_id] = per_document.get(h.document_id, 0) + 1
        score = min(1.0, max(0.0, round(1 - h.distance, 3)))
        chunks.append(RetrievedChunk(h.chunk_id, h.text, h.source_path, h.heading_path, score))
        if len(chunks) == k:
            break
    return chunks
