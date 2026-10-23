"""KN-RET-08: per-request metrics and the ownership-mismatch signal. IDs, counts, scores and timings only.

__author__ = "Van Vo"
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

EMPTY_RATE_THRESHOLD = 0.30
EMPTY_RATE_MIN_REQUESTS = 20


@dataclass(frozen=True)
class RequestMetrics:
    request_id: str
    caller: str | None
    scope: str | None
    k: int | None
    result_count: int
    top_score: float | None
    error_code: str | None
    validate_ms: float
    embed_ms: float
    search_ms: float
    total_ms: float


class RetrievalMetrics(Protocol):
    def request(self, metrics: RequestMetrics) -> None: ...
    def ownership_mismatch(self, chunk_id: str, request_id: str) -> None: ...


class LogMetrics:
    """Structured log lines that a metric filter turns into the dashboard and alarms (IaC not built yet)."""

    def request(self, metrics: RequestMetrics) -> None:
        logger.info(json.dumps({"event": "kn.retrieve.request", **asdict(metrics)}))

    def ownership_mismatch(self, chunk_id: str, request_id: str) -> None:
        # Alarm fires on the first occurrence; never any chunk text.
        logger.error(
            json.dumps(
                {"event": "kn.retrieve.ownership_mismatch", "value": 1, "chunk_id": chunk_id, "request_id": request_id}
            )
        )


def empty_result_alarm(requests: int, empty: int) -> bool:
    """Over 30% empty results across one hour, with at least 20 requests."""
    return requests >= EMPTY_RATE_MIN_REQUESTS and empty / requests > EMPTY_RATE_THRESHOLD
