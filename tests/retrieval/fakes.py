"""Test doubles for the retrieval service.

__author__ = "Van Vo"
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

from src.core.exceptions import EmbeddingUnavailableError
from src.retrieval.metrics import RequestMetrics
from src.retrieval.models import SearchHit, SearchResult

USER_ID = "user-a"
OTHER_USER_ID = "user-b"
CALLER_ARN = "arn:aws:iam::111122223333:role/ai-assistant"
REQUEST_ID = "req-123"
MODEL = "amazon.titan-embed-text-v2:0"
VECTOR = tuple([1.0] + [0.0] * 1023)


def hit(
    distance: float = 0.1,
    *,
    document_id: uuid.UUID | None = None,
    scope: str = "system",
    user_id: str | None = None,
    source_path: str = "help/statements/csv-mapping.md",
    heading_path: str = "Statements > CSV mapping",
    text: str = "To confirm a column mapping, open the import.",
) -> SearchHit:
    return SearchHit(
        chunk_id=str(uuid.uuid4()),
        document_id=str(document_id or uuid.uuid4()),
        scope=scope,
        user_id=user_id,
        source_path=source_path,
        heading_path=heading_path,
        text=text,
        distance=distance,
    )


class FakeEmbedder:
    def __init__(self, outcome: Exception | None = None, delay: Callable[[], None] | None = None) -> None:
        self.outcome = outcome
        self.delay = delay
        self.queries: list[str] = []

    def embed_query(self, query: str) -> tuple[float, ...]:
        self.queries.append(query)
        if self.delay:
            self.delay()
        if self.outcome:
            raise self.outcome
        return VECTOR


class FakeSearch:
    """Each call pops the next outcome (a result or an exception); the last one repeats."""

    def __init__(self, *outcomes: SearchResult | Exception) -> None:
        self.outcomes = list(outcomes) or [SearchResult([], "9f2c1ab")]
        self.calls: list[dict[str, object]] = []

    def search(
        self, user_id: str, vector: tuple[float, ...], scope: str, model: str, limit: int
    ) -> SearchResult:
        self.calls.append({"user_id": user_id, "vector": vector, "scope": scope, "model": model, "limit": limit})
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class RecordingMetrics:
    def __init__(self) -> None:
        self.requests: list[RequestMetrics] = []
        self.mismatches: list[tuple[str, str]] = []

    def request(self, metrics: RequestMetrics) -> None:
        self.requests.append(metrics)

    def ownership_mismatch(self, chunk_id: str, request_id: str) -> None:
        self.mismatches.append((chunk_id, request_id))


def unavailable() -> EmbeddingUnavailableError:
    return EmbeddingUnavailableError()
