"""KN-RET-01, -07: caller allow-list, user context and the error body on the internal route.

__author__ = "Van Vo"
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from src.core.exceptions import EmbeddingUnavailableError
from src.retrieval.models import SearchResult
from src.retrieval.router import RetrieveHandler
from src.retrieval.service import RetrievalService
from tests.retrieval.fakes import (
    CALLER_ARN,
    REQUEST_ID,
    USER_ID,
    FakeEmbedder,
    FakeSearch,
    RecordingMetrics,
    hit,
)

BODY = {"query": "How do I confirm a CSV mapping?", "scope": "system", "k": 5}


def event(
    body: Any = BODY, *, caller: str | None = CALLER_ARN, user: str | None = USER_ID
) -> dict[str, Any]:
    headers = {"content-type": "application/json"}
    if user is not None:
        headers["x-internal-user-id"] = user
    iam = {"userArn": caller} if caller else {}
    return {
        "headers": headers,
        "body": body if isinstance(body, str) else json.dumps(body),
        "requestContext": {"requestId": REQUEST_ID, "authorizer": {"iam": iam}},
    }


def build(
    search: FakeSearch | None = None, embedder: FakeEmbedder | None = None
) -> tuple[RetrieveHandler, FakeSearch, RecordingMetrics]:
    search = search or FakeSearch(SearchResult([hit()], "9f2c1ab"))
    metrics = RecordingMetrics()
    service = RetrievalService(embedder or FakeEmbedder(), search, metrics, sleep=lambda _s: None)
    return RetrieveHandler(service, {CALLER_ARN}, metrics), search, metrics


def test_allow_listed_caller_with_user_context_is_accepted() -> None:
    handler, search, _ = build()
    response = handler.handle(event())
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert len(body["chunks"]) == 1 and search.calls[0]["user_id"] == USER_ID


def test_caller_not_on_the_allow_list_gets_403_and_no_work_is_done() -> None:
    handler, search, _ = build()
    response = handler.handle(event(caller="arn:aws:iam::111122223333:role/other"))
    assert response["statusCode"] == 403
    assert json.loads(response["body"])["code"] == "caller_not_allowed"
    assert search.calls == []


def test_request_with_no_caller_identity_is_refused_fail_closed() -> None:
    handler, _, _ = build()
    assert handler.handle(event(caller=None))["statusCode"] == 403


@pytest.mark.parametrize("user", [None, "", "   "])
def test_missing_or_blank_user_context_gets_401(user: str | None) -> None:
    handler, search, _ = build()
    response = handler.handle(event(user=user))
    assert response["statusCode"] == 401
    assert json.loads(response["body"])["code"] == "user_context_missing"
    assert search.calls == []


def test_user_id_in_the_body_is_rejected_and_never_used() -> None:
    handler, search, _ = build()
    response = handler.handle(event({**BODY, "user_id": "user-b"}))
    assert response["statusCode"] == 400
    assert search.calls == []


def test_body_that_is_not_json_gets_400() -> None:
    handler, _, _ = build()
    response = handler.handle(event("{not json"))
    assert response["statusCode"] == 400
    assert json.loads(response["body"])["code"] == "invalid_argument"


def test_error_body_is_rfc_9457_with_code_and_trace_id_but_no_request_id() -> None:
    handler, _, _ = build(embedder=FakeEmbedder(EmbeddingUnavailableError()))
    response = handler.handle(event())
    body = json.loads(response["body"])
    assert response["statusCode"] == 503
    assert response["headers"]["content-type"] == "application/problem+json"
    assert set(body) == {"type", "title", "status", "detail", "code", "retryable", "trace_id"}
    assert (body["code"], body["status"], body["retryable"]) == ("embedding_unavailable", 503, True)
    assert body["type"] == "https://fintracker.app/errors/embedding-unavailable"
    assert len(body["trace_id"]) == 32 and REQUEST_ID not in response["body"]  # app doc 4.8


def test_trace_id_comes_from_the_incoming_traceparent() -> None:
    handler, _, _ = build(embedder=FakeEmbedder(EmbeddingUnavailableError()))
    traced = event()
    traced["headers"]["traceparent"] = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    assert json.loads(handler.handle(traced)["body"])["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_unexpected_exception_returns_500_without_its_text_and_logs_the_request_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    handler, _, _ = build(embedder=FakeEmbedder(RuntimeError("secret-table kn_chunks exploded")))
    response = handler.handle(event())
    assert response["statusCode"] == 500
    assert json.loads(response["body"])["code"] == "internal_error"
    assert "secret-table" not in response["body"]
    trace_id = json.loads(response["body"])["trace_id"]
    logged = "\n".join(r.getMessage() + str(r.__dict__) for r in caplog.records)
    assert REQUEST_ID in logged and trace_id in logged  # both IDs on the log line
    assert "secret-table" not in logged  # messages of exceptions that aren't ours are redacted


def test_refused_requests_are_still_counted_in_metrics() -> None:
    handler, _, metrics = build()
    handler.handle(event(caller="arn:aws:iam::111122223333:role/other"))
    handler.handle(event(user=None))
    assert [m.error_code for m in metrics.requests] == ["caller_not_allowed", "user_context_missing"]
