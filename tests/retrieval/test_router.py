"""KN-RET-01, -07: caller allow-list, user context and the error body on the internal route.

__author__ = "Van Vo"
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from src.retrieval.models import RetrievalError, SearchResult
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
    assert json.loads(response["body"])["error"]["code"] == "caller_not_allowed"
    assert search.calls == []


def test_request_with_no_caller_identity_is_refused_fail_closed() -> None:
    handler, _, _ = build()
    assert handler.handle(event(caller=None))["statusCode"] == 403


@pytest.mark.parametrize("user", [None, "", "   "])
def test_missing_or_blank_user_context_gets_401(user: str | None) -> None:
    handler, search, _ = build()
    response = handler.handle(event(user=user))
    assert response["statusCode"] == 401
    assert json.loads(response["body"])["error"]["code"] == "user_context_missing"
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
    assert json.loads(response["body"])["error"]["code"] == "invalid_argument"


def test_error_body_has_code_generic_message_and_request_id() -> None:
    handler, _, _ = build(embedder=FakeEmbedder(RetrievalError("embedding_unavailable")))
    body = json.loads(handler.handle(event())["body"])
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "request_id"}
    assert body["error"]["request_id"] == REQUEST_ID
    assert response_is_json(handler.handle(event()))


def response_is_json(response: dict[str, Any]) -> bool:
    return bool(response["headers"]["content-type"] == "application/json")


def test_unexpected_exception_returns_500_without_its_text_and_logs_the_request_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    handler, _, _ = build(embedder=FakeEmbedder(RuntimeError("secret-table kn_chunks exploded")))
    response = handler.handle(event())
    assert response["statusCode"] == 500
    assert json.loads(response["body"])["error"]["code"] == "internal_error"
    assert "secret-table" not in response["body"]
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert REQUEST_ID in logged


def test_refused_requests_are_still_counted_in_metrics() -> None:
    handler, _, metrics = build()
    handler.handle(event(caller="arn:aws:iam::111122223333:role/other"))
    handler.handle(event(user=None))
    assert [m.error_code for m in metrics.requests] == ["caller_not_allowed", "user_context_missing"]
