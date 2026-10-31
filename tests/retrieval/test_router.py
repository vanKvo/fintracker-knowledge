"""KN-RET-01, -07: caller allow-list, user context and the error body on the internal route.

KN-FAPI-10: these cases were written against the Lambda handler; they run unchanged (same request, status,
error `code` and body) against the FastAPI route.

__author__ = "Van Vo"
"""

from __future__ import annotations

import json
import logging

import pytest

from src.core.exceptions import EmbeddingUnavailableError
from src.retrieval.models import SearchResult
from tests.retrieval.app_support import BODY, build
from tests.retrieval.fakes import (
    CALLER_ARN,
    REQUEST_ID,
    USER_ID,
    FakeEmbedder,
    FakeSearch,
    hit,
)

MARKER = "UNIQUE-QUERY-MARKER-9914"


def test_allow_listed_caller_with_user_context_is_accepted() -> None:
    client, search, _ = build()
    response = client.retrieve()
    assert response.status_code == 200
    body = response.json()
    assert len(body["chunks"]) == 1 and search.calls[0]["user_id"] == USER_ID


def test_the_success_body_has_exactly_the_contract_fields() -> None:
    """KN-RET Appendix B."""
    client, _, _ = build()
    body = client.retrieve().json()
    assert set(body) == {"chunks", "embedding_model", "index_commit"}
    assert set(body["chunks"][0]) == {"chunk_id", "text", "source_path", "heading_path", "score"}


def test_caller_not_on_the_allow_list_gets_403_and_no_work_is_done() -> None:
    client, search, _ = build()
    response = client.retrieve(caller="arn:aws:iam::111122223333:role/other")
    assert response.status_code == 403
    assert response.json()["code"] == "caller_not_allowed"
    assert search.calls == []


def test_request_with_no_caller_identity_is_401_unauthorized_and_no_work_is_done() -> None:
    """LB-IHTTP-05: no verified caller is 401."""
    client, search, _ = build()
    response = client.retrieve(caller=None)
    assert response.status_code == 401 and response.json()["code"] == "unauthorized"
    assert search.calls == []


@pytest.mark.parametrize("user", [None, "", "   "])
def test_missing_or_blank_user_context_gets_401(user: str | None) -> None:
    client, search, _ = build()
    response = client.retrieve(user=user)
    assert response.status_code == 401
    assert response.json()["code"] == "user_context_missing"
    assert search.calls == []


def test_user_id_in_the_body_is_rejected_and_never_used() -> None:
    client, search, _ = build()
    response = client.retrieve({**BODY, "user_id": "user-b"})
    assert response.status_code == 400
    assert search.calls == []


def test_body_that_is_not_json_gets_400_not_422() -> None:
    client, _, _ = build()
    response = client.retrieve("{not json")
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_argument"


def test_empty_body_gets_400() -> None:
    client, _, _ = build()
    response = client.retrieve("")
    assert response.status_code == 400 and response.json()["code"] == "invalid_argument"


def test_a_body_far_larger_than_any_valid_request_is_rejected_unread() -> None:
    client, search, _ = build()
    response = client.retrieve(json.dumps({**BODY, "query": "x" * 100_000}))
    assert response.status_code == 400 and response.json()["code"] == "invalid_argument" and search.calls == []


def test_a_validation_failure_keeps_its_400_and_names_the_field_not_the_value() -> None:
    """KN-RET-02 through the route."""
    client, search, _ = build()
    response = client.retrieve({**BODY, "k": 99})
    assert response.status_code == 400 and response.json()["code"] == "invalid_argument"
    assert response.json()["detail"].endswith("k.") and "99" not in response.text
    assert search.calls == []


def test_error_body_is_rfc_9457_with_code_and_trace_id_but_no_request_id() -> None:
    client, _, _ = build(embedder=FakeEmbedder(EmbeddingUnavailableError()))
    response = client.retrieve()
    body = response.json()
    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/problem+json")
    assert set(body) == {"type", "title", "status", "detail", "code", "retryable", "trace_id"}
    assert (body["code"], body["status"], body["retryable"]) == ("embedding_unavailable", 503, True)
    assert body["type"] == "https://fintracker.app/errors/embedding-unavailable"
    assert len(body["trace_id"]) == 32 and REQUEST_ID not in response.text  # app doc 4.8


def test_trace_id_comes_from_the_incoming_traceparent() -> None:
    client, _, _ = build(embedder=FakeEmbedder(EmbeddingUnavailableError()))
    response = client.retrieve(headers={"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"})
    assert response.json()["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_unexpected_exception_returns_500_without_its_text_and_logs_the_request_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    client, _, _ = build(embedder=FakeEmbedder(RuntimeError("secret-table kn_chunks exploded")))
    response = client.retrieve()
    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    assert "secret-table" not in response.text
    trace_id = response.json()["trace_id"]
    logged = "\n".join(r.getMessage() + str(r.__dict__) for r in caplog.records)
    assert REQUEST_ID in logged and trace_id in logged  # both IDs on the log line
    assert "secret-table" not in logged  # messages of exceptions that aren't ours are redacted


def test_an_exception_outside_the_service_is_still_a_500_problem_without_its_text(caplog: pytest.LogCaptureFixture) -> None:
    """The route's last line of defense: a bug while wiring the request is `internal_error`, never a stack trace."""
    from src.core.dependencies import get_retrieval_service

    client, _, _ = build()

    detail = "secret-" + "wiring-detail"  # built at run time: a traceback quotes source lines, not values

    def broken() -> None:
        raise RuntimeError(detail)

    client.app.dependency_overrides[get_retrieval_service] = broken
    caplog.set_level(logging.DEBUG)

    response = client.retrieve()

    assert response.status_code == 500 and response.json()["code"] == "internal_error"
    assert detail not in response.text
    assert detail not in "\n".join(r.getMessage() + str(r.__dict__) for r in caplog.records)


def test_refused_requests_are_still_counted_in_metrics() -> None:
    """KN-RET-08 / KN-FAPI-10: the metric lines keep their name and fields."""
    client, _, metrics = build()
    client.retrieve(caller="arn:aws:iam::111122223333:role/other")
    client.retrieve(user=None)
    client.retrieve(caller=None)
    assert [m.error_code for m in metrics.requests] == ["caller_not_allowed", "user_context_missing", "unauthorized"]


def test_a_served_request_emits_one_metric_with_the_caller_and_no_error() -> None:
    client, _, metrics = build()
    client.retrieve()
    [metric] = metrics.requests
    assert (metric.caller, metric.scope, metric.k, metric.result_count, metric.error_code) == (
        CALLER_ARN, "system", 5, 1, None,
    )


def test_a_chunk_owned_by_another_user_is_dropped_and_signalled_through_the_route() -> None:
    """KN-RET-05: the mismatch signal is unchanged."""
    foreign = hit(scope="system", user_id="22222222-2222-4222-8222-222222222222")
    client, _, metrics = build(search=FakeSearch(SearchResult([foreign, hit(0.2)], "c1")))
    assert len(client.retrieve().json()["chunks"]) == 1
    assert [chunk_id for chunk_id, _ in metrics.mismatches] == [foreign.chunk_id]


def test_a_unique_query_marker_appears_in_no_log_line(caplog: pytest.LogCaptureFixture) -> None:
    """KN-RET-08 / KN-FAPI-10: no content in logs, for a served request and for each failure."""
    caplog.set_level(logging.DEBUG)
    served, _, _ = build()
    failing, _, _ = build(embedder=FakeEmbedder(EmbeddingUnavailableError()))
    body = {**BODY, "query": MARKER}
    served.retrieve(body)
    failing.retrieve(body)
    served.retrieve({**body, "user_id": "x"})
    logged = "\n".join(r.getMessage() + json.dumps(r.__dict__, default=str) for r in caplog.records)
    assert MARKER not in logged
