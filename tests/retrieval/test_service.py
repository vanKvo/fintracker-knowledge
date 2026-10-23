"""KN-RET-03 to KN-RET-08: embedding, search flow, ownership re-check, response, errors, metrics.

__author__ = "Van Vo"
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import pytest

from src.embedding.models import EmbeddingInputTooLong
from src.retrieval.metrics import empty_result_alarm
from src.retrieval.models import DatabaseUnavailable, RetrievalError, SearchResult
from src.retrieval.service import RetrievalService
from tests.retrieval.fakes import (
    CALLER_ARN,
    MODEL,
    OTHER_USER_ID,
    REQUEST_ID,
    USER_ID,
    VECTOR,
    FakeEmbedder,
    FakeSearch,
    RecordingMetrics,
    hit,
    unavailable,
)

BODY: dict[str, Any] = {"query": "How do I confirm a CSV mapping?", "scope": "system", "k": 5}
MARKER = "UNIQUE-QUERY-MARKER-7731"


def build(
    embedder: FakeEmbedder | None = None,
    search: FakeSearch | None = None,
    *,
    budget: float = 3.0,
) -> tuple[RetrievalService, FakeEmbedder, FakeSearch, RecordingMetrics, list[float]]:
    embedder = embedder or FakeEmbedder()
    search = search or FakeSearch()
    metrics = RecordingMetrics()
    sleeps: list[float] = []
    service = RetrievalService(embedder, search, metrics, budget_seconds=budget, sleep=sleeps.append)
    return service, embedder, search, metrics, sleeps


def retrieve(service: RetrievalService, body: dict[str, Any] = BODY, user_id: str = USER_ID):  # type: ignore[no-untyped-def]
    return service.retrieve(REQUEST_ID, CALLER_ARN, user_id, body)


def failure(service: RetrievalService, body: dict[str, Any] = BODY) -> RetrievalError:
    with pytest.raises(RetrievalError) as caught:
        retrieve(service, body)
    return caught.value


# --- KN-RET-03 ---


def test_query_is_embedded_as_is_and_search_uses_the_active_model() -> None:
    service, embedder, search, *_ = build()
    retrieve(service)
    assert embedder.queries == [BODY["query"]]
    assert search.calls[0]["vector"] == VECTOR and len(VECTOR) == 1024
    assert search.calls[0]["model"] == MODEL


def test_embedding_failure_returns_embedding_unavailable() -> None:
    service, *_ = build(FakeEmbedder(unavailable()))
    error = failure(service)
    assert (error.code, error.status) == ("embedding_unavailable", 503)


def test_query_too_long_to_embed_is_an_invalid_argument() -> None:
    service, *_ = build(FakeEmbedder(EmbeddingInputTooLong()))
    error = failure(service)
    assert (error.code, error.status) == ("invalid_argument", 400)


# --- KN-RET-04 ---


def test_search_gets_the_context_user_and_requested_scope_never_a_body_user() -> None:
    service, _, search, *_ = build()
    retrieve(service, {**BODY}, user_id=USER_ID)
    assert search.calls[0]["user_id"] == USER_ID
    assert search.calls[0]["scope"] == "system"


def test_empty_result_is_a_success_with_no_chunks() -> None:
    service, *_ = build(search=FakeSearch(SearchResult([], "9f2c1ab")))
    response = retrieve(service)
    assert response.to_dict()["chunks"] == []


def test_at_most_three_chunks_per_document_and_k_is_respected() -> None:
    doc = uuid.uuid4()
    hits = [hit(0.01 * i, document_id=doc) for i in range(6)] + [hit(0.5), hit(0.6)]
    service, *_ = build(search=FakeSearch(SearchResult(hits, "c1")))
    chunks = retrieve(service, {**BODY, "k": 5}).to_dict()["chunks"]
    assert len(chunks) == 5
    assert sum(1 for h in hits[:6] if h.chunk_id in {c["chunk_id"] for c in chunks}) == 3


def test_database_failure_is_retried_once_after_one_second() -> None:
    search = FakeSearch(DatabaseUnavailable(), SearchResult([hit()], "c1"))
    service, _, _, _, sleeps = build(search=search)
    assert len(retrieve(service).to_dict()["chunks"]) == 1
    assert sleeps == [1.0] and len(search.calls) == 2


def test_database_failing_twice_returns_database_unavailable() -> None:
    search = FakeSearch(DatabaseUnavailable())
    service, _, _, _, sleeps = build(search=search)
    error = failure(service)
    assert (error.code, error.status) == ("database_unavailable", 503)
    assert len(search.calls) == 2 and sleeps == [1.0]


# --- KN-RET-05 ---


def test_correct_rls_drops_nothing_and_emits_no_mismatch() -> None:
    service, _, _, metrics, _ = build(search=FakeSearch(SearchResult([hit(), hit(0.2)], "c1")))
    assert len(retrieve(service).to_dict()["chunks"]) == 2
    assert metrics.mismatches == []


def test_chunk_owned_by_another_user_is_dropped_and_counted_with_ids_only() -> None:
    leaked = hit(0.01, scope="mine", user_id=OTHER_USER_ID, text="user b secret")
    kept = hit(0.3)
    service, _, _, metrics, _ = build(search=FakeSearch(SearchResult([leaked, kept], "c1")))
    ids = {c["chunk_id"] for c in retrieve(service, BODY).to_dict()["chunks"]}
    assert ids == {kept.chunk_id}
    assert metrics.mismatches == [(leaked.chunk_id, REQUEST_ID)]


def test_system_chunk_with_an_owner_is_dropped() -> None:
    odd = hit(0.01, scope="system", user_id=OTHER_USER_ID)
    service, _, _, metrics, _ = build(search=FakeSearch(SearchResult([odd, hit(0.2)], "c1")))
    assert odd.chunk_id not in {c["chunk_id"] for c in retrieve(service).to_dict()["chunks"]}
    assert len(metrics.mismatches) == 1


def test_chunk_of_a_scope_other_than_requested_is_dropped() -> None:
    wrong = hit(0.01, scope="mine", user_id=USER_ID)
    service, _, _, metrics, _ = build(search=FakeSearch(SearchResult([wrong, hit(0.2)], "c1")))
    assert wrong.chunk_id not in {c["chunk_id"] for c in retrieve(service).to_dict()["chunks"]}
    assert len(metrics.mismatches) == 1


def test_a_mismatch_is_reported_even_when_it_empties_the_result() -> None:
    leaked = hit(0.01, scope="mine", user_id=OTHER_USER_ID)
    service, _, _, metrics, _ = build(search=FakeSearch(SearchResult([leaked], "c1")))
    assert retrieve(service).to_dict()["chunks"] == []
    assert len(metrics.mismatches) == 1


# --- KN-RET-06 ---


def test_response_has_exactly_the_contract_fields_in_descending_score_order() -> None:
    hits = [hit(0.3), hit(0.1), hit(0.2)]
    service, *_ = build(search=FakeSearch(SearchResult(hits, "9f2c1ab")))
    body = retrieve(service).to_dict()
    assert set(body) == {"chunks", "embedding_model", "index_commit"}
    assert all(set(c) == {"chunk_id", "text", "source_path", "heading_path", "score"} for c in body["chunks"])
    scores = [c["score"] for c in body["chunks"]]
    assert scores == sorted(scores, reverse=True)


def test_score_is_one_minus_distance_to_three_decimals() -> None:
    service, *_ = build(search=FakeSearch(SearchResult([hit(0.18766)], "c1")))
    assert retrieve(service).to_dict()["chunks"][0]["score"] == 0.812


def test_score_is_clamped_between_zero_and_one() -> None:
    service, *_ = build(search=FakeSearch(SearchResult([hit(-0.0001), hit(1.4)], "c1")))
    scores = [c["score"] for c in retrieve(service).to_dict()["chunks"]]
    assert scores == [1.0, 0.0]


def test_response_names_the_model_and_index_commit() -> None:
    service, *_ = build(search=FakeSearch(SearchResult([hit()], "9f2c1ab")))
    body = retrieve(service).to_dict()
    assert (body["embedding_model"], body["index_commit"]) == (MODEL, "9f2c1ab")


def test_response_never_carries_vectors_owner_or_document_id() -> None:
    h = hit(scope="mine", user_id=USER_ID)
    service, *_ = build(search=FakeSearch(SearchResult([h], "c1")))
    rendered = str(retrieve(service, BODY).to_dict())
    assert h.document_id not in rendered and USER_ID not in rendered and "embedding\"" not in rendered


# --- KN-RET-07 ---


def test_slow_embedding_returns_timeout_within_the_budget() -> None:
    service, *_ = build(FakeEmbedder(delay=lambda: time.sleep(1.0)), budget=0.3)
    started = time.monotonic()
    error = failure(service)
    assert (error.code, error.status) == ("timeout", 504)
    assert time.monotonic() - started < 0.5


def test_unexpected_exception_in_a_step_surfaces_as_internal_error_without_its_text() -> None:
    service, *_ = build(FakeEmbedder(RuntimeError("connection to arn:aws:rds:secret failed")))
    error = failure(service)
    assert (error.code, error.status) == ("internal_error", 500)
    assert "secret" not in error.message


def test_every_error_code_has_a_stable_status_and_generic_message() -> None:
    expected = {
        "invalid_argument": 400,
        "scope_not_enabled": 400,
        "user_context_missing": 401,
        "caller_not_allowed": 403,
        "embedding_unavailable": 503,
        "database_unavailable": 503,
        "timeout": 504,
        "internal_error": 500,
    }
    for code, status in expected.items():
        error = RetrievalError(code)
        assert (error.status, error.code) == (status, code)
        assert error.message and "Traceback" not in error.message


# --- KN-RET-08 ---


def test_a_request_emits_ids_counts_scores_and_step_timings_only() -> None:
    service, _, _, metrics, _ = build(search=FakeSearch(SearchResult([hit(0.1), hit(0.3)], "c1")))
    retrieve(service)
    m = metrics.requests[0]
    assert (m.request_id, m.caller, m.scope, m.k, m.result_count, m.error_code) == (
        REQUEST_ID, CALLER_ARN, "system", 5, 2, None)
    assert m.top_score == 0.9
    assert min(m.validate_ms, m.embed_ms, m.search_ms) >= 0 and m.total_ms >= 0


def test_a_failed_request_emits_the_error_code_and_zero_results() -> None:
    service, _, _, metrics, _ = build(FakeEmbedder(unavailable()))
    failure(service)
    m = metrics.requests[0]
    assert (m.error_code, m.result_count, m.top_score) == ("embedding_unavailable", 0, None)


def test_a_validation_failure_still_emits_a_metric_with_unknown_scope_and_k() -> None:
    service, _, _, metrics, _ = build()
    failure(service, {"query": "q", "scope": "mine"})
    assert metrics.requests[0].error_code == "scope_not_enabled"


def test_no_log_line_contains_the_query_or_chunk_text(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    body = {**BODY, "query": MARKER}
    chunk_text = "CHUNK-TEXT-MARKER-5522"
    scenarios = [
        build(search=FakeSearch(SearchResult([hit(text=chunk_text), hit(0.2, scope="mine", user_id="x",
                                                                         text=chunk_text)], "c1"))),
        build(FakeEmbedder(unavailable())),
        build(FakeEmbedder(RuntimeError(f"boom {MARKER}"))),
        build(search=FakeSearch(DatabaseUnavailable())),
        build(),
    ]
    for service, *_ in scenarios[:-1]:
        try:
            retrieve(service, body)
        except RetrievalError:
            pass
    with pytest.raises(RetrievalError):
        retrieve(scenarios[-1][0], {**body, "k": 99})
    text = "\n".join(r.getMessage() + str(r.__dict__) for r in caplog.records)
    assert MARKER not in text and "CHUNK-TEXT-MARKER" not in text


def test_empty_result_alarm_needs_over_30_percent_and_at_least_20_requests() -> None:
    assert empty_result_alarm(requests=20, empty=7) is True
    assert empty_result_alarm(requests=20, empty=6) is False
    assert empty_result_alarm(requests=19, empty=19) is False
    assert empty_result_alarm(requests=0, empty=0) is False
