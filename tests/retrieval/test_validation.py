"""KN-RET-02: argument validation.

__author__ = "Van Vo"
"""

from __future__ import annotations

from typing import Any

import pytest

from src.core.exceptions import AppError
from src.retrieval.schemas import parse_request

VALID: dict[str, Any] = {"query": "How do I confirm a CSV mapping?", "scope": "system", "k": 5}


def rejected(body: Any) -> AppError:
    with pytest.raises(AppError) as caught:
        parse_request(body)
    return caught.value


def test_valid_request_passes() -> None:
    request = parse_request(VALID)
    assert (request.query, request.scope, request.k) == (VALID["query"], "system", 5)


def test_missing_k_defaults_to_five() -> None:
    assert parse_request({"query": "q", "scope": "system"}).k == 5


@pytest.mark.parametrize("k", [0, 11, -1])
def test_k_out_of_range_is_invalid_and_names_the_field(k: int) -> None:
    error = rejected({**VALID, "k": k})
    assert (error.code, error.status) == ("invalid_argument", 400)
    assert "k" in error.user_message.split()[-1]


@pytest.mark.parametrize("k", ["5", 5.0, True, None])
def test_non_integer_k_is_invalid(k: Any) -> None:
    assert rejected({**VALID, "k": k}).code == "invalid_argument"


@pytest.mark.parametrize("query", ["", "   \n", None, 5])
def test_blank_or_non_string_query_is_invalid(query: Any) -> None:
    error = rejected({**VALID, "query": query})
    assert error.code == "invalid_argument"
    assert error.user_message.endswith("query.")


def test_query_over_1000_characters_is_invalid_and_1000_passes() -> None:
    assert rejected({**VALID, "query": "a" * 1001}).code == "invalid_argument"
    assert parse_request({**VALID, "query": "a" * 1000}).query == "a" * 1000


def test_extra_field_is_rejected_including_a_user_id() -> None:
    error = rejected({**VALID, "user_id": "user-b"})
    assert error.code == "invalid_argument"
    assert "user-b" not in error.user_message


def test_missing_query_or_scope_is_invalid() -> None:
    assert rejected({"scope": "system"}).code == "invalid_argument"
    assert rejected({"query": "q"}).code == "invalid_argument"


def test_scope_mine_is_not_enabled_yet() -> None:
    error = rejected({**VALID, "scope": "mine"})
    assert (error.code, error.status) == ("scope_not_enabled", 400)


def test_unknown_scope_is_invalid() -> None:
    assert rejected({**VALID, "scope": "all"}).code == "invalid_argument"


def test_body_that_is_not_an_object_is_invalid() -> None:
    assert rejected(["query"]).code == "invalid_argument"
    assert rejected(None).code == "invalid_argument"


def test_error_message_never_echoes_the_input() -> None:
    assert "SECRET-MARKER" not in rejected({**VALID, "k": "SECRET-MARKER"}).user_message
