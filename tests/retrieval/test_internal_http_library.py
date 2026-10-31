"""LB-IHTTP-09 / KN-FAPI-02: Knowledge's internal route uses fintracker-internal-http for the caller check and the
user context, and keeps no copy of either. The shared Appendix A rows run through the real route.

__author__ = "Van Vo"
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fintracker_internal_http.testing import api_gateway_event, load_vectors, run_conformance

from src.app_factory import create_app
from tests.retrieval.app_support import BODY, app_settings, build
from tests.retrieval.fakes import USER_ID

SRC = Path(__file__).resolve().parents[2] / "src"
ROWS = load_vectors()["caller_matrix"]
PASSING_ROWS = [row["id"] for row in ROWS if row["status"] == 200]


@pytest.mark.parametrize("row", ROWS, ids=[row["id"] for row in ROWS])
def test_the_route_answers_every_row_of_the_shared_caller_matrix(row: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    """LB-IHTTP-09 / LB-IHTTP-08: the same rows every service must answer the same way (Appendix A)."""
    for name in ("AWS_EXECUTION_ENV", "AWS_LAMBDA_FUNCTION_NAME", "ECS_CONTAINER_METADATA_URI_V4"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("INTERNAL_AUTH_MODE", str(row["mode"]))
    allowed = row["allowed_caller_arns"]
    if allowed is None:
        monkeypatch.delenv("INTERNAL_ALLOWED_CALLER_ARNS", raising=False)
    else:
        monkeypatch.setenv("INTERNAL_ALLOWED_CALLER_ARNS", ",".join(allowed))  # type: ignore[arg-type]
    headers = dict(row["headers"])  # type: ignore[call-overload]
    if row["header_caller"] is not None:
        headers["X-Internal-Caller-Arn"] = row["header_caller"]  # type: ignore[assignment]
    client, _, _ = build()

    response = client.retrieve(
        BODY,
        caller=row["verified_caller"] if row["mode"] == "iam" else None,  # type: ignore[arg-type]
        user=None,
        headers=headers,
    )

    assert response.status_code == row["status"]
    if row["code"] is not None:
        assert response.json()["code"] == row["code"]


def test_the_library_conformance_suite_passes_against_the_real_route() -> None:
    """KN-FAPI-02: `run_conformance` sends no body, so a caller that passes the checks reaches body parsing and
    gets the route's own 400 instead of the matrix's 200. Every rejection row keeps the matrix's answer."""
    own_answers = {row_id: (400, "invalid_argument") for row_id in PASSING_ROWS}

    run_conformance(
        lambda: create_app(app_settings()),
        path="/internal/retrieve",
        method="POST",
        own_answers=own_answers,
    )


def test_a_forwarded_user_id_that_is_not_a_uuid_is_400_and_never_used() -> None:
    """XS-CTX-04: the internal user ID is a UUID; the bad value is not echoed and no search runs."""
    client, search, _ = build()

    response = client.retrieve(user="user-a")

    assert response.status_code == 400 and response.json()["code"] == "invalid_argument"
    assert "user-a" not in response.text and search.calls == []


def test_the_caller_is_decided_before_the_body_is_read() -> None:
    """KN-FAPI-02: an unlisted caller with a broken body is a 403, not a 400."""
    client, search, _ = build()

    response = client.retrieve("{not json", caller="arn:aws:iam::111122223333:role/other")

    assert response.status_code == 403 and search.calls == []


def test_the_service_keeps_no_copy_of_signing_or_caller_handling() -> None:
    """LB-IHTTP-09: a repository check; only the library may sign requests or read the caller header."""
    banned = re.compile(r"botocore\.auth|SigV4Auth|AWSRequest|x-internal-caller-arn", re.IGNORECASE)
    offenders = [str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if banned.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_the_route_depends_on_the_shared_caller_check_and_user_context() -> None:
    """KN-FAPI-02: the dependency chain is the library's, with no raw-event adapter left."""
    source = (SRC / "core" / "internal_auth.py").read_text(encoding="utf-8")
    assert "require_internal_caller" in source and "get_internal_user" in source
    assert "caller_rejection" not in source and "requestContext" not in source
