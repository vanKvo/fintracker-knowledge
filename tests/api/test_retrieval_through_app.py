"""KN-FAPI-04, -10: the real wiring (container, Bedrock-shaped client, login role, RLS) answers through the route.

Only the edges are fake: Bedrock returns a fixed vector, and Postgres is a Testcontainers pgvector database.

__author__ = "Van Vo"
"""

from __future__ import annotations

import io
import json
import logging
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from testcontainers.postgres import PostgresContainer

from src.app_factory import create_app
from src.core.config import load_knowledge_settings
from src.core.container import Container
from tests.retrieval.app_support import BODY, TEST_ENVIRONMENT, AppClient
from tests.retrieval.fakes import CALLER_ARN, OTHER_USER_ID, USER_ID
from tests.support.db import insert_chunk, insert_document, unit_vector

PASSWORD = "test-only-password"


class FixedVectorBedrock:
    """Answers every embedding call with unit vector 0, so the nearest chunks are those seeded at vector 0."""

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        vector = [1.0] + [0.0] * 1023
        return {"body": io.BytesIO(json.dumps({"embedding": vector}).encode())}


@pytest.fixture
def login_role(admin_engine: Engine) -> Iterator[str]:
    name = f"kn_app_{uuid.uuid4().hex[:8]}"
    with admin_engine.connect() as conn:
        conn.execute(text(f"CREATE ROLE {name} LOGIN PASSWORD '{PASSWORD}' IN ROLE kn_retrieve"))
    yield name
    with admin_engine.connect() as conn:
        conn.execute(text(f"DROP ROLE {name}"))


@pytest.fixture(autouse=True)
def allowed_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERNAL_AUTH_MODE", raising=False)
    monkeypatch.setenv("INTERNAL_ALLOWED_CALLER_ARNS", CALLER_ARN)


def app_for(postgres: PostgresContainer, login_role: str) -> AppClient:
    environment = {
        **TEST_ENVIRONMENT,
        "KNOWLEDGE_DB_HOST": postgres.get_container_host_ip(),
        "KNOWLEDGE_DB_PORT": str(postgres.get_exposed_port(5432)),
        "KNOWLEDGE_DB_RETRIEVE_USER": login_role,
        "KNOWLEDGE_DB_RETRIEVE_PASSWORD": PASSWORD,
    }
    settings = load_knowledge_settings(environment)
    container = Container(settings, bedrock_client_factory=lambda *_: FixedVectorBedrock())  # type: ignore[arg-type,return-value]
    container.verify_database()
    return AppClient(create_app(settings, container=container))


def test_a_user_row_of_user_a_is_never_returned_to_user_b_and_no_mismatch_is_signalled(
    postgres: PostgresContainer, admin_engine: Engine, login_role: str, caplog: pytest.LogCaptureFixture
) -> None:
    system = insert_document(admin_engine, "system", None, "help/statements/csv-mapping.md")
    insert_chunk(admin_engine, system, "system", None, unit_vector(1), chunk_text="Confirm the CSV mapping.")
    private = insert_document(admin_engine, "user", USER_ID, "user/a.md")
    insert_chunk(admin_engine, private, "user", USER_ID, unit_vector(0), chunk_text="USER A PRIVATE TEXT")
    client = app_for(postgres, login_role)
    caplog.set_level(logging.DEBUG)

    response = client.retrieve(BODY, user=OTHER_USER_ID)

    assert response.status_code == 200
    assert [c["text"] for c in response.json()["chunks"]] == ["Confirm the CSV mapping."]
    assert "USER A PRIVATE TEXT" not in response.text
    assert "ownership_mismatch" not in "\n".join(r.getMessage() for r in caplog.records)


def test_the_route_returns_ranked_system_chunks_with_the_index_commit(
    postgres: PostgresContainer, admin_engine: Engine, login_role: str
) -> None:
    document = insert_document(admin_engine, "system", None, "help/statements/csv-mapping.md")
    insert_chunk(admin_engine, document, "system", None, unit_vector(0), chunk_text="Confirm the mapping.")

    response = app_for(postgres, login_role).retrieve(BODY)

    body = response.json()
    assert response.status_code == 200 and body["chunks"][0]["source_path"] == "help/statements/csv-mapping.md"
    assert body["chunks"][0]["score"] == 1.0 and body["embedding_model"] == "amazon.titan-embed-text-v2:0"


def test_a_stopped_database_through_the_route_is_a_503_database_unavailable() -> None:
    environment = {**TEST_ENVIRONMENT, "KNOWLEDGE_DB_HOST": "127.0.0.1", "KNOWLEDGE_DB_PORT": "1"}
    settings = load_knowledge_settings(environment)
    # sleep: the one retry of KN-RET-04, without the one-second wait
    container = Container(
        settings, bedrock_client_factory=lambda *_: FixedVectorBedrock(), sleep=lambda _s: None  # type: ignore[arg-type,return-value]
    )

    response = AppClient(create_app(settings, container=container)).retrieve(BODY)

    assert response.status_code == 503 and response.json()["code"] == "database_unavailable"
    assert "127.0.0.1" not in response.text
