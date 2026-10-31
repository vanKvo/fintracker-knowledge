"""KN-FAPI-04: one connection factory per database role, in `postgres` mode (login role, RLS enforced) and
`data_api` mode (fake client, no AWS call), behind the one transaction-per-call interface.

__author__ = "Van Vo"
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from botocore.exceptions import ClientError
from sqlalchemy import text
from sqlalchemy.engine import Engine
from testcontainers.postgres import PostgresContainer

from src.core.exceptions import DatabaseUnavailableError
from src.retrieval.repository import PostgresChunkSearch
from src.vector_store.connections import (
    DataApiConnections,
    PostgresConnections,
    UnsafeDatabaseRoleError,
    postgres_engine,
)
from tests.support.db import EMBEDDING_MODEL, insert_chunk, insert_document, unit_vector

USER_A = "11111111-1111-4111-8111-111111111111"
USER_B = "22222222-2222-4222-8222-222222222222"
PASSWORD = "test-only-password"
QUERY = tuple(float(x) for x in unit_vector(0).strip("[]").split(","))


# --- postgres mode ---


@pytest.fixture
def login_role(admin_engine: Engine) -> Iterator[str]:
    """A login role that is a member of kn_retrieve, like kn_app_retrieve in the local stack."""
    name = f"kn_app_{uuid.uuid4().hex[:8]}"
    with admin_engine.connect() as conn:
        conn.execute(text(f"CREATE ROLE {name} LOGIN PASSWORD '{PASSWORD}' IN ROLE kn_retrieve"))
    yield name
    with admin_engine.connect() as conn:
        conn.execute(text(f"DROP ROLE {name}"))


def engine_for(postgres: PostgresContainer, user: str, password: str = PASSWORD) -> Engine:
    return postgres_engine(
        postgres.get_container_host_ip(), int(postgres.get_exposed_port(5432)), "knowledge", user, password
    )


def test_a_search_as_the_login_role_sets_the_user_and_returns_only_visible_rows(
    postgres: PostgresContainer, admin_engine: Engine, login_role: str
) -> None:
    """KN-VS-03, -04 through a real login role: the user context rides in the query's own transaction."""
    system = insert_document(admin_engine, "system", None, "help/a.md")
    insert_chunk(admin_engine, system, "system", None, unit_vector(0), chunk_text="system chunk")
    owned = insert_document(admin_engine, "user", USER_A, "user/a.md")
    insert_chunk(admin_engine, owned, "user", USER_A, unit_vector(0), chunk_text="private chunk")
    connections = PostgresConnections(engine_for(postgres, login_role))
    search = PostgresChunkSearch(connections.connection)

    system_hits = search.search(USER_B, QUERY, "system", EMBEDDING_MODEL, 10).hits
    own_hits = search.search(USER_A, QUERY, "user", EMBEDDING_MODEL, 10).hits
    other_hits = search.search(USER_B, QUERY, "user", EMBEDDING_MODEL, 10).hits

    assert [h.text for h in system_hits] == ["system chunk"]
    assert [h.text for h in own_hits] == ["private chunk"]
    assert other_hits == []  # RLS holds for the login role: user B never sees user A's row


def test_a_superuser_connection_is_refused_at_startup_naming_the_role_and_never_the_password(
    postgres: PostgresContainer, admin_engine: Engine
) -> None:
    connections = PostgresConnections(engine_for(postgres, "test", "test"))
    with pytest.raises(UnsafeDatabaseRoleError) as caught:
        connections.assert_role_is_safe()
    assert "test" in str(caught.value) and "superuser" in str(caught.value)
    assert PASSWORD not in str(caught.value)


def test_a_bypassrls_role_is_refused_at_startup_naming_the_role(
    postgres: PostgresContainer, admin_engine: Engine
) -> None:
    name = f"kn_bypass_{uuid.uuid4().hex[:8]}"
    with admin_engine.connect() as conn:
        conn.execute(text(f"CREATE ROLE {name} LOGIN PASSWORD '{PASSWORD}' BYPASSRLS IN ROLE kn_retrieve"))
    try:
        with pytest.raises(UnsafeDatabaseRoleError, match=f"{name}.*BYPASSRLS"):
            PostgresConnections(engine_for(postgres, name)).assert_role_is_safe()
    finally:
        with admin_engine.connect() as conn:
            conn.execute(text(f"DROP ROLE {name}"))


def test_a_safe_login_role_passes_the_startup_check(postgres: PostgresContainer, login_role: str) -> None:
    PostgresConnections(engine_for(postgres, login_role)).assert_role_is_safe()


def test_a_stopped_database_is_database_unavailable_without_the_driver_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """KN-RET-04: the driver's text (host, user, port) is neither returned nor logged."""
    engine = postgres_engine("127.0.0.1", 1, "knowledge", "secret-user-name", PASSWORD)
    search = PostgresChunkSearch(PostgresConnections(engine).connection)
    caplog.set_level("DEBUG")

    with pytest.raises(DatabaseUnavailableError) as caught:
        search.search(USER_A, QUERY, "system", EMBEDDING_MODEL, 10)

    assert "secret-user-name" not in str(caught.value) and "127.0.0.1" not in str(caught.value)
    assert "secret-user-name" not in "\n".join(r.getMessage() + str(r.__dict__) for r in caplog.records)


def test_the_ping_answers_for_a_reachable_database_and_fails_as_database_unavailable_otherwise(
    postgres: PostgresContainer, login_role: str
) -> None:
    PostgresConnections(engine_for(postgres, login_role)).ping()
    with pytest.raises(DatabaseUnavailableError):
        PostgresConnections(postgres_engine("127.0.0.1", 1, "knowledge", "x", "y")).ping()


# --- data_api mode ---


class FakeRdsData:
    """The four rds-data calls a transaction uses. SQL decides the answer; every call is recorded."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail_on = fail_on

    def begin_transaction(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("begin", kwargs))
        return {"transactionId": "tx-1"}

    def execute_statement(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("execute", kwargs))
        sql: str = kwargs["sql"]
        if self.fail_on and self.fail_on in sql:
            raise ClientError({"Error": {"Code": "BadRequestException", "Message": "secret-sql-detail"}}, "ExecuteStatement")
        if "FROM kn_chunks" in sql:
            return {"records": [[
                {"stringValue": "c1"}, {"stringValue": "d1"}, {"stringValue": "system"}, {"isNull": True},
                {"stringValue": "help/a.md"}, {"stringValue": "A > B"}, {"stringValue": "chunk text"},
                {"doubleValue": 0.25},
            ]]}
        if "source_commit" in sql:
            return {"records": [[{"stringValue": "9f2c1ab"}]]}
        return {"records": []}

    def commit_transaction(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("commit", kwargs))
        return {}

    def rollback_transaction(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("rollback", kwargs))
        return {}

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.calls]


def data_api(client: FakeRdsData) -> DataApiConnections:
    return DataApiConnections(client, cluster_arn="arn:aws:rds:cluster", secret_arn="arn:aws:secret:retrieve", database="knowledge")


def test_data_api_search_runs_context_settings_and_query_in_one_transaction_and_commits() -> None:
    client = FakeRdsData()

    result = PostgresChunkSearch(data_api(client).connection).search(USER_A, QUERY, "system", EMBEDDING_MODEL, 50)

    assert [h.chunk_id for h in result.hits] == ["c1"] and result.hits[0].distance == 0.25
    assert result.hits[0].user_id is None and result.index_commit == "9f2c1ab"
    executes = [kw for kind, kw in client.calls if kind == "execute"]
    assert {kw["transactionId"] for kw in executes} == {"tx-1"}
    assert all(kw["resourceArn"] == "arn:aws:rds:cluster" and kw["secretArn"] == "arn:aws:secret:retrieve" for kw in executes)
    assert "app.current_user_id" in executes[0]["sql"]
    assert executes[0]["parameters"] == [{"name": "uid", "value": {"stringValue": USER_A}}]
    assert any("hnsw.ef_search" in kw["sql"] for kw in executes)
    assert client.kinds() == ["begin", *["execute"] * len(executes), "commit"]


def test_data_api_failure_rolls_back_and_surfaces_database_unavailable_without_the_aws_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = FakeRdsData(fail_on="FROM kn_chunks")
    caplog.set_level("DEBUG")

    with pytest.raises(DatabaseUnavailableError) as caught:
        PostgresChunkSearch(data_api(client).connection).search(USER_A, QUERY, "system", EMBEDDING_MODEL, 50)

    assert "secret-sql-detail" not in str(caught.value) and "arn:aws" not in str(caught.value)
    assert "secret-sql-detail" not in "\n".join(r.getMessage() + str(r.__dict__) for r in caplog.records)
    assert client.kinds()[-1] == "rollback" and "commit" not in client.kinds()


def test_data_api_ping_runs_a_trivial_query() -> None:
    client = FakeRdsData()
    data_api(client).ping()
    assert any(kw["sql"].strip().upper().startswith("SELECT 1") for kind, kw in client.calls if kind == "execute")
