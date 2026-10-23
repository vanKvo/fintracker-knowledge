"""KN-VS-02: database roles and grants.

__author__ = "Van Vo"
"""

from __future__ import annotations

import psycopg
import psycopg.errors
import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError
from testcontainers.postgres import PostgresContainer

from tests.support.db import insert_chunk, insert_document, transaction_as, unit_vector

# Appendix B
EXPECTED_GRANTS = {
    "kn_ingest": {"SELECT", "INSERT", "UPDATE", "DELETE"},
    "kn_retrieve": {"SELECT"},
    "kn_delete": {"SELECT", "DELETE"},
}
TABLES = ("kn_documents", "kn_chunks")


@pytest.mark.parametrize("role", sorted(EXPECTED_GRANTS))
def test_each_workflow_role_exists_and_may_connect(admin_engine: Engine, role: str) -> None:
    with admin_engine.connect() as conn:
        connect = conn.execute(
            text("SELECT has_database_privilege(:role, current_database(), 'CONNECT')"), {"role": role}
        ).scalar_one()
    assert connect is True


@pytest.mark.parametrize("role", sorted(EXPECTED_GRANTS))
@pytest.mark.parametrize("table", TABLES)
def test_role_holds_exactly_the_appendix_b_privileges(admin_engine: Engine, role: str, table: str) -> None:
    with admin_engine.connect() as conn:
        granted = {
            privilege
            for privilege in ("SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER")
            if conn.execute(
                text("SELECT has_table_privilege(:role, :table, :privilege)"),
                {"role": role, "table": table, "privilege": privilege},
            ).scalar_one()
        }
    assert granted == EXPECTED_GRANTS[role]


def test_retrieve_role_can_select_but_not_write(app_engine: Engine, admin_engine: Engine) -> None:
    insert_document(admin_engine, "system", None, "help/a.md")
    with transaction_as(app_engine, "kn_retrieve") as conn:
        assert conn.execute(text("SELECT count(*) FROM kn_documents")).scalar_one() == 1
    statements = [
        "INSERT INTO kn_documents (document_id, scope, source_path, content_hash, embedding_model, "
        "chunk_count, indexed_at) VALUES (gen_random_uuid(), 'system', 'x', repeat('a', 64), 'm', 0, now())",
        "UPDATE kn_documents SET chunk_count = 9",
        "DELETE FROM kn_documents",
    ]
    for statement in statements:
        with pytest.raises(DBAPIError) as exc:
            with transaction_as(app_engine, "kn_retrieve") as conn:
                conn.execute(text(statement))
        assert isinstance(exc.value.orig, psycopg.errors.InsufficientPrivilege)


def test_ingest_role_can_write_system_documents(app_engine: Engine, admin_engine: Engine) -> None:
    with transaction_as(app_engine, "kn_ingest") as conn:
        conn.execute(
            text(
                "INSERT INTO kn_documents (document_id, scope, source_path, content_hash, embedding_model, "
                "chunk_count, indexed_at) VALUES (gen_random_uuid(), 'system', 'help/a.md', repeat('a', 64), 'm', 0, now())"
            )
        )
        conn.execute(text("UPDATE kn_documents SET chunk_count = 2"))
    with transaction_as(app_engine, "kn_ingest") as conn:
        deleted = conn.execute(text("DELETE FROM kn_documents")).rowcount
    assert deleted == 1


def test_delete_role_cannot_delete_system_rows(app_engine: Engine, admin_engine: Engine) -> None:
    doc = insert_document(admin_engine, "system", None, "help/a.md")
    insert_chunk(admin_engine, doc, "system", None, unit_vector(0))
    with transaction_as(app_engine, "kn_delete", "user-a") as conn:
        assert conn.execute(text("DELETE FROM kn_chunks")).rowcount == 0
        assert conn.execute(text("DELETE FROM kn_documents")).rowcount == 0
    with admin_engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM kn_documents")).scalar_one() == 1


def test_delete_role_deletes_only_the_current_users_mine_rows(app_engine: Engine, admin_engine: Engine) -> None:
    doc_a = insert_document(admin_engine, "mine", "user-a", "doc-1")
    doc_b = insert_document(admin_engine, "mine", "user-b", "doc-1")
    insert_chunk(admin_engine, doc_a, "mine", "user-a", unit_vector(0))
    insert_chunk(admin_engine, doc_b, "mine", "user-b", unit_vector(1))
    with transaction_as(app_engine, "kn_delete", "user-a") as conn:
        assert conn.execute(text("DELETE FROM kn_documents")).rowcount == 1
    with admin_engine.connect() as conn:
        owners = [r[0] for r in conn.execute(text("SELECT user_id FROM kn_documents"))]
        chunk_owners = [r[0] for r in conn.execute(text("SELECT user_id FROM kn_chunks"))]
    assert owners == ["user-b"]
    assert chunk_owners == ["user-b"]


def test_delete_role_without_user_context_deletes_nothing(app_engine: Engine, admin_engine: Engine) -> None:
    insert_document(admin_engine, "mine", "user-a", "doc-1")
    with transaction_as(app_engine, "kn_delete") as conn:
        assert conn.execute(text("DELETE FROM kn_documents")).rowcount == 0


def test_other_services_roles_cannot_connect_or_read(
    postgres: PostgresContainer, admin_engine: Engine
) -> None:
    with admin_engine.connect() as conn:
        for role in ("ledger_app", "analytics_ro"):
            conn.execute(text(f"CREATE ROLE {role} LOGIN PASSWORD 'pw'"))
    try:
        for role in ("ledger_app", "analytics_ro"):
            with pytest.raises(psycopg.OperationalError, match="permission denied for database"):
                psycopg.connect(
                    host=postgres.get_container_host_ip(),
                    port=postgres.get_exposed_port(5432),
                    dbname="knowledge",
                    user=role,
                    password="pw",
                )
        with admin_engine.connect() as conn:
            for table in TABLES:
                readable = conn.execute(
                    text("SELECT has_table_privilege('ledger_app', :table, 'SELECT')"), {"table": table}
                ).scalar_one()
                assert readable is False
    finally:
        with admin_engine.connect() as conn:
            conn.execute(text("DROP ROLE ledger_app"))
            conn.execute(text("DROP ROLE analytics_ro"))
