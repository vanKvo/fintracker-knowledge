"""KN-VS-03: Row-Level Security by scope and owner.

__author__ = "Van Vo"
"""

from __future__ import annotations

import psycopg.errors
import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError

from src.vector_store.session import SET_USER_CONTEXT_SQL
from tests.support.db import insert_chunk, insert_document, transaction_as, unit_vector


@pytest.fixture
def seeded(admin_engine: Engine) -> None:
    system = insert_document(admin_engine, "system", None, "help/a.md")
    doc_a = insert_document(admin_engine, "mine", "user-a", "doc-1")
    doc_b = insert_document(admin_engine, "mine", "user-b", "doc-1")
    insert_chunk(admin_engine, system, "system", None, unit_vector(0))
    insert_chunk(admin_engine, doc_a, "mine", "user-a", unit_vector(1))
    insert_chunk(admin_engine, doc_b, "mine", "user-b", unit_vector(2))


@pytest.mark.parametrize("table", ["kn_documents", "kn_chunks"])
def test_rls_is_enabled_and_forced_on_both_tables(admin_engine: Engine, table: str) -> None:
    with admin_engine.connect() as conn:
        enabled, forced = conn.execute(
            text("SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = :t"), {"t": table}
        ).one()
    assert (enabled, forced) == (True, True)


@pytest.mark.parametrize("table", ["kn_documents", "kn_chunks"])
def test_user_context_shows_system_rows_and_own_rows_only(
    app_engine: Engine, seeded: None, table: str
) -> None:
    with transaction_as(app_engine, "kn_retrieve", "user-a") as conn:
        rows = conn.execute(text(f"SELECT scope, user_id FROM {table} ORDER BY scope")).all()
    assert sorted(rows) == [("mine", "user-a"), ("system", None)]


@pytest.mark.parametrize("table", ["kn_documents", "kn_chunks"])
def test_no_user_context_shows_system_rows_only(app_engine: Engine, seeded: None, table: str) -> None:
    with transaction_as(app_engine, "kn_retrieve") as conn:
        rows = conn.execute(text(f"SELECT scope FROM {table}")).all()
    assert rows == [("system",)]


def test_context_set_in_a_separate_call_does_not_apply(admin_engine: Engine, seeded: None) -> None:
    """Data API without a transaction: each call is its own session, so the setting is gone."""
    with admin_engine.connect() as conn:  # autocommit: every statement is its own transaction
        conn.execute(text("SET ROLE kn_retrieve"))
        try:
            conn.execute(text(SET_USER_CONTEXT_SQL), {"uid": "user-a"})
            mine = conn.execute(text("SELECT count(*) FROM kn_documents WHERE scope = 'mine'")).scalar_one()
        finally:
            conn.execute(text("RESET ROLE"))
    assert mine == 0


def test_user_context_set_after_the_transaction_ends_does_not_leak_into_the_next(
    app_engine: Engine, seeded: None
) -> None:
    with transaction_as(app_engine, "kn_retrieve", "user-a"):
        pass
    with transaction_as(app_engine, "kn_retrieve") as conn:
        mine = conn.execute(text("SELECT count(*) FROM kn_documents WHERE scope = 'mine'")).scalar_one()
    assert mine == 0


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO kn_documents (document_id, scope, user_id, source_path, content_hash, embedding_model, "
        "chunk_count, indexed_at) VALUES (gen_random_uuid(), 'mine', 'user-b', 'x', repeat('a', 64), 'm', 0, now())",
        "UPDATE kn_documents SET user_id = 'user-b' WHERE user_id = 'user-a'",
    ],
    ids=["insert-for-another-user", "update-owner-to-another-user"],
)
def test_writing_a_mine_row_for_another_user_is_rejected(
    app_engine: Engine, admin_engine: Engine, seeded: None, statement: str
) -> None:
    with pytest.raises(DBAPIError) as exc:
        with transaction_as(app_engine, "kn_ingest", "user-a") as conn:
            conn.execute(text(statement))
    assert isinstance(exc.value.orig, psycopg.errors.InsufficientPrivilege)


def test_writing_a_mine_row_without_user_context_is_rejected(app_engine: Engine, seeded: None) -> None:
    with pytest.raises(DBAPIError) as exc:
        with transaction_as(app_engine, "kn_ingest") as conn:
            conn.execute(
                text(
                    "INSERT INTO kn_documents (document_id, scope, user_id, source_path, content_hash, "
                    "embedding_model, chunk_count, indexed_at) "
                    "VALUES (gen_random_uuid(), 'mine', 'user-a', 'x', repeat('a', 64), 'm', 0, now())"
                )
            )
    assert isinstance(exc.value.orig, psycopg.errors.InsufficientPrivilege)


def test_writing_a_mine_row_for_the_current_user_is_accepted(app_engine: Engine, seeded: None) -> None:
    with transaction_as(app_engine, "kn_ingest", "user-c") as conn:
        conn.execute(
            text(
                "INSERT INTO kn_documents (document_id, scope, user_id, source_path, content_hash, "
                "embedding_model, chunk_count, indexed_at) "
                "VALUES (gen_random_uuid(), 'mine', 'user-c', 'x', repeat('a', 64), 'm', 0, now())"
            )
        )
