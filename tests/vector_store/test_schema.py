"""KN-VS-01: document and chunk schema.

__author__ = "Van Vo"
"""

from __future__ import annotations

import uuid
from pathlib import Path

import psycopg.errors
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError
from testcontainers.postgres import PostgresContainer

from tests.support.db import DIMENSIONS, insert_chunk, insert_document, unit_vector
from tests.support.migrate import MigrationError, apply_migrations


def test_migration_creates_tables_and_pgvector_extension(admin_engine: Engine) -> None:
    with admin_engine.connect() as conn:
        tables = {
            r[0]
            for r in conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            )
        }
        extensions = {r[0] for r in conn.execute(text("SELECT extname FROM pg_extension"))}
    assert {"kn_documents", "kn_chunks"} <= tables
    assert "vector" in extensions


@pytest.mark.parametrize(
    ("scope", "user_id"),
    [("system", "user-a"), ("mine", None), ("mine", ""), ("shared", None)],
    ids=["system-with-owner", "mine-without-owner", "mine-with-empty-owner", "unknown-scope"],
)
def test_scope_and_owner_check_constraint_rejects_invalid_combinations(
    admin_engine: Engine, scope: str, user_id: str | None
) -> None:
    with pytest.raises(DBAPIError) as exc:
        insert_document(admin_engine, scope, user_id, "help/a.md")
    assert isinstance(exc.value.orig, psycopg.errors.CheckViolation)


@pytest.mark.parametrize(("scope", "user_id"), [("system", None), ("mine", "user-a")])
def test_second_document_with_same_scope_owner_and_path_is_rejected(
    admin_engine: Engine, scope: str, user_id: str | None
) -> None:
    insert_document(admin_engine, scope, user_id, "help/a.md")
    with pytest.raises(DBAPIError) as exc:
        insert_document(admin_engine, scope, user_id, "help/a.md")
    assert isinstance(exc.value.orig, psycopg.errors.UniqueViolation)


def test_same_path_under_a_different_owner_or_model_is_allowed(admin_engine: Engine) -> None:
    insert_document(admin_engine, "mine", "user-a", "doc-1")
    insert_document(admin_engine, "mine", "user-b", "doc-1")
    insert_document(admin_engine, "mine", "user-a", "doc-1", model="other-model")


def test_deleting_a_document_deletes_its_chunks(admin_engine: Engine) -> None:
    doc = insert_document(admin_engine, "system", None, "help/a.md")
    insert_chunk(admin_engine, doc, "system", None, unit_vector(0))
    insert_chunk(admin_engine, doc, "system", None, unit_vector(1), ordinal=1)
    with admin_engine.connect() as conn:
        conn.execute(text("DELETE FROM kn_documents WHERE document_id = :id"), {"id": doc})
        remaining = conn.execute(text("SELECT count(*) FROM kn_chunks")).scalar_one()
    assert remaining == 0


def test_chunk_embedding_model_must_equal_the_documents(admin_engine: Engine) -> None:
    doc = insert_document(admin_engine, "system", None, "help/a.md")
    with pytest.raises(DBAPIError) as exc:
        insert_chunk(admin_engine, doc, "system", None, unit_vector(0), model="other-model")
    assert isinstance(exc.value.orig, psycopg.errors.ForeignKeyViolation)


@pytest.mark.parametrize(
    ("chunk_scope", "chunk_user"),
    [("system", None), ("mine", "user-b"), ("mine", "user-a")],
    ids=["system-chunk-on-mine-doc", "other-owner", "mine-chunk-on-system-doc"],
)
def test_chunk_scope_and_owner_must_equal_the_documents(
    admin_engine: Engine, chunk_scope: str, chunk_user: str | None
) -> None:
    mine_doc = insert_document(admin_engine, "mine", "user-a", "doc-1")
    system_doc = insert_document(admin_engine, "system", None, "help/a.md")
    document = system_doc if chunk_scope == "mine" and chunk_user == "user-a" else mine_doc
    with pytest.raises(DBAPIError) as exc:
        insert_chunk(admin_engine, document, chunk_scope, chunk_user, unit_vector(0))
    assert isinstance(exc.value.orig, psycopg.errors.ForeignKeyViolation)


def test_chunk_with_matching_document_is_accepted(admin_engine: Engine) -> None:
    doc = insert_document(admin_engine, "mine", "user-a", "doc-1")
    insert_chunk(admin_engine, doc, "mine", "user-a", unit_vector(0))


def test_embedding_must_have_the_configured_dimensions(admin_engine: Engine) -> None:
    doc = insert_document(admin_engine, "system", None, "help/a.md")
    assert DIMENSIONS == 1024
    with pytest.raises(DBAPIError):
        insert_chunk(admin_engine, doc, "system", None, "[1,2,3]")


def test_failed_migration_stops_later_migrations_and_leaves_nothing_partial(
    postgres: PostgresContainer, admin_engine: Engine, tmp_path: Path
) -> None:
    (tmp_path / "V1__first.sql").write_text("CREATE TABLE first_ok (id int);")
    (tmp_path / "V2__broken.sql").write_text("CREATE TABLE half_done (id int); SELECT * FROM missing_table;")
    (tmp_path / "V3__never_runs.sql").write_text("CREATE TABLE third (id int);")
    with admin_engine.connect() as conn:
        conn.execute(text("CREATE DATABASE scratch"))
    scratch = create_engine(postgres.get_connection_url().replace("/knowledge", "/scratch"), isolation_level="AUTOCOMMIT")
    try:
        with pytest.raises(MigrationError):
            apply_migrations(scratch, tmp_path)
        with scratch.connect() as conn:
            tables = {r[0] for r in conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))}
    finally:
        scratch.dispose()
    assert tables == {"first_ok"}
