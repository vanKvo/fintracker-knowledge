"""KN-VS-05: isolation test suite — two users plus system rows, every Knowledge role, both call paths.

__author__ = "Van Vo"
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from src.vector_store.session import SET_USER_CONTEXT_SQL
from tests.support.db import insert_chunk, insert_document, transaction_as, unit_vector

SEARCH = "SELECT chunk_id FROM kn_chunks WHERE {where} ORDER BY embedding <=> CAST(:q AS vector) LIMIT 10"
READ_ROLES = ["kn_retrieve", "kn_ingest", "kn_delete"]


@dataclass(frozen=True)
class Seed:
    user_b_documents: set[uuid.UUID]
    user_b_chunks: set[uuid.UUID]
    user_a_chunks: set[uuid.UUID]
    system_chunks: set[uuid.UUID]


def assert_no_foreign_rows(table: str, found: set[uuid.UUID], forbidden: set[uuid.UUID]) -> None:
    """Fails naming the table only — never the row text."""
    if found & forbidden:
        raise AssertionError(f"isolation leak in {table}: {len(found & forbidden)} foreign row(s) visible")


@pytest.fixture
def seed(admin_engine: Engine) -> Seed:
    doc_sys = insert_document(admin_engine, "system", None, "help/a.md")
    doc_a = insert_document(admin_engine, "mine", "user-a", "doc-1")
    doc_b = insert_document(admin_engine, "mine", "user-b", "doc-1")
    # User B's chunk is the nearest to the query vector (position 0), then user A's, then system.
    b = insert_chunk(admin_engine, doc_b, "mine", "user-b", unit_vector(0), chunk_text="user b secret")
    a = insert_chunk(admin_engine, doc_a, "mine", "user-a", unit_vector(1))
    s = insert_chunk(admin_engine, doc_sys, "system", None, unit_vector(2))
    return Seed({doc_b}, {b}, {a}, {s})


@contextmanager
def _single_transaction(app_engine: Engine, admin_engine: Engine, role: str) -> Iterator[Connection]:
    with transaction_as(app_engine, role, "user-a") as conn:
        yield conn


@contextmanager
def _separate_call(app_engine: Engine, admin_engine: Engine, role: str) -> Iterator[Connection]:
    """No-transaction Data API case: the context is set in an earlier call and is gone."""
    with admin_engine.connect() as conn:
        conn.execute(text(f"SET ROLE {role}"))
        try:
            conn.execute(text(SET_USER_CONTEXT_SQL), {"uid": "user-a"})
            yield conn
        finally:
            conn.execute(text("RESET ROLE"))


PATHS: dict[str, Callable[[Engine, Engine, str], "contextmanager[Connection]"]] = {  # type: ignore[type-arg]
    "single-transaction": _single_transaction,
    "separate-call": _separate_call,
}


@pytest.mark.parametrize("role", READ_ROLES)
@pytest.mark.parametrize("path", PATHS)
def test_user_a_never_sees_user_b_rows_in_either_table_or_in_vector_search(
    app_engine: Engine, admin_engine: Engine, seed: Seed, role: str, path: str
) -> None:
    with PATHS[path](app_engine, admin_engine, role) as conn:
        documents = {r[0] for r in conn.execute(text("SELECT document_id FROM kn_documents"))}
        chunks = {r[0] for r in conn.execute(text("SELECT chunk_id FROM kn_chunks"))}
        searched = {r[0] for r in conn.execute(text(SEARCH.format(where="true")), {"q": unit_vector(0)})}
    assert_no_foreign_rows("kn_documents", documents, seed.user_b_documents)
    assert_no_foreign_rows("kn_chunks", chunks, seed.user_b_chunks)
    assert_no_foreign_rows("kn_chunks (vector search)", searched, seed.user_b_chunks)


@pytest.mark.parametrize("role", READ_ROLES)
@pytest.mark.parametrize("path", PATHS)
def test_system_scope_search_returns_no_user_rows_even_when_user_rows_are_nearer(
    app_engine: Engine, admin_engine: Engine, seed: Seed, role: str, path: str
) -> None:
    with PATHS[path](app_engine, admin_engine, role) as conn:
        found = {
            r[0]
            for r in conn.execute(text(SEARCH.format(where="scope = 'system'")), {"q": unit_vector(0)})
        }
    assert found == seed.system_chunks


def test_single_transaction_path_returns_the_callers_own_rows(
    app_engine: Engine, admin_engine: Engine, seed: Seed
) -> None:
    with transaction_as(app_engine, "kn_retrieve", "user-a") as conn:
        chunks = {r[0] for r in conn.execute(text("SELECT chunk_id FROM kn_chunks"))}
    assert chunks == seed.user_a_chunks | seed.system_chunks


def test_separate_call_path_returns_system_rows_only(app_engine: Engine, admin_engine: Engine, seed: Seed) -> None:
    with _separate_call(app_engine, admin_engine, "kn_retrieve") as conn:
        chunks = {r[0] for r in conn.execute(text("SELECT chunk_id FROM kn_chunks"))}
    assert chunks == seed.system_chunks


def test_a_leak_fails_the_build_naming_the_table_and_never_the_row_text() -> None:
    leaked = uuid.uuid4()
    with pytest.raises(AssertionError) as exc:
        assert_no_foreign_rows("kn_chunks", {leaked}, {leaked})
    message = str(exc.value)
    assert "kn_chunks" in message
    assert "user b secret" not in message
    assert str(leaked) not in message
