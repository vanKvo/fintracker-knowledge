"""KN-RET-04: scoped vector search against a real pgvector database, as role kn_retrieve.

__author__ = "Van Vo"
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import OperationalError

from src.core.exceptions import DatabaseUnavailableError
from src.retrieval.repository import PostgresChunkSearch
from tests.support.db import (
    EMBEDDING_MODEL,
    insert_chunk,
    insert_document,
    transaction_as,
    unit_vector,
)

QUERY = tuple(float(x) for x in unit_vector(0).strip("[]").split(","))


def blend(position: int) -> str:
    """Unit vector 0 plus a second axis: closer to the query than unit_vector(position) alone."""
    values = unit_vector(position).strip("[]").split(",")
    values[0] = "1"
    return "[" + ",".join(values) + "]"


def search_as_retrieve(engine: Engine) -> PostgresChunkSearch:
    @contextmanager
    def open_transaction() -> Iterator[Connection]:
        with transaction_as(engine, "kn_retrieve") as conn:  # the repository sets the user context itself
            yield conn

    return PostgresChunkSearch(open_transaction)


def test_system_scope_returns_only_system_chunks_nearest_first_even_when_user_chunks_are_nearer(
    admin_engine: Engine, app_engine: Engine
) -> None:
    system_doc = insert_document(admin_engine, "system", None, "help/a.md")
    mine_doc = insert_document(admin_engine, "mine", "user-a", "doc-1")
    insert_chunk(admin_engine, mine_doc, "mine", "user-a", unit_vector(0), chunk_text="nearest but mine")
    far = insert_chunk(admin_engine, system_doc, "system", None, unit_vector(2), ordinal=0)
    near = insert_chunk(admin_engine, system_doc, "system", None, blend(1), ordinal=1)

    hits = search_as_retrieve(app_engine).search("user-a", QUERY, "system", EMBEDDING_MODEL, 10).hits

    assert [h.chunk_id for h in hits] == [str(near), str(far)]
    assert all(h.scope == "system" and h.user_id is None for h in hits)
    assert hits[0].distance <= hits[1].distance


def test_hits_carry_source_path_heading_text_and_distance(admin_engine: Engine, app_engine: Engine) -> None:
    doc = insert_document(admin_engine, "system", None, "help/statements/csv-mapping.md")
    insert_chunk(admin_engine, doc, "system", None, unit_vector(0), chunk_text="Confirm the columns.")

    (found,) = search_as_retrieve(app_engine).search("user-a", QUERY, "system", EMBEDDING_MODEL, 10).hits

    assert (found.source_path, found.heading_path, found.text) == (
        "help/statements/csv-mapping.md", "Heading", "Confirm the columns.")
    assert found.distance == pytest.approx(0.0, abs=1e-6)
    assert found.document_id


def test_user_context_is_set_in_the_same_transaction_so_only_own_mine_rows_are_visible(
    admin_engine: Engine, app_engine: Engine
) -> None:
    doc_a = insert_document(admin_engine, "mine", "user-a", "doc-1")
    doc_b = insert_document(admin_engine, "mine", "user-b", "doc-1")
    a = insert_chunk(admin_engine, doc_a, "mine", "user-a", unit_vector(1))
    insert_chunk(admin_engine, doc_b, "mine", "user-b", unit_vector(0))

    hits = search_as_retrieve(app_engine).search("user-a", QUERY, "mine", EMBEDDING_MODEL, 10).hits

    assert [h.chunk_id for h in hits] == [str(a)]


def test_only_chunks_of_the_active_model_are_searched(admin_engine: Engine, app_engine: Engine) -> None:
    old_doc = insert_document(admin_engine, "system", None, "help/a.md", model="old-model")
    new_doc = insert_document(admin_engine, "system", None, "help/a.md")
    insert_chunk(admin_engine, old_doc, "system", None, unit_vector(0), model="old-model")
    current = insert_chunk(admin_engine, new_doc, "system", None, unit_vector(5))

    hits = search_as_retrieve(app_engine).search("user-a", QUERY, "system", EMBEDDING_MODEL, 10).hits

    assert [h.chunk_id for h in hits] == [str(current)]


def test_fewer_matches_than_the_limit_returns_them_all_without_error(
    admin_engine: Engine, app_engine: Engine
) -> None:
    doc = insert_document(admin_engine, "system", None, "help/a.md")
    insert_chunk(admin_engine, doc, "system", None, unit_vector(1))

    assert len(search_as_retrieve(app_engine).search("user-a", QUERY, "system", EMBEDDING_MODEL, 50).hits) == 1


def test_empty_index_returns_no_hits_and_no_commit(app_engine: Engine) -> None:
    result = search_as_retrieve(app_engine).search("user-a", QUERY, "system", EMBEDDING_MODEL, 10)
    assert result.hits == [] and result.index_commit is None


def test_index_commit_is_the_commit_of_the_latest_indexed_system_doc(
    admin_engine: Engine, app_engine: Engine
) -> None:
    old = insert_document(admin_engine, "system", None, "help/a.md")
    new = insert_document(admin_engine, "system", None, "help/b.md")
    with admin_engine.connect() as conn:
        conn.execute(text("UPDATE kn_documents SET source_commit = 'aaa1111', indexed_at = now() - interval '1 day' "
                          "WHERE document_id = :id"), {"id": old})
        conn.execute(text("UPDATE kn_documents SET source_commit = '9f2c1ab' WHERE document_id = :id"), {"id": new})

    assert search_as_retrieve(app_engine).search("user-a", QUERY, "system", EMBEDDING_MODEL, 10).index_commit == "9f2c1ab"


def test_database_error_surfaces_as_database_unavailable_without_driver_text() -> None:
    def broken() -> AbstractContextManager[Connection]:
        raise OperationalError("SELECT secret_table", {}, Exception("password authentication failed"))

    with pytest.raises(DatabaseUnavailableError) as caught:
        PostgresChunkSearch(broken).search("user-a", QUERY, "system", EMBEDDING_MODEL, 10)
    assert "secret_table" not in str(caught.value) and "password" not in str(caught.value)


def test_search_runs_as_kn_retrieve_and_cannot_write(admin_engine: Engine, app_engine: Engine) -> None:
    seen: list[str] = []

    def spying() -> AbstractContextManager[Connection]:
        @contextmanager
        def open_transaction() -> Iterator[Connection]:
            with transaction_as(app_engine, "kn_retrieve") as conn:
                yield conn
                seen.append(conn.execute(text("SELECT current_user")).scalar_one())

        return open_transaction()

    PostgresChunkSearch(spying).search("user-a", QUERY, "system", EMBEDDING_MODEL, 10)
    assert seen == ["kn_retrieve"]

