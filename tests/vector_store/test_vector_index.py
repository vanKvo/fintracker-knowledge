"""KN-VS-04: vector index and filtered-search recall.

__author__ = "Van Vo"
"""

from __future__ import annotations

import random

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from src.vector_store.session import HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, HNSW_ITERATIVE_SCAN, HNSW_M
from tests.support.db import EMBEDDING_MODEL, insert_document, random_vector, transaction_as

SEARCH = "SELECT chunk_id FROM kn_chunks ORDER BY embedding <=> CAST(:q AS vector) LIMIT :k"


def _seed_corpus(engine: Engine, owners: dict[str, int]) -> str:
    """Random vectors per owner (None = system). Returns a random query vector."""
    rng = random.Random(42)
    ordinal = 0
    rows: list[dict[str, object]] = []
    for owner, count in owners.items():
        scope = "system" if owner == "system" else "mine"
        user_id = None if owner == "system" else owner
        doc = insert_document(engine, scope, user_id, "doc")
        for _ in range(count):
            rows.append({"doc": doc, "scope": scope, "uid": user_id, "ordinal": ordinal, "emb": random_vector(rng)})
            ordinal += 1
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO kn_chunks (chunk_id, document_id, scope, user_id, ordinal, heading_path, text, "
                "token_count, embedding, embedding_model) VALUES (gen_random_uuid(), :doc, :scope, :uid, "
                f":ordinal, 'Heading', 'chunk text', 10, CAST(:emb AS vector), '{EMBEDDING_MODEL}')"
            ),
            rows,
        )
    return random_vector(rng)


def _search(conn: Connection, query: str, k: int = 5) -> list[tuple[object]]:
    """Forces the planner onto the HNSW index; on a small test table it would otherwise seq-scan
    and never exercise the approximate-search recall behaviour."""
    conn.execute(text("SELECT set_config('enable_seqscan', 'off', true)"))
    plan = "\n".join(r[0] for r in conn.execute(text("EXPLAIN " + SEARCH), {"q": query, "k": k}))
    assert "kn_chunks_embedding_hnsw" in plan
    return [tuple(r) for r in conn.execute(text(SEARCH), {"q": query, "k": k})]


def test_index_is_hnsw_cosine_with_configured_build_parameters_and_is_valid(admin_engine: Engine) -> None:
    with admin_engine.connect() as conn:
        definition, valid = conn.execute(
            text(
                "SELECT pg_get_indexdef(i.indexrelid), i.indisvalid FROM pg_index i "
                "JOIN pg_class c ON c.oid = i.indrelid WHERE c.relname = 'kn_chunks' "
                "AND pg_get_indexdef(i.indexrelid) LIKE '%hnsw%'"
            )
        ).one()
    assert valid is True
    assert "(embedding vector_cosine_ops)" in definition
    assert f"m='{HNSW_M}'" in definition
    assert f"ef_construction='{HNSW_EF_CONSTRUCTION}'" in definition


def test_search_session_applies_ef_search_and_iterative_scan(app_engine: Engine) -> None:
    with transaction_as(app_engine, "kn_retrieve", "user-a", search=True) as conn:
        ef_search = conn.execute(text("SHOW hnsw.ef_search")).scalar_one()
        iterative = conn.execute(text("SHOW hnsw.iterative_scan")).scalar_one()
    assert (ef_search, iterative) == (str(HNSW_EF_SEARCH), HNSW_ITERATIVE_SCAN)


def test_search_returns_exactly_k_when_enough_rows_pass_the_filter(
    app_engine: Engine, admin_engine: Engine
) -> None:
    query = _seed_corpus(admin_engine, {"user-a": 40, "user-b": 40})
    with transaction_as(app_engine, "kn_retrieve", "user-a", search=True) as conn:
        rows = _search(conn, query)
    assert len(rows) == 5


def test_search_still_returns_k_when_99_percent_of_rows_belong_to_other_users(
    app_engine: Engine, admin_engine: Engine
) -> None:
    query = _seed_corpus(admin_engine, {"user-a": 20, "user-b": 2980})
    with transaction_as(app_engine, "kn_retrieve", "user-a", search=True) as conn:
        rows = _search(conn, query)
        owners = conn.execute(text("SELECT DISTINCT user_id FROM kn_chunks")).all()
    assert len(rows) == 5
    assert owners == [("user-a",)]


def test_search_returns_all_matching_rows_without_error_when_fewer_than_k_pass_the_filter(
    app_engine: Engine, admin_engine: Engine
) -> None:
    query = _seed_corpus(admin_engine, {"user-a": 3, "user-b": 300})
    with transaction_as(app_engine, "kn_retrieve", "user-a", search=True) as conn:
        rows = _search(conn, query)
    assert len(rows) == 3
