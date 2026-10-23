"""Test helpers: role-scoped sessions that mirror how Knowledge workflows call the database.

__author__ = "Van Vo"
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from src.vector_store.session import SEARCH_SESSION_SQL, SET_USER_CONTEXT_SQL

EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 1024


@contextmanager
def transaction_as(
    engine: Engine, role: str, user_id: str | None = None, *, search: bool = False
) -> Iterator[Connection]:
    """One Data API-style transaction: role, user context and (optionally) search tuning set
    inside the same transaction as the caller's queries."""
    with engine.begin() as conn:
        conn.execute(text(f"SET LOCAL ROLE {role}"))
        if user_id is not None:
            conn.execute(text(SET_USER_CONTEXT_SQL), {"uid": user_id})
        if search:
            for statement in SEARCH_SESSION_SQL:
                conn.execute(text(statement))
        yield conn


def unit_vector(position: int) -> str:
    """A 1024-dim vector literal with a single 1 — distances between these are exact and easy to reason about."""
    values = ["0"] * DIMENSIONS
    values[position] = "1"
    return "[" + ",".join(values) + "]"


def random_vector(rng: random.Random) -> str:
    return "[" + ",".join(f"{rng.uniform(-1, 1):.4f}" for _ in range(DIMENSIONS)) + "]"


def insert_document(
    engine: Engine, scope: str, user_id: str | None, source_path: str, model: str = EMBEDDING_MODEL
) -> uuid.UUID:
    document_id = uuid.uuid4()
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO kn_documents (document_id, scope, user_id, source_path, content_hash, "
                "embedding_model, chunk_count, indexed_at) "
                "VALUES (:id, :scope, :uid, :path, repeat('a', 64), :model, 1, now())"
            ),
            {"id": document_id, "scope": scope, "uid": user_id, "path": source_path, "model": model},
        )
    return document_id


def insert_chunk(
    engine: Engine,
    document_id: uuid.UUID,
    scope: str,
    user_id: str | None,
    embedding: str,
    *,
    ordinal: int = 0,
    chunk_text: str = "chunk text",
    model: str = EMBEDDING_MODEL,
) -> uuid.UUID:
    chunk_id = uuid.uuid4()
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO kn_chunks (chunk_id, document_id, scope, user_id, ordinal, heading_path, "
                "text, token_count, embedding, embedding_model) "
                "VALUES (:id, :doc, :scope, :uid, :ordinal, 'Heading', :text, 10, CAST(:emb AS vector), :model)"
            ),
            {
                "id": chunk_id,
                "doc": document_id,
                "scope": scope,
                "uid": user_id,
                "ordinal": ordinal,
                "text": chunk_text,
                "emb": embedding,
                "model": model,
            },
        )
    return chunk_id
