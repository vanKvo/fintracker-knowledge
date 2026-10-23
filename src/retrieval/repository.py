"""Scoped vector search, run as role kn_retrieve (KN-RET-04).

`connection` opens one transaction per call, already running as kn_retrieve. The user context and
search settings are set inside it, so Row-Level Security applies to the query (KN-VS-03, -04). Any
database failure surfaces as DatabaseUnavailable; the driver's message is never passed on.

__author__ = "Van Vo"
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from src.retrieval.models import DatabaseUnavailable, SearchHit, SearchResult
from src.vector_store.session import SEARCH_SESSION_SQL, SET_USER_CONTEXT_SQL

logger = logging.getLogger(__name__)

# Rows fetched per search. The per-document cap is applied afterwards, so this must exceed k by enough
# that capped documents can't starve the result (k <= 10, cap 3).
SEARCH_POOL = 50


class ChunkSearch(Protocol):
    def search(
        self, user_id: str, vector: tuple[float, ...], scope: str, model: str, limit: int
    ) -> SearchResult: ...


def _vector_literal(values: tuple[float, ...]) -> str:
    return "[" + ",".join(repr(v) for v in values) + "]"


class PostgresChunkSearch:
    def __init__(self, connection: Callable[[], AbstractContextManager[Connection]]) -> None:
        self._connection = connection

    def search(
        self, user_id: str, vector: tuple[float, ...], scope: str, model: str, limit: int = SEARCH_POOL
    ) -> SearchResult:
        try:
            with self._connection() as conn:
                conn.execute(text(SET_USER_CONTEXT_SQL), {"uid": user_id})
                for statement in SEARCH_SESSION_SQL:
                    conn.execute(text(statement))
                rows = conn.execute(
                    text(
                        "SELECT c.chunk_id, c.document_id, c.scope, c.user_id, d.source_path, c.heading_path, "
                        "c.text, c.embedding <=> CAST(:q AS vector) AS distance "
                        "FROM kn_chunks c JOIN kn_documents d ON d.document_id = c.document_id "
                        "WHERE c.scope = :scope AND c.embedding_model = :model "
                        "ORDER BY c.embedding <=> CAST(:q AS vector) LIMIT :limit"
                    ),
                    {"q": _vector_literal(vector), "scope": scope, "model": model, "limit": limit},
                ).all()
                commit = conn.execute(
                    text(
                        "SELECT source_commit FROM kn_documents WHERE scope = 'system' AND embedding_model = :model "
                        "AND source_commit IS NOT NULL ORDER BY indexed_at DESC LIMIT 1"
                    ),
                    {"model": model},
                ).scalar()
        except SQLAlchemyError as exc:
            logger.error("Database call failed: %s", type(exc).__name__)
            raise DatabaseUnavailable() from None
        hits = [
            SearchHit(str(r[0]), str(r[1]), r[2], r[3], r[4], r[5], r[6], float(r[7])) for r in rows
        ]
        return SearchResult(hits, commit)
