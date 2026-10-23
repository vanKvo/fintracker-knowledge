"""System-doc rows and run records, written as role kn_ingest (KN-SDOC-03, -06).

`connection` opens one transaction per call, already running as kn_ingest. Any database failure
surfaces as DatabaseUnavailable; the driver's message (SQL, identifiers) is never passed on.

__author__ = "Van Vo"
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from src.embedding.models import EmbeddedChunk
from src.system_docs.models import DatabaseUnavailable, IndexStatus, RunRecord

logger = logging.getLogger(__name__)


class DocumentIndex(Protocol):
    def is_indexed(self, path: str, content_hash: str, model: str) -> bool: ...
    def replace(self, path: str, content_hash: str, commit: str, model: str, chunks: list[EmbeddedChunk]) -> int: ...
    def delete(self, path: str) -> int: ...
    def record_run(self, record: RunRecord) -> None: ...
    def last_succeeded_run(self) -> IndexStatus: ...


def _vector_literal(values: tuple[float, ...]) -> str:
    return "[" + ",".join(repr(v) for v in values) + "]"


class PostgresDocumentIndex:
    def __init__(self, connection: Callable[[], AbstractContextManager[Connection]]) -> None:
        self._connection = connection

    def is_indexed(self, path: str, content_hash: str, model: str) -> bool:
        with self._transaction() as conn:
            row = conn.execute(
                text(
                    "SELECT 1 FROM kn_documents WHERE scope = 'system' AND source_path = :path "
                    "AND content_hash = :hash AND embedding_model = :model"
                ),
                {"path": path, "hash": content_hash, "model": model},
            ).first()
            return row is not None

    def replace(self, path: str, content_hash: str, commit: str, model: str, chunks: list[EmbeddedChunk]) -> int:
        """One transaction: drop this model's old version, then write the new one. Returns chunks written."""
        document_id = uuid.uuid4()
        with self._transaction() as conn:
            conn.execute(
                text(
                    "DELETE FROM kn_documents WHERE scope = 'system' AND source_path = :path "
                    "AND embedding_model = :model"
                ),
                {"path": path, "model": model},
            )
            conn.execute(
                text(
                    "INSERT INTO kn_documents (document_id, scope, user_id, source_path, content_hash, "
                    "source_commit, embedding_model, chunk_count, indexed_at) "
                    "VALUES (:id, 'system', NULL, :path, :hash, :commit, :model, :count, now())"
                ),
                {"id": document_id, "path": path, "hash": content_hash, "commit": commit, "model": model,
                 "count": len(chunks)},
            )
            if chunks:
                conn.execute(
                    text(
                        "INSERT INTO kn_chunks (chunk_id, document_id, scope, user_id, ordinal, heading_path, "
                        "text, token_count, embedding, embedding_model) "
                        "VALUES (:chunk_id, :doc, 'system', NULL, :ordinal, :heading, :text, :tokens, "
                        "CAST(:embedding AS vector), :model)"
                    ),
                    [
                        {
                            "chunk_id": uuid.uuid4(),
                            "doc": document_id,
                            "ordinal": e.chunk.ordinal,
                            "heading": e.chunk.heading_path,
                            "text": e.chunk.text,
                            "tokens": e.chunk.token_count,
                            "embedding": _vector_literal(e.embedding),
                            "model": model,
                        }
                        for e in chunks
                    ],
                )
        return len(chunks)

    def delete(self, path: str) -> int:
        """Removes the doc under every model (chunks cascade). 1 if anything was indexed, else 0."""
        with self._transaction() as conn:
            result = conn.execute(
                text("DELETE FROM kn_documents WHERE scope = 'system' AND source_path = :path"), {"path": path}
            )
            return 1 if result.rowcount else 0

    def record_run(self, record: RunRecord) -> None:
        with self._transaction() as conn:
            conn.execute(
                text(
                    'INSERT INTO kn_index_runs (run_id, mode, "commit", embedding_model, status, docs_indexed, '
                    "docs_skipped, docs_deleted, docs_failed, chunks_written, started_at, finished_at) "
                    "VALUES (:run_id, :mode, :commit, :model, :status, :indexed, :skipped, :deleted, :failed, "
                    ":chunks, :started, :finished)"
                ),
                {
                    "run_id": uuid.UUID(record.run_id),
                    "mode": record.mode,
                    "commit": record.commit,
                    "model": record.embedding_model,
                    "status": record.status,
                    "indexed": record.docs_indexed,
                    "skipped": record.docs_skipped,
                    "deleted": record.docs_deleted,
                    "failed": record.docs_failed,
                    "chunks": record.chunks_written,
                    "started": record.started_at,
                    "finished": record.finished_at,
                },
            )

    def last_succeeded_run(self) -> IndexStatus:
        with self._transaction() as conn:
            row = conn.execute(
                text(
                    'SELECT "commit", finished_at FROM kn_index_runs WHERE status = \'succeeded\' '
                    "ORDER BY finished_at DESC LIMIT 1"
                )
            ).first()
        finished: datetime | None = row[1] if row else None
        return IndexStatus(row[0] if row else None, finished)

    @contextmanager
    def _transaction(self) -> Iterator[Connection]:
        try:
            with self._connection() as conn:
                yield conn
        except SQLAlchemyError as exc:
            logger.error("Database call failed: %s", type(exc).__name__)
            raise DatabaseUnavailable() from None
