"""Document rows per embedding model, read and deleted as role kn_ingest (system docs only).

__author__ = "Van Vo"
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.engine import Connection


class ModelIndex(Protocol):
    def docs_missing_model(self, model: str) -> list[str]: ...
    def delete_model_rows(self, model: str) -> None: ...
    def delete_other_models_rows(self, keep_model: str) -> None: ...


class ActiveModelSetting(Protocol):
    def get(self) -> str: ...
    def set(self, model: str) -> None: ...


class PostgresModelIndex:
    """`connection` opens one transaction per call, already running as kn_ingest."""

    def __init__(self, connection: Callable[[], AbstractContextManager[Connection]]) -> None:
        self._connection = connection

    def docs_missing_model(self, model: str) -> list[str]:
        with self._connection() as conn:
            rows = conn.execute(
                text(
                    "SELECT DISTINCT d.source_path FROM kn_documents d WHERE d.scope = 'system' "
                    "AND NOT EXISTS (SELECT 1 FROM kn_documents n WHERE n.scope = 'system' "
                    "AND n.source_path = d.source_path AND n.embedding_model = :model AND n.chunk_count > 0) "
                    "ORDER BY 1"
                ),
                {"model": model},
            )
            return [row[0] for row in rows]

    def delete_model_rows(self, model: str) -> None:
        with self._connection() as conn:  # chunks follow by ON DELETE CASCADE
            conn.execute(
                text("DELETE FROM kn_documents WHERE scope = 'system' AND embedding_model = :model"),
                {"model": model},
            )

    def delete_other_models_rows(self, keep_model: str) -> None:
        with self._connection() as conn:
            conn.execute(
                text("DELETE FROM kn_documents WHERE scope = 'system' AND embedding_model <> :model"),
                {"model": keep_model},
            )
