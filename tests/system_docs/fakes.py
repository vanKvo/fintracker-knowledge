"""Test doubles and helpers shared by the KN-SDOC suites.

__author__ = "Van Vo"
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from src.chunking.models import Chunk
from src.embedding.models import EmbeddedChunk, EmbeddingUnavailable
from src.system_docs.models import SourceMissing
from src.system_docs.repository import PostgresDocumentIndex
from tests.support.db import DIMENSIONS, EMBEDDING_MODEL, transaction_as

NEW_MODEL = "new.model:2"
COMMIT = "9f2c1ab"
NOW = datetime(2026, 10, 20, 18, 5, tzinfo=UTC)
SECRET = "SECRET-DOC-PHRASE"


def doc(marker: str, sections: int = 2) -> bytes:
    """A help doc whose every section is long enough to stay its own chunk."""
    parts = [f"# Guide {marker}"]
    for n in range(sections):
        parts.append(f"## Section {n}\n\n" + f"{marker} text for section {n}. " * 20)
    return "\n\n".join(parts).encode()


class FakeSource:
    def __init__(self, files: dict[str, bytes] | None = None, delay: float = 0.0) -> None:
        self.files = dict(files or {})
        self.delay = delay
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def read(self, path: str) -> bytes:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if path not in self.files:
                raise SourceMissing()
            return self.files[path]
        finally:
            with self._lock:
                self._active -= 1

    def list_paths(self) -> list[str]:
        return sorted(self.files)


class FakeEmbedder:
    """Vectors are unit vectors; `failures` are raised one per call before any success."""

    def __init__(self, model: str = EMBEDDING_MODEL, failures: list[Exception] | None = None) -> None:
        self.model = model
        self.failures = list(failures or [])
        self.calls = 0
        self.texts: list[str] = []

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        self.texts.extend(chunk.text for chunk in chunks)
        out = []
        for chunk in chunks:
            vector = [0.0] * DIMENSIONS
            vector[chunk.ordinal % DIMENSIONS] = 1.0
            out.append(EmbeddedChunk(chunk, tuple(vector), self.model))
        return out


class AlwaysUnavailable(FakeEmbedder):
    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        self.calls += 1
        raise EmbeddingUnavailable()


class Sleeps:
    def __init__(self) -> None:
        self.seconds: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.seconds.append(seconds)


def make_index(
    app_engine: Engine,
    before_commit: Callable[[], None] | None = None,
) -> PostgresDocumentIndex:
    """One transaction per call as kn_ingest; `before_commit` runs inside it, after the caller's statements."""

    def connection() -> AbstractContextManager[Connection]:
        @contextmanager
        def open_transaction() -> Iterator[Connection]:
            with transaction_as(app_engine, "kn_ingest") as conn:
                yield conn
                if before_commit is not None:
                    before_commit()

        return open_transaction()

    return PostgresDocumentIndex(connection)


def indexed_docs(engine: Engine) -> dict[tuple[str, str], dict[str, Any]]:
    """(path, model) -> row, read with the owner/admin engine."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT source_path, embedding_model, content_hash, source_commit, chunk_count, scope, user_id "
                "FROM kn_documents"
            )
        ).mappings()
        return {(r["source_path"], r["embedding_model"]): dict(r) for r in rows}


def searchable_texts(app_engine: Engine, path: str) -> list[str]:
    """What a search would see for a path: chunk text read as kn_retrieve."""
    with transaction_as(app_engine, "kn_retrieve") as conn:
        rows = conn.execute(
            text(
                "SELECT c.text FROM kn_chunks c JOIN kn_documents d USING (document_id) "
                "WHERE d.source_path = :p ORDER BY c.ordinal"
            ),
            {"p": path},
        )
        return [r[0] for r in rows]
