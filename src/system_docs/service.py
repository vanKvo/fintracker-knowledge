"""KN-SDOC-02 to -06: index system help docs, one isolated branch per doc.

A branch ends as indexed, skipped, deleted or failed (Appendix B); a failed branch never stops
the others, and its previous version stays searchable because replace is one transaction.
Logs carry paths, counts and error codes, never doc or chunk text.

__author__ = "Van Vo"
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Protocol, TypeVar

from src.chunking.schemas import Chunk
from src.chunking.service import chunk_markdown
from src.core.exceptions import DatabaseUnavailableError, EmbeddingUnavailableError
from src.embedding.schemas import EmbeddedChunk
from src.system_docs import schemas as m
from src.system_docs.exceptions import InvalidSource, SourceMissing
from src.system_docs.models import RunRecord
from src.system_docs.publisher import MAX_DOC_BYTES
from src.system_docs.repository import DocumentIndex
from src.system_docs.source import DocSource

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_CONCURRENCY = 4
RETRY_BACKOFF_SECONDS = (2.0, 4.0, 8.0)  # KN-SDOC-04: 3 retries


class DocEmbedder(Protocol):
    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]: ...


class SystemDocsIndexer:
    def __init__(
        self,
        source: DocSource,
        index: DocumentIndex,
        embedder_for: Callable[[str], DocEmbedder],
        *,
        active_model: str,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        emit_metrics: Callable[[RunRecord], None] | None = None,
    ) -> None:
        self._source = source
        self._index = index
        self._embedder_for = embedder_for
        self._active_model = active_model
        self._sleep = sleep
        self._clock = clock
        self._emit_metrics = emit_metrics or _log_metrics

    def run_manifest(self, manifest: m.Manifest) -> m.RunResult:
        """Incremental run: changed paths are hash-checked, deleted paths are removed."""
        started = self._clock()
        with ThreadPoolExecutor(MAX_CONCURRENCY) as pool:
            changed = list(pool.map(lambda p: self.process_changed(p, manifest.commit), manifest.changed))
            deleted = list(pool.map(self.process_deleted, manifest.deleted))
        return self._finish("incremental", manifest.commit, self._active_model, started, [*changed, *deleted])

    def run_rebuild(self, commit: str, embedding_model: str) -> m.RunResult:
        """Re-index every doc, ignoring the hash skip. A new model's rows sit beside the old ones."""
        started = self._clock()
        with ThreadPoolExecutor(MAX_CONCURRENCY) as pool:
            branches = list(
                pool.map(
                    lambda p: self.process_changed(p, commit, model=embedding_model, force=True),
                    self._source.list_paths(),
                )
            )
        return self._finish("rebuild", commit, embedding_model, started, branches)

    def process_changed(
        self, path: str, commit: str, *, model: str | None = None, force: bool = False
    ) -> m.BranchResult:
        model = model or self._active_model
        try:
            data = self._source.read(path)
            if len(data) > MAX_DOC_BYTES:
                raise InvalidSource()
            try:
                markdown = data.decode("utf-8")
            except UnicodeDecodeError:
                raise InvalidSource() from None
            digest = hashlib.sha256(data).hexdigest()
            if not force and self._retry(lambda: self._index.is_indexed(path, digest, model)):
                return self._done(m.BranchResult(path, m.SKIPPED))
            chunks = chunk_markdown(path, markdown)
            embedder = self._embedder_for(model)
            embedded = self._retry(lambda: embedder.embed_chunks(chunks))
            written = self._retry(lambda: self._index.replace(path, digest, commit, model, embedded))
            return self._done(m.BranchResult(path, m.INDEXED, written))
        except SourceMissing:
            return self._done(m.BranchResult(path, m.SOURCE_MISSING))
        except InvalidSource:
            return self._done(m.BranchResult(path, m.INVALID_SOURCE))
        except EmbeddingUnavailableError:
            return self._done(m.BranchResult(path, m.EMBEDDING_UNAVAILABLE))
        except DatabaseUnavailableError:
            return self._done(m.BranchResult(path, m.DATABASE_UNAVAILABLE))

    def process_deleted(self, path: str) -> m.BranchResult:
        try:
            return self._done(m.BranchResult(path, m.deleted_outcome(self._retry(lambda: self._index.delete(path)))))
        except DatabaseUnavailableError:
            return self._done(m.BranchResult(path, m.DATABASE_UNAVAILABLE))

    def _retry(self, step: Callable[[], T]) -> T:
        for delay in RETRY_BACKOFF_SECONDS:
            try:
                return step()
            except (EmbeddingUnavailableError, DatabaseUnavailableError):
                self._sleep(delay)
        return step()

    @staticmethod
    def _done(result: m.BranchResult) -> m.BranchResult:
        logger.info("Doc branch finished: path=%s outcome=%s", result.path, result.outcome)
        return result

    def _finish(
        self, mode: m.RunMode, commit: str, model: str, started: datetime, branches: list[m.BranchResult]
    ) -> m.RunResult:
        def count(prefix: str) -> int:
            return sum(1 for b in branches if b.outcome.startswith(prefix))

        failed = count("failed")
        status: m.RunStatus = "failed" if branches and failed == len(branches) else "partial" if failed else "succeeded"
        record = RunRecord(
            run_id=str(uuid.uuid4()),
            mode=mode,
            commit=commit,
            embedding_model=model,
            status=status,
            docs_indexed=count(m.INDEXED),
            docs_skipped=count(m.SKIPPED),
            docs_deleted=sum(1 for b in branches if b.outcome == m.deleted_outcome(1)),
            docs_failed=failed,
            chunks_written=sum(b.chunks_written for b in branches),
            started_at=started,
            finished_at=self._clock(),
        )
        self._retry(lambda: self._index.record_run(record))
        self._emit_metrics(record)
        return m.RunResult(record, tuple(branches))


def _log_metrics(record: RunRecord) -> None:
    logger.info(
        "Index run finished: run_id=%s mode=%s commit=%s status=%s indexed=%d skipped=%d deleted=%d failed=%d chunks=%d",
        record.run_id, record.mode, record.commit, record.status, record.docs_indexed, record.docs_skipped,
        record.docs_deleted, record.docs_failed, record.chunks_written,
    )
