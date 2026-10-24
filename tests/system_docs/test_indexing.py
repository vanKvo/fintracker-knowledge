"""KN-SDOC-02, -03, -04: change detection, atomic replace and delete, failure isolation and retry.

__author__ = "Van Vo"
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import AbstractContextManager

import pytest
from sqlalchemy.engine import Connection, Engine

from src.chunking.schemas import Chunk
from src.core.exceptions import EmbeddingUnavailableError
from src.embedding.schemas import EmbeddedChunk
from src.system_docs.repository import PostgresDocumentIndex
from src.system_docs.schemas import Manifest
from src.system_docs.service import SystemDocsIndexer
from tests.support.db import DIMENSIONS, EMBEDDING_MODEL, transaction_as
from tests.system_docs.fakes import (
    COMMIT,
    NOW,
    SECRET,
    AlwaysUnavailable,
    FakeEmbedder,
    FakeSource,
    Sleeps,
    doc,
    indexed_docs,
    make_index,
    searchable_texts,
)

PATH = "help/statements/csv-mapping.md"


def build(
    app_engine: Engine,
    source: FakeSource,
    embedder: FakeEmbedder | None = None,
    sleeps: Sleeps | None = None,
    **index_kwargs: object,
) -> tuple[SystemDocsIndexer, FakeEmbedder, Sleeps]:
    embedder = embedder or FakeEmbedder()
    sleeps = sleeps or Sleeps()
    indexer = SystemDocsIndexer(
        source,
        make_index(app_engine, **index_kwargs),  # type: ignore[arg-type]
        lambda model: embedder,
        active_model=EMBEDDING_MODEL,
        sleep=sleeps,
        clock=lambda: NOW,
    )
    return indexer, embedder, sleeps


def manifest(changed: list[str], deleted: list[str] | None = None, commit: str = COMMIT) -> Manifest:
    return Manifest(commit, "2026-10-20T18:04:00Z", tuple(changed), tuple(deleted or []))


def outcomes(result) -> dict[str, str]:  # type: ignore[no-untyped-def]
    return {branch.path: branch.outcome for branch in result.branches}


# --- KN-SDOC-02 ---------------------------------------------------------------------------


def test_each_changed_path_is_its_own_branch_ending_indexed_or_skipped(app_engine: Engine, admin_engine: Engine) -> None:
    source = FakeSource({"help/a.md": doc("a"), "help/b.md": doc("b"), "help/c.md": doc("c")})
    indexer, _, _ = build(app_engine, source)
    indexer.run_manifest(manifest(["help/a.md"]))

    result = indexer.run_manifest(manifest(["help/a.md", "help/b.md", "help/c.md"], commit="aaaaaaa"))

    assert outcomes(result) == {"help/a.md": "skipped", "help/b.md": "indexed", "help/c.md": "indexed"}


def test_unchanged_hash_is_skipped_without_embedding_calls(app_engine: Engine, admin_engine: Engine) -> None:
    source = FakeSource({PATH: doc("a")})
    indexer, embedder, _ = build(app_engine, source)
    indexer.run_manifest(manifest([PATH]))
    calls_after_first = embedder.calls

    result = indexer.run_manifest(manifest([PATH], commit="bbbbbbb"))

    assert outcomes(result) == {PATH: "skipped"}
    assert embedder.calls == calls_after_first
    assert indexed_docs(admin_engine)[(PATH, EMBEDDING_MODEL)]["source_commit"] == COMMIT


def test_hash_is_the_sha256_of_the_uploaded_bytes(app_engine: Engine, admin_engine: Engine) -> None:
    content = doc("a")
    indexer, _, _ = build(app_engine, FakeSource({PATH: content}))

    indexer.run_manifest(manifest([PATH]))

    assert indexed_docs(admin_engine)[(PATH, EMBEDDING_MODEL)]["content_hash"] == hashlib.sha256(content).hexdigest()


def test_missing_source_fails_its_branch_and_other_branches_continue(app_engine: Engine, admin_engine: Engine) -> None:
    source = FakeSource({"help/b.md": doc("b")})
    indexer, _, sleeps = build(app_engine, source)

    result = indexer.run_manifest(manifest(["help/missing.md", "help/b.md"]))

    assert outcomes(result) == {"help/missing.md": "failed: source_missing", "help/b.md": "indexed"}
    assert sleeps.seconds == []  # a missing file is not retried


@pytest.mark.parametrize("content", [b"\xff\xfe not utf-8", b"x" * (200 * 1024 + 1)], ids=["not-utf8", "too-large"])
def test_unreadable_source_fails_as_invalid_source(app_engine: Engine, content: bytes) -> None:
    indexer, embedder, _ = build(app_engine, FakeSource({PATH: content}))

    result = indexer.run_manifest(manifest([PATH]))

    assert outcomes(result) == {PATH: "failed: invalid_source"}
    assert embedder.calls == 0


def test_at_most_four_branches_run_at_once(app_engine: Engine) -> None:
    paths = [f"help/{n}.md" for n in range(12)]
    source = FakeSource({p: doc(p) for p in paths}, delay=0.05)
    indexer, _, _ = build(app_engine, source)

    indexer.run_manifest(manifest(paths))

    assert 2 <= source.max_active <= 4


# --- KN-SDOC-03 ---------------------------------------------------------------------------


def test_new_version_replaces_the_old_one_for_search(app_engine: Engine, admin_engine: Engine) -> None:
    source = FakeSource({PATH: doc("old", sections=3)})
    indexer, _, _ = build(app_engine, source)
    indexer.run_manifest(manifest([PATH]))
    source.files[PATH] = doc("new", sections=2)

    result = indexer.run_manifest(manifest([PATH], commit="bbbbbbb"))

    assert outcomes(result) == {PATH: "indexed"}
    texts = searchable_texts(app_engine, PATH)
    assert len(texts) == 2 and all("new text" in t for t in texts)
    row = indexed_docs(admin_engine)[(PATH, EMBEDDING_MODEL)]
    assert (row["source_commit"], row["chunk_count"], row["scope"], row["user_id"]) == ("bbbbbbb", 2, "system", None)


def test_concurrent_search_sees_only_the_old_version_until_commit(app_engine: Engine) -> None:
    source = FakeSource({PATH: doc("old")})
    indexer, _, _ = build(app_engine, source)
    indexer.run_manifest(manifest([PATH]))
    seen_during: list[list[str]] = []
    source.files[PATH] = doc("new")
    indexer, _, _ = build(app_engine, source, before_commit=lambda: seen_during.append(searchable_texts(app_engine, PATH)))

    indexer.run_manifest(manifest([PATH], commit="bbbbbbb"))

    # Transactions in order: hash check, replace, run record. A search taken just before the replace
    # commits still sees the whole old version; once it has committed, the whole new one.
    hash_check, replace, run_record = seen_during
    assert hash_check == replace and replace and all("old text" in t for t in replace)
    assert run_record and all("new text" in t for t in run_record)


def test_deleted_path_removes_its_document_and_chunks(app_engine: Engine, admin_engine: Engine) -> None:
    source = FakeSource({PATH: doc("a")})
    indexer, _, _ = build(app_engine, source)
    indexer.run_manifest(manifest([PATH]))

    result = indexer.run_manifest(manifest([], [PATH], commit="bbbbbbb"))

    assert outcomes(result) == {PATH: "deleted: 1"}
    assert indexed_docs(admin_engine) == {}
    assert searchable_texts(app_engine, PATH) == []


def test_deleting_a_path_that_is_not_indexed_is_a_noop(app_engine: Engine) -> None:
    indexer, _, _ = build(app_engine, FakeSource())

    result = indexer.run_manifest(manifest([], ["help/never-indexed.md"]))

    assert outcomes(result) == {"help/never-indexed.md": "deleted: 0"}


def test_failed_transaction_commits_nothing_and_old_version_stays_searchable(
    app_engine: Engine, admin_engine: Engine
) -> None:
    source = FakeSource({PATH: doc("old")})
    indexer, _, _ = build(app_engine, source)
    indexer.run_manifest(manifest([PATH]))
    source.files[PATH] = doc("new")

    class WrongSizeVectors(FakeEmbedder):
        def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
            return [EmbeddedChunk(c, (1.0,) * (DIMENSIONS - 1), self.model) for c in chunks]

    indexer, _, _ = build(app_engine, source, WrongSizeVectors())
    result = indexer.run_manifest(manifest([PATH], commit="bbbbbbb"))

    assert outcomes(result) == {PATH: "failed: database_unavailable"}
    assert all("old text" in t for t in searchable_texts(app_engine, PATH))
    assert indexed_docs(admin_engine)[(PATH, EMBEDDING_MODEL)]["source_commit"] == COMMIT


# --- KN-SDOC-04 ---------------------------------------------------------------------------


def test_transient_embedding_failure_is_retried_and_the_doc_is_indexed(app_engine: Engine) -> None:
    embedder = FakeEmbedder(failures=[EmbeddingUnavailableError()])
    indexer, _, sleeps = build(app_engine, FakeSource({PATH: doc("a")}), embedder)

    result = indexer.run_manifest(manifest([PATH]))

    assert outcomes(result) == {PATH: "indexed"}
    assert sleeps.seconds == [2]


def test_retries_back_off_two_four_eight_seconds_then_fail(app_engine: Engine) -> None:
    embedder = AlwaysUnavailable()
    indexer, _, sleeps = build(app_engine, FakeSource({PATH: doc("a")}), embedder)

    result = indexer.run_manifest(manifest([PATH]))

    assert outcomes(result) == {PATH: "failed: embedding_unavailable"}
    assert sleeps.seconds == [2, 4, 8]
    assert embedder.calls == 4


def test_one_permanent_failure_leaves_four_docs_indexed_and_the_run_partial(app_engine: Engine, admin_engine: Engine) -> None:
    paths = [f"help/{n}.md" for n in range(5)]
    source = FakeSource({p: doc(p) for p in paths})

    class FailsFor(FakeEmbedder):
        def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
            if "help/3.md" in chunks[0].source_path:
                raise EmbeddingUnavailableError()
            return super().embed_chunks(chunks)

    indexer, _, _ = build(app_engine, source, FailsFor())
    result = indexer.run_manifest(manifest(paths))

    assert sorted(b.outcome for b in result.branches).count("indexed") == 4
    assert outcomes(result)["help/3.md"] == "failed: embedding_unavailable"
    assert result.record.status == "partial"
    assert len(indexed_docs(admin_engine)) == 4


def test_permanently_failed_doc_keeps_its_previous_version_searchable(app_engine: Engine) -> None:
    source = FakeSource({PATH: doc("old")})
    indexer, _, _ = build(app_engine, source)
    indexer.run_manifest(manifest([PATH]))
    source.files[PATH] = doc("new")
    indexer, _, _ = build(app_engine, source, AlwaysUnavailable())

    result = indexer.run_manifest(manifest([PATH], commit="bbbbbbb"))

    assert result.record.status == "failed"
    assert all("old text" in t for t in searchable_texts(app_engine, PATH))


def test_database_resuming_is_retried_like_any_transient_error(app_engine: Engine) -> None:
    from sqlalchemy.exc import OperationalError

    opened: list[int] = []

    def resuming_then_ready() -> AbstractContextManager[Connection]:
        opened.append(1)
        if len(opened) == 1:
            raise OperationalError("resuming", {}, Exception("database is resuming"))
        return transaction_as(app_engine, "kn_ingest")

    sleeps = Sleeps()
    indexer = SystemDocsIndexer(
        FakeSource({PATH: doc("a")}),
        PostgresDocumentIndex(resuming_then_ready),
        lambda model: FakeEmbedder(),
        active_model=EMBEDDING_MODEL,
        sleep=sleeps,
        clock=lambda: NOW,
    )

    result = indexer.run_manifest(manifest([PATH]))

    assert outcomes(result) == {PATH: "indexed"}
    assert sleeps.seconds[0] >= 2


def test_logs_carry_paths_and_codes_but_never_doc_text(app_engine: Engine, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    source = FakeSource({PATH: doc(SECRET), "help/b.md": doc(SECRET)})
    embedder = AlwaysUnavailable()
    indexer, _, _ = build(app_engine, source, embedder)

    indexer.run_manifest(manifest([PATH, "help/b.md"]))

    assert SECRET not in caplog.text
    assert PATH in caplog.text and "embedding_unavailable" in caplog.text
