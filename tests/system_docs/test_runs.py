"""KN-SDOC-05, -06: full rebuild, run record, index status and alarms.

__author__ = "Van Vo"
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.system_docs.access import CallerNotAllowed, authorize_caller
from src.system_docs.models import IndexStatus, RunRecord
from src.system_docs.service import SystemDocsIndexer
from src.system_docs.status import get_index_status, run_alarm, stale_alarm
from tests.support.db import EMBEDDING_MODEL, transaction_as
from tests.system_docs.fakes import (
    COMMIT,
    NEW_MODEL,
    NOW,
    FakeEmbedder,
    FakeSource,
    Sleeps,
    doc,
    indexed_docs,
    make_index,
    searchable_texts,
)

PATH = "help/statements/csv-mapping.md"


def build(app_engine: Engine, source: FakeSource, embedders: dict[str, FakeEmbedder] | None = None):  # type: ignore[no-untyped-def]
    embedders = embedders or {}
    metrics: list[RunRecord] = []
    indexer = SystemDocsIndexer(
        source,
        make_index(app_engine),
        lambda model: embedders.setdefault(model, FakeEmbedder(model)),
        active_model=EMBEDDING_MODEL,
        sleep=Sleeps(),
        clock=lambda: NOW,
        emit_metrics=metrics.append,
    )
    return indexer, metrics


def runs(engine: Engine) -> list[dict[str, object]]:
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(text("SELECT * FROM kn_index_runs ORDER BY started_at")).mappings()]


# --- KN-SDOC-05 ---------------------------------------------------------------------------


def test_rebuild_indexes_every_doc_under_the_requested_model(app_engine: Engine, admin_engine: Engine) -> None:
    source = FakeSource({f"help/{n}.md": doc(str(n)) for n in range(40)})
    indexer, _ = build(app_engine, source)

    result = indexer.run_rebuild(COMMIT, NEW_MODEL)

    assert result.record.status == "succeeded" and result.record.mode == "rebuild"
    assert result.record.docs_indexed == 40 and result.record.embedding_model == NEW_MODEL
    assert {model for _, model in indexed_docs(admin_engine)} == {NEW_MODEL}
    assert len(runs(admin_engine)) == 1


def test_rebuild_with_a_new_model_writes_alongside_the_old_chunks(app_engine: Engine, admin_engine: Engine) -> None:
    source = FakeSource({PATH: doc("a")})
    indexer, _ = build(app_engine, source)
    indexer.run_manifest(_manifest([PATH]))

    indexer.run_rebuild(COMMIT, NEW_MODEL)

    assert set(indexed_docs(admin_engine)) == {(PATH, EMBEDDING_MODEL), (PATH, NEW_MODEL)}


def test_rebuild_with_the_active_model_ignores_the_hash_and_replaces_in_place(
    app_engine: Engine, admin_engine: Engine
) -> None:
    source = FakeSource({PATH: doc("a")})
    indexer, embedder_calls = build(app_engine, source)
    indexer.run_manifest(_manifest([PATH]))

    result = indexer.run_rebuild("bbbbbbb", EMBEDDING_MODEL)

    assert [b.outcome for b in result.branches] == ["indexed"]
    assert list(indexed_docs(admin_engine)) == [(PATH, EMBEDDING_MODEL)]
    assert indexed_docs(admin_engine)[(PATH, EMBEDDING_MODEL)]["source_commit"] == "bbbbbbb"
    assert len(searchable_texts(app_engine, PATH)) == 2


def test_rebuild_with_a_failed_doc_ends_partial(app_engine: Engine) -> None:
    source = FakeSource({"help/a.md": doc("a"), "help/bad.md": b"\xff\xfe"})
    indexer, _ = build(app_engine, source)

    result = indexer.run_rebuild(COMMIT, NEW_MODEL)

    assert result.record.status == "partial" and result.record.docs_failed == 1


def test_rebuild_is_limited_to_allow_listed_callers() -> None:
    allowed = ("arn:aws:iam::1:role/ci-deploy", "arn:aws:iam::1:role/dev-admin")

    authorize_caller("arn:aws:iam::1:role/dev-admin", allowed)
    for caller in ("arn:aws:iam::1:role/ai-assistant", "", None):
        with pytest.raises(CallerNotAllowed):
            authorize_caller(caller, allowed)
    with pytest.raises(CallerNotAllowed):
        authorize_caller("arn:aws:iam::1:role/ci-deploy", ())  # empty allow-list fails closed


# --- KN-SDOC-06 ---------------------------------------------------------------------------


def test_finished_run_records_commit_status_and_counts(app_engine: Engine, admin_engine: Engine) -> None:
    source = FakeSource({"help/a.md": doc("a"), "help/b.md": doc("b"), "help/c.md": doc("c", 3)})
    indexer, metrics = build(app_engine, source)
    indexer.run_manifest(_manifest(["help/a.md"], commit="aaaaaaa"))

    indexer.run_manifest(_manifest(["help/a.md", "help/b.md", "help/c.md", "help/gone.md"], ["help/old.md"]))

    row = runs(admin_engine)[-1]
    assert (row["mode"], row["commit"], row["status"], row["embedding_model"]) == (
        "incremental",
        COMMIT,
        "partial",
        EMBEDDING_MODEL,
    )
    assert (row["docs_indexed"], row["docs_skipped"], row["docs_deleted"], row["docs_failed"]) == (2, 1, 0, 1)
    assert row["chunks_written"] == 5
    assert row["started_at"] <= row["finished_at"]
    assert len(metrics) == 2 and metrics[-1].docs_failed == 1


def test_index_status_returns_last_successful_commit_and_time(app_engine: Engine) -> None:
    source = FakeSource({"help/a.md": doc("a")})
    indexer, _ = build(app_engine, source)
    indexer.run_manifest(_manifest(["help/a.md"], commit="aaaaaaa"))
    indexer.run_manifest(_manifest(["help/missing.md"], commit="bbbbbbb"))  # all failed: not a success

    status = get_index_status(make_index(app_engine), "arn:aws:iam::1:role/knowledge", ("arn:aws:iam::1:role/knowledge",))

    assert status == IndexStatus("aaaaaaa", NOW)


def test_index_status_is_empty_before_any_run_and_denied_to_other_callers(app_engine: Engine) -> None:
    index = make_index(app_engine)
    allowed = ("arn:aws:iam::1:role/knowledge",)

    assert get_index_status(index, allowed[0], allowed) == IndexStatus(None, None)
    with pytest.raises(CallerNotAllowed):
        get_index_status(index, "arn:aws:iam::1:role/ai-assistant", allowed)


def test_alarm_fires_for_partial_or_failed_runs_only() -> None:
    assert run_alarm("partial") is True
    assert run_alarm("failed") is True
    assert run_alarm("succeeded") is False


def test_alarm_fires_when_no_success_follows_a_manifest_within_thirty_minutes() -> None:
    manifest_at = NOW

    assert stale_alarm(manifest_at, None, NOW + timedelta(minutes=29)) is False
    assert stale_alarm(manifest_at, None, NOW + timedelta(minutes=31)) is True
    assert stale_alarm(manifest_at, NOW - timedelta(minutes=5), NOW + timedelta(minutes=31)) is True  # older success
    assert stale_alarm(manifest_at, NOW + timedelta(minutes=8), NOW + timedelta(minutes=31)) is False


def test_run_table_is_writable_by_ingest_only(app_engine: Engine) -> None:
    from sqlalchemy.exc import DBAPIError

    with transaction_as(app_engine, "kn_ingest") as conn:
        assert conn.execute(text("SELECT count(*) FROM kn_index_runs")).scalar_one() == 0
    with pytest.raises(DBAPIError):
        with transaction_as(app_engine, "kn_retrieve") as conn:
            conn.execute(text("SELECT count(*) FROM kn_index_runs"))


def _manifest(changed: list[str], deleted: list[str] | None = None, commit: str = COMMIT):  # type: ignore[no-untyped-def]
    from src.system_docs.models import Manifest

    return Manifest(commit, "2026-10-20T18:04:00Z", tuple(changed), tuple(deleted or []))
