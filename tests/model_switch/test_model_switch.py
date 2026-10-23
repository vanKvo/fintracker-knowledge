"""KN-CHUNK-03: embedding model switch.

__author__ = "Van Vo"
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from src.model_switch.models import ModelSwitchRefused
from src.model_switch.repository import PostgresModelIndex
from src.model_switch.service import switch_embedding_model
from src.retrieval_quality.models import QualityGateFailed
from tests.support.db import insert_chunk, insert_document, transaction_as, unit_vector

OLD = "old.model:1"
NEW = "new.model:2"


class MemorySetting:
    def __init__(self, model: str) -> None:
        self.model = model

    def get(self) -> str:
        return self.model

    def set(self, model: str) -> None:
        self.model = model


def seed(engine: Engine, path: str, model: str) -> None:
    document_id = insert_document(engine, "system", None, path, model)
    insert_chunk(engine, document_id, "system", None, unit_vector(0), model=model)


def models_present(engine: Engine) -> dict[str, set[str]]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT embedding_model, source_path FROM kn_documents")).all()
    present: dict[str, set[str]] = {}
    for model, path in rows:
        present.setdefault(model, set()).add(path)
    return present


def chunk_models(engine: Engine) -> set[str]:
    with engine.connect() as conn:
        return {row[0] for row in conn.execute(text("SELECT DISTINCT embedding_model FROM kn_chunks"))}


@pytest.fixture
def index(app_engine: Engine) -> PostgresModelIndex:
    def connection() -> AbstractContextManager[Connection]:
        return transaction_as(app_engine, "kn_ingest")

    return PostgresModelIndex(connection)


def passing_gate() -> None:
    return None


def failing_gate() -> None:
    raise QualityGateFailed("recall@5 0.800 below 0.900 (baseline 0.950)")


def test_switch_sets_the_new_model_and_deletes_the_old_models_rows(
    admin_engine: Engine, index: PostgresModelIndex
) -> None:
    for path in ("help/a.md", "help/b.md"):
        seed(admin_engine, path, OLD)
        seed(admin_engine, path, NEW)
    setting = MemorySetting(OLD)
    switch_embedding_model(NEW, rebuild_status="succeeded", index=index, setting=setting, quality_gate=passing_gate)
    assert setting.get() == NEW
    assert set(models_present(admin_engine)) == {NEW}
    assert chunk_models(admin_engine) == {NEW}


def test_switch_is_refused_naming_docs_with_no_chunks_under_the_target_model(
    admin_engine: Engine, index: PostgresModelIndex
) -> None:
    seed(admin_engine, "help/a.md", OLD)
    seed(admin_engine, "help/b.md", OLD)
    seed(admin_engine, "help/a.md", NEW)
    setting = MemorySetting(OLD)
    with pytest.raises(ModelSwitchRefused) as raised:
        switch_embedding_model(NEW, rebuild_status="succeeded", index=index, setting=setting, quality_gate=passing_gate)
    assert raised.value.missing_docs == ["help/b.md"]
    assert "help/b.md" in str(raised.value)
    assert setting.get() == OLD
    assert set(models_present(admin_engine)) == {OLD, NEW}


def test_partial_rebuild_is_refused_because_failed_docs_have_no_new_chunks(
    admin_engine: Engine, index: PostgresModelIndex
) -> None:
    seed(admin_engine, "help/a.md", OLD)
    seed(admin_engine, "help/b.md", OLD)
    seed(admin_engine, "help/a.md", NEW)
    setting = MemorySetting(OLD)
    with pytest.raises(ModelSwitchRefused):
        switch_embedding_model(NEW, rebuild_status="partial", index=index, setting=setting, quality_gate=passing_gate)
    assert setting.get() == OLD


def test_failed_rebuild_keeps_the_setting_and_deletes_the_new_models_rows(
    admin_engine: Engine, index: PostgresModelIndex
) -> None:
    seed(admin_engine, "help/a.md", OLD)
    seed(admin_engine, "help/a.md", NEW)
    setting = MemorySetting(OLD)
    with pytest.raises(ModelSwitchRefused):
        switch_embedding_model(NEW, rebuild_status="failed", index=index, setting=setting, quality_gate=passing_gate)
    assert setting.get() == OLD
    assert set(models_present(admin_engine)) == {OLD}
    assert chunk_models(admin_engine) == {OLD}


def test_failed_quality_gate_keeps_the_setting_and_deletes_the_new_models_rows(
    admin_engine: Engine, index: PostgresModelIndex
) -> None:
    seed(admin_engine, "help/a.md", OLD)
    seed(admin_engine, "help/a.md", NEW)
    setting = MemorySetting(OLD)
    with pytest.raises(QualityGateFailed):
        switch_embedding_model(NEW, rebuild_status="succeeded", index=index, setting=setting, quality_gate=failing_gate)
    assert setting.get() == OLD
    assert set(models_present(admin_engine)) == {OLD}
    assert chunk_models(admin_engine) == {OLD}


def test_switching_to_the_model_already_active_changes_nothing(
    admin_engine: Engine, index: PostgresModelIndex
) -> None:
    seed(admin_engine, "help/a.md", OLD)
    setting = MemorySetting(OLD)
    switch_embedding_model(OLD, rebuild_status="succeeded", index=index, setting=setting, quality_gate=failing_gate)
    assert set(models_present(admin_engine)) == {OLD}


def test_the_quality_gate_is_not_run_when_docs_are_missing(admin_engine: Engine, index: PostgresModelIndex) -> None:
    seed(admin_engine, "help/a.md", OLD)
    ran: list[bool] = []
    with pytest.raises(ModelSwitchRefused):
        switch_embedding_model(
            NEW, rebuild_status="succeeded", index=index, setting=MemorySetting(OLD),
            quality_gate=lambda: ran.append(True),
        )
    assert ran == []
