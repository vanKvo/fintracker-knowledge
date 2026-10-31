"""KN-FAPI-06: `python -m src.system_docs.cli index|rebuild|status` for a local stack.

The command runs for real: settings, the local-host guard, the folder source, the indexer. Only its edges are fake:
the database index (in memory) and Bedrock.

__author__ = "Van Vo"
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest

from src.system_docs.cli import CliEdges, main, resolve_commit
from tests.retrieval.app_support import TEST_ENVIRONMENT
from tests.system_docs.fakes import AlwaysUnavailable, FailingFor, FakeEmbedder, MemoryIndex, Sleeps, doc

INGEST = {"KNOWLEDGE_DB_INGEST_USER": "kn_app_ingest", "KNOWLEDGE_DB_INGEST_PASSWORD": "not-a-real-password"}
# The command needs the ingest role only; the retrieve credentials belong to the API.
ENVIRONMENT = {**{k: v for k, v in TEST_ENVIRONMENT.items() if "RETRIEVE" not in k}, **INGEST}
MARKER = "BROKEN-DOC"


def write(root: Path, relative: str, content: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    write(tmp_path, "help/accounts.md", doc("accounts"))
    write(tmp_path, "help/statements/csv-mapping.md", doc("csv"))
    write(tmp_path, "help/budgets.md", doc("budgets"))
    write(tmp_path, "architecture/design.md", doc("not-help"))
    write(tmp_path, "README.md", doc("not-help-either"))
    return tmp_path


class Run:
    def __init__(self, index: MemoryIndex, embedder: FakeEmbedder) -> None:
        self.index = index
        self.embedder = embedder
        self.out = io.StringIO()

    def __call__(self, *argv: str, environment: dict[str, str] | None = None) -> int:
        edges = CliEdges(
            index=lambda settings: self.index,
            embedder=lambda settings, model: self.embedder,
            sleep=Sleeps(),
            out=self.out,
        )
        return main(list(argv), environ=environment or ENVIRONMENT, edges=edges)

    @property
    def text(self) -> str:
        return self.out.getvalue()


def runner(embedder: FakeEmbedder | None = None) -> Run:
    return Run(MemoryIndex(), embedder or FakeEmbedder())


def test_indexing_a_folder_writes_every_help_document_and_prints_the_counts(docs: Path) -> None:
    run = runner()
    assert run("index", "--root", str(docs), "--commit", "abc1234") == 0
    assert {path for path, _ in run.index.documents} == {"help/accounts.md", "help/statements/csv-mapping.md", "help/budgets.md"}
    assert all(commit == "abc1234" for _, commit, _ in run.index.documents.values())
    assert "documents seen: 3" in run.text and "skipped unchanged: 0" in run.text and "failures: 0" in run.text
    chunks = sum(count for _, _, count in run.index.documents.values())
    assert chunks > 0 and f"chunks written: {chunks}" in run.text


def test_a_second_run_with_no_changes_writes_nothing_and_skips_every_document(docs: Path) -> None:
    run = runner()
    run("index", "--root", str(docs), "--commit", "abc1234")
    calls = run.embedder.calls
    run.out.truncate(0)
    run.out.seek(0)

    assert run("index", "--root", str(docs), "--commit", "abc1234") == 0

    assert "chunks written: 0" in run.text and "skipped unchanged: 3" in run.text
    assert run.embedder.calls == calls  # nothing was embedded again, so nothing was paid for


def test_a_changed_file_replaces_only_its_own_chunks(docs: Path) -> None:
    run = runner()
    run("index", "--root", str(docs), "--commit", "abc1234")
    before = dict(run.index.documents)
    write(docs, "help/accounts.md", doc("accounts changed", sections=3))

    assert run("index", "--root", str(docs), "--commit", "def5678") == 0

    after = run.index.documents
    changed = {key for key in after if after[key] != before[key]}
    assert {path for path, _ in changed} == {"help/accounts.md"}
    assert after[("help/accounts.md", "amazon.titan-embed-text-v2:0")][1] == "def5678"


def test_only_help_markdown_is_read_and_other_files_under_the_root_are_ignored(docs: Path) -> None:
    run = runner()
    run("index", "--root", str(docs), "--commit", "abc1234")
    assert not any(path.startswith(("architecture", "README")) for path, _ in run.index.documents)
    assert not any("not-help" in text for text in run.embedder.texts)


def test_a_bedrock_failure_keeps_the_previous_chunks_exits_non_zero_and_names_the_document(docs: Path) -> None:
    index = MemoryIndex()
    Run(index, FakeEmbedder())("index", "--root", str(docs), "--commit", "abc1234")
    before = index.documents[("help/accounts.md", "amazon.titan-embed-text-v2:0")]
    write(docs, "help/accounts.md", doc(f"{MARKER} new text"))
    run = Run(index, FailingFor(MARKER))

    code = run("index", "--root", str(docs), "--commit", "def5678")

    assert code == 1
    assert "help/accounts.md" in run.text and "embedding_unavailable" in run.text and "failures: 1" in run.text
    assert index.documents[("help/accounts.md", "amazon.titan-embed-text-v2:0")] == before  # still the old version


def test_a_total_bedrock_outage_exits_non_zero_after_the_retries(docs: Path) -> None:
    run = runner(AlwaysUnavailable())
    assert run("index", "--root", str(docs), "--commit", "abc1234") == 1
    assert "failures: 3" in run.text


@pytest.mark.parametrize("host", ["db.cluster-abc.us-east-1.rds.amazonaws.com", "8.8.8.8", "example.com", "10.1.2.3", "192.168.1.5"])
def test_a_database_host_that_is_not_local_exits_before_any_database_or_bedrock_call(docs: Path, host: str) -> None:
    calls: list[str] = []
    edges = CliEdges(
        index=lambda settings: calls.append("index"),  # type: ignore[arg-type,return-value,func-returns-value]
        embedder=lambda settings, model: calls.append("bedrock"),  # type: ignore[arg-type,return-value,func-returns-value]
        sleep=Sleeps(),
        out=io.StringIO(),
    )

    code = main(["index", "--root", str(docs)], environ={**ENVIRONMENT, "KNOWLEDGE_DB_HOST": host}, edges=edges)

    assert code == 2 and calls == []
    assert host not in edges.out.getvalue()  # type: ignore[attr-defined]


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1", "host.docker.internal", "knowledge-postgres"])
def test_local_hosts_are_accepted(docs: Path, host: str) -> None:
    run = runner()
    assert run("index", "--root", str(docs), "--commit", "c", environment={**ENVIRONMENT, "KNOWLEDGE_DB_HOST": host}) == 0


def test_the_command_runs_only_against_a_direct_connection_and_names_the_variable_when_it_does_not(docs: Path) -> None:
    run = runner()
    environment = {**ENVIRONMENT, "KNOWLEDGE_DB_MODE": "data_api"}
    assert run("index", "--root", str(docs), environment=environment) == 2
    assert "KNOWLEDGE_DB_MODE" in run.text


def test_a_missing_ingest_credential_stops_the_command_naming_the_variable_and_never_a_value(docs: Path) -> None:
    run = runner()
    environment = {k: v for k, v in ENVIRONMENT.items() if k != "KNOWLEDGE_DB_INGEST_PASSWORD"}
    assert run("index", "--root", str(docs), environment=environment) == 2
    assert "KNOWLEDGE_DB_INGEST_PASSWORD" in run.text and "not-a-real-password" not in run.text


def test_a_root_without_a_help_folder_is_an_error_not_an_empty_success(tmp_path: Path) -> None:
    run = runner()
    assert run("index", "--root", str(tmp_path)) == 2
    assert "help/" in run.text


def test_rebuild_embeds_every_document_again_even_when_unchanged(docs: Path) -> None:
    run = runner()
    run("index", "--root", str(docs), "--commit", "abc1234")
    calls = run.embedder.calls

    assert run("rebuild", "--root", str(docs), "--commit", "abc1234") == 0

    assert run.embedder.calls == calls + 3 and "skipped unchanged: 0" in run.text


def test_status_prints_the_document_and_chunk_counts_and_the_last_run_commit(docs: Path) -> None:
    run = runner()
    run("index", "--root", str(docs), "--commit", "abc1234")
    run.out.truncate(0)
    run.out.seek(0)

    assert run("status") == 0

    assert "documents: 3" in run.text and "last indexed commit: abc1234" in run.text


# --- the commit label ---


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=root, check=True, capture_output=True
    )


def test_the_default_commit_is_the_short_commit_and_gets_dirty_when_the_docs_have_uncommitted_changes(tmp_path: Path) -> None:
    write(tmp_path, "help/a.md", doc("a"))
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "docs")
    short = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()

    assert resolve_commit(tmp_path, None) == short

    write(tmp_path, "help/a.md", doc("a changed"))
    assert resolve_commit(tmp_path, None) == f"{short}-dirty"
    assert resolve_commit(tmp_path, "given") == "given"


def test_outside_a_git_repository_the_commit_label_says_so(tmp_path: Path) -> None:
    write(tmp_path, "help/a.md", doc("a"))
    assert resolve_commit(tmp_path, None) == "local-unversioned"

