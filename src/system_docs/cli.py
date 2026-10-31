"""KN-FAPI-06: index help docs from a folder into a local database.

    python -m src.system_docs.cli index   --root docs [--commit LABEL]   # changed documents only
    python -m src.system_docs.cli rebuild --root docs [--commit LABEL]   # every document again
    python -m src.system_docs.cli status

Reads `<root>/help/**/*.md` as the `kn_ingest` role over a direct connection. It refuses any database host that is not
local before it opens a connection or calls anything.

Embedding follows KNOWLEDGE_EMBEDDER (KN-CHUNK-05). `replay` (the local stack) answers from the recording in
KNOWLEDGE_REPLAY_DIR with no credentials, so only the snapshot corpus can be indexed. `bedrock` makes paid calls and starts
only for the repo owner: a terminal, a named AWS profile (RAG_EVAL_AWS_PROFILE) and no agent environment (owner_guard.py).
Exit codes: 0 done, 1 at least one document failed (each is named), 2 refused or misconfigured. Output carries paths and
counts only, never document text.

__author__ = "Van Vo"
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from fintracker_internal_http import InternalConfigError

from src.config.knowledge_config import EmbeddingConfig
from src.core.config import ConfigError, KnowledgeSettings, load_knowledge_settings
from src.core.exceptions import DatabaseUnavailableError
from src.system_docs import schemas as m
from src.system_docs.repository import DocumentIndex, PostgresDocumentIndex
from src.system_docs.service import DocEmbedder, SystemDocsIndexer
from src.system_docs.source import HELP_DIRECTORY, LocalDocSource
from src.vector_store.connections import PostgresConnections, UnsafeDatabaseRoleError, postgres_engine

LOCAL_HOST_NAMES = frozenset({"localhost", "host.docker.internal"})
UNVERSIONED = "local-unversioned"


@dataclass
class CliEdges:
    """What the command reaches outside itself: the database, Bedrock, the clock and the output."""

    index: Callable[[KnowledgeSettings], DocumentIndex]
    embedder: Callable[[KnowledgeSettings, str], DocEmbedder]
    sleep: Callable[[float], None] = time.sleep
    out: TextIO = field(default_factory=lambda: sys.stdout)
    # Why real Bedrock calls may not start, or None. Tests never reach Bedrock, so their default allows.
    bedrock_refusal: Callable[[], str | None] = lambda: None


def is_local_host(host: str) -> bool:
    """Loopback, Docker's host alias, or a single-label name (a compose service on the local network)."""
    name = host.strip().lower().strip("[]")
    if name in LOCAL_HOST_NAMES:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return bool(name) and "." not in name


def resolve_commit(root: Path, given: str | None) -> str:
    """`--commit`, else the repository's short commit, with `-dirty` when the help docs have uncommitted changes."""
    if given:
        return given
    git = shutil.which("git")
    if git is None:
        return UNVERSIONED
    head = _run([git, "-C", str(root), "rev-parse", "--short", "HEAD"])
    if head is None:
        return UNVERSIONED
    changed = _run([git, "-C", str(root), "status", "--porcelain", "--", HELP_DIRECTORY])
    return f"{head}-dirty" if changed else head


def _run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None, edges: CliEdges | None = None) -> int:
    args = _parser().parse_args(argv)
    edges = edges or _real_edges()
    out = edges.out
    try:
        settings = load_knowledge_settings(environ, role="ingest")
    except (ConfigError, InternalConfigError) as error:  # both messages name the variable and never a value
        print(f"Configuration error: {error}", file=out)
        return 2
    if not is_local_host(settings.database.host):
        print("Refusing to run: KNOWLEDGE_DB_HOST is not a local database host.", file=out)
        return 2
    if settings.embedder == "bedrock" and args.command != "status":
        refusal = edges.bedrock_refusal()
        if refusal:
            print(refusal, file=out)
            return 2
    try:
        index = edges.index(settings)
    except (UnsafeDatabaseRoleError, DatabaseUnavailableError) as error:
        print(f"Cannot use the database: {error if isinstance(error, UnsafeDatabaseRoleError) else 'it is not reachable'}", file=out)
        return 2
    if args.command == "status":
        return _status(index, out)

    root = Path(args.root)
    source = LocalDocSource(root)
    if not source.list_paths():
        print(f"No documents found: expected Markdown files under {root}/{HELP_DIRECTORY}/.", file=out)
        return 2
    commit = resolve_commit(root, args.commit)
    model = EmbeddingConfig().active_embedding_model
    indexer = SystemDocsIndexer(
        source, index, lambda name: edges.embedder(settings, name), active_model=model, sleep=edges.sleep
    )
    if args.command == "rebuild":
        result = indexer.run_rebuild(commit, model)
    else:
        manifest = m.Manifest(commit, datetime.now(UTC).isoformat(), tuple(source.list_paths()), ())
        result = indexer.run_manifest(manifest)
    return _report(result, commit, out)


def _report(result: m.RunResult, commit: str, out: TextIO) -> int:
    record = result.record
    print(f"commit: {commit}", file=out)
    print(f"documents seen: {len(result.branches)}", file=out)
    print(f"chunks written: {record.chunks_written}", file=out)
    print(f"skipped unchanged: {record.docs_skipped}", file=out)
    print(f"failures: {record.docs_failed}", file=out)
    for branch in result.branches:
        if branch.outcome.startswith("failed"):
            print(f"FAILED {branch.path}: {branch.outcome}", file=out)
    return 1 if record.docs_failed else 0


def _status(index: DocumentIndex, out: TextIO) -> int:
    documents, chunks = index.counts()
    last = index.last_succeeded_run()
    print(f"documents: {documents}", file=out)
    print(f"chunks: {chunks}", file=out)
    print(f"last indexed commit: {last.commit or 'none'}", file=out)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.system_docs.cli", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("index", "index changed documents"), ("rebuild", "index every document again")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--root", required=True, help="folder that holds help/")
        command.add_argument("--commit", help="label stored with the documents (default: git short commit)")
    commands.add_parser("status", help="print document and chunk counts")
    return parser


def _real_edges() -> CliEdges:
    from src.core.container import build_replay_client
    from src.embedding.service import Embedder
    from src.rag_eval import owner_guard
    from src.rag_eval.live import build_live_client

    clients: dict[str, object] = {}

    def index(settings: KnowledgeSettings) -> DocumentIndex:
        database = settings.database
        connections = PostgresConnections(
            postgres_engine(database.host, database.port, database.name, database.user, database.password)
        )
        connections.assert_role_is_safe()  # a superuser would write through Row-Level Security unchecked
        return PostgresDocumentIndex(connections.connection)

    def embedder(settings: KnowledgeSettings, model: str) -> DocEmbedder:
        if "client" not in clients:
            if settings.embedder == "replay":
                clients["client"] = build_replay_client(settings)
            else:  # the refusal below already ran in main(); this is the owner's own session
                clients["client"] = build_live_client(owner_guard.check_live_allowed(os.environ), settings.region)
        return Embedder(clients["client"], EmbeddingConfig(active_embedding_model=model))  # type: ignore[arg-type]

    def bedrock_refusal() -> str | None:
        try:
            owner_guard.check_live_allowed(os.environ)
        except owner_guard.LiveRunRefused as refusal:
            return str(refusal)
        return None

    return CliEdges(index=index, embedder=embedder, bedrock_refusal=bedrock_refusal)


if __name__ == "__main__":
    from src.core.logging_config import configure_logging

    configure_logging()
    raise SystemExit(main())
