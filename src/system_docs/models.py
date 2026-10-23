"""Types for system-docs indexing (KN-SDOC).

Branch outcomes are the Appendix B strings. Exceptions carry generic messages only, never doc
text or the underlying error.

__author__ = "Van Vo"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

RunStatus = Literal["succeeded", "partial", "failed"]
RunMode = Literal["incremental", "rebuild"]

INDEXED = "indexed"
SKIPPED = "skipped"
SOURCE_MISSING = "failed: source_missing"
EMBEDDING_UNAVAILABLE = "failed: embedding_unavailable"
DATABASE_UNAVAILABLE = "failed: database_unavailable"
INVALID_SOURCE = "failed: invalid_source"


def deleted_outcome(count: int) -> str:
    return f"deleted: {count}"


@dataclass(frozen=True)
class Manifest:
    """Appendix A."""

    commit: str
    merged_at: str
    changed: tuple[str, ...]
    deleted: tuple[str, ...]


@dataclass(frozen=True)
class BranchResult:
    path: str
    outcome: str
    chunks_written: int = 0


@dataclass(frozen=True)
class RunRecord:
    """Appendix C."""

    run_id: str
    mode: RunMode
    commit: str
    embedding_model: str
    status: RunStatus
    docs_indexed: int
    docs_skipped: int
    docs_deleted: int
    docs_failed: int
    chunks_written: int
    started_at: datetime
    finished_at: datetime


@dataclass(frozen=True)
class RunResult:
    record: RunRecord
    branches: tuple[BranchResult, ...]


@dataclass(frozen=True)
class IndexStatus:
    commit: str | None
    finished_at: datetime | None


class InvalidManifest(ValueError):
    def __init__(self) -> None:
        super().__init__("The manifest is not valid.")


class SourceMissing(Exception):
    def __init__(self) -> None:
        super().__init__("The source file is not available.")


class InvalidSource(Exception):
    def __init__(self) -> None:
        super().__init__("The source file is not valid.")


class DatabaseUnavailable(Exception):
    def __init__(self) -> None:
        super().__init__("The database is temporarily unavailable.")
