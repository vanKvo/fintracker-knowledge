"""Types for system-docs indexing (KN-SDOC).

Branch outcomes are the Appendix B strings. Errors are in `exceptions.py`.

__author__ = "Van Vo"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.system_docs.models import RunRecord

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
class RunResult:
    record: RunRecord
    branches: tuple[BranchResult, ...]
