"""Rows of `kn_index_runs` (KN-SDOC Appendix C; app doc 9.1: stored records).

__author__ = "Van Vo"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.system_docs.schemas import RunMode, RunStatus


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
class IndexStatus:
    commit: str | None
    finished_at: datetime | None
