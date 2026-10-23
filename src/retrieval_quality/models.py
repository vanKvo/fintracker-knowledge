"""KN-CHUNK-04: retrieval quality gate types.

__author__ = "Van Vo"
"""

from __future__ import annotations

from dataclasses import dataclass

RECALL_K = 5
RECALL_THRESHOLD = 0.90
MAX_DROP = 0.05
MIN_GOLDEN_QUESTIONS = 30


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    expected_paths: tuple[str, ...]


@dataclass(frozen=True)
class MissedQuestion:
    question_id: str
    question: str
    returned_paths: tuple[str, ...]


@dataclass(frozen=True)
class QualityReport:
    recall: float
    baseline: float | None
    misses: tuple[MissedQuestion, ...]
    failure: str | None  # None when the gate passes

    @property
    def passed(self) -> bool:
        return self.failure is None


class GoldenSetTooSmall(ValueError):
    pass


class QualityGateFailed(Exception):
    """Carries the score and baseline in its message; never question or chunk text."""

    def __init__(self, message: str, report: QualityReport | None = None) -> None:
        super().__init__(message)
        self.report = report
