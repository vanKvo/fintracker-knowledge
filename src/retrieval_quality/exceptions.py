"""Retrieval quality gate errors (KN-CHUNK-04).

__author__ = "Van Vo"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval_quality.schemas import QualityReport


class GoldenSetTooSmall(ValueError):
    pass


class QualityGateFailed(Exception):
    """Carries the score and baseline in its message; never question or chunk text."""

    def __init__(self, message: str, report: QualityReport | None = None) -> None:
        super().__init__(message)
        self.report = report
