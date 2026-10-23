"""KN-CHUNK-04: recall@5 on the golden question set blocks chunking and model changes.

`retrieve` is injected: it returns the source paths of the top-k chunks for a query, so the
gate runs against whatever search KN-RET provides.

__author__ = "Van Vo"
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from src.retrieval_quality.models import (
    MAX_DROP,
    MIN_GOLDEN_QUESTIONS,
    RECALL_K,
    RECALL_THRESHOLD,
    GoldenQuestion,
    GoldenSetTooSmall,
    MissedQuestion,
    QualityGateFailed,
    QualityReport,
)
from src.retrieval_quality.repository import BaselineStore

Retrieve = Callable[[str, int], Sequence[str]]
_EPSILON = 1e-9


def evaluate(questions: Sequence[GoldenQuestion], retrieve: Retrieve, baseline: float | None) -> QualityReport:
    if len(questions) < MIN_GOLDEN_QUESTIONS:
        raise GoldenSetTooSmall(f"The golden set needs at least {MIN_GOLDEN_QUESTIONS} questions.")

    misses: list[MissedQuestion] = []
    for item in questions:
        returned = tuple(retrieve(item.question, RECALL_K))[:RECALL_K]
        if not set(item.expected_paths) & set(returned):
            misses.append(MissedQuestion(item.id, item.question, returned))
    recall = (len(questions) - len(misses)) / len(questions)

    failure = None
    if recall < RECALL_THRESHOLD - _EPSILON:
        failure = f"recall@{RECALL_K} {recall:.3f} is below {RECALL_THRESHOLD:.3f}"
    elif baseline is not None and baseline - recall > MAX_DROP + _EPSILON:
        failure = f"recall@{RECALL_K} {recall:.3f} dropped more than {MAX_DROP:.2f} below the baseline"
    if failure is not None and baseline is not None:
        failure += f" (baseline {baseline:.3f})"
    return QualityReport(recall, baseline, tuple(misses), failure)


def run_quality_gate(questions: Sequence[GoldenQuestion], retrieve: Retrieve, store: BaselineStore) -> QualityReport:
    """Raises QualityGateFailed on a miss; on a pass, the score becomes the new baseline."""
    report = evaluate(questions, retrieve, store.load())
    if report.failure is not None:
        raise QualityGateFailed(report.failure, report)
    store.save(report.recall)
    return report
