"""KN-CHUNK-04: retrieval quality gate.

__author__ = "Van Vo"
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.retrieval_quality.exceptions import GoldenSetTooSmall, QualityGateFailed
from src.retrieval_quality.repository import BaselineStore, load_golden_questions
from src.retrieval_quality.schemas import GoldenQuestion
from src.retrieval_quality.service import evaluate, run_quality_gate


def questions(count: int) -> list[GoldenQuestion]:
    return [GoldenQuestion(f"GQ-{i:03}", f"question {i}?", (f"help/doc{i}.md",)) for i in range(count)]


def retriever(missed: set[str] = frozenset()):  # type: ignore[no-untyped-def]
    """Returns the expected path for every question except the missed ones."""

    def retrieve(question: str, k: int) -> list[str]:
        index = question.split()[1].rstrip("?")
        if f"GQ-{int(index):03}" in missed:
            return ["help/other-1.md", "help/other-2.md"]
        return ["help/noise.md", f"help/doc{index}.md"]

    return retrieve


def test_recall_above_threshold_with_no_drop_passes_and_becomes_the_baseline(tmp_path: Path) -> None:
    store = BaselineStore(tmp_path / "baseline.json")
    store.save(0.93)
    report = run_quality_gate(questions(30), retriever({"GQ-000", "GQ-001"}), store)  # 28/30 = 0.933
    assert report.passed and report.recall == pytest.approx(28 / 30)
    assert store.load() == pytest.approx(28 / 30)


def test_first_run_without_a_baseline_passes_on_the_threshold_alone(tmp_path: Path) -> None:
    store = BaselineStore(tmp_path / "baseline.json")
    assert store.load() is None
    assert run_quality_gate(questions(30), retriever(), store).passed
    assert store.load() == 1.0


def test_a_question_is_a_hit_when_any_expected_path_is_in_the_top_k() -> None:
    qs = [GoldenQuestion(f"GQ-{i:03}", f"question {i}?", ("help/x.md", f"help/doc{i}.md")) for i in range(30)]
    assert evaluate(qs, retriever(), baseline=None).recall == 1.0


def test_paths_beyond_the_top_k_do_not_count() -> None:
    qs = [GoldenQuestion(f"GQ-{i:03}", "q?", ("help/e.md",)) for i in range(30)]
    seen: list[int] = []

    def retrieve(question: str, k: int) -> list[str]:
        seen.append(k)
        return ["a", "b", "c", "d", "e", "help/e.md"]

    assert evaluate(qs, retrieve, baseline=None).recall == 0.0
    assert set(seen) == {5}


def test_report_lists_each_missed_question_with_the_paths_returned_instead() -> None:
    report = evaluate(questions(30), retriever({"GQ-004"}), baseline=None)
    assert [(m.question_id, m.returned_paths) for m in report.misses] == [
        ("GQ-004", ("help/other-1.md", "help/other-2.md"))
    ]


def test_recall_below_threshold_fails_with_score_and_baseline_in_the_message(tmp_path: Path) -> None:
    store = BaselineStore(tmp_path / "baseline.json")
    store.save(0.95)
    with pytest.raises(QualityGateFailed) as raised:
        run_quality_gate(questions(30), retriever({f"GQ-{i:03}" for i in range(4)}), store)  # 26/30 = 0.867
    assert "0.867" in str(raised.value) and "0.950" in str(raised.value)
    assert store.load() == 0.95, "a failed run must not replace the baseline"


def test_drop_of_more_than_five_points_from_the_baseline_fails_even_above_threshold(tmp_path: Path) -> None:
    store = BaselineStore(tmp_path / "baseline.json")
    store.save(1.0)
    with pytest.raises(QualityGateFailed):
        run_quality_gate(questions(40), retriever({"GQ-000", "GQ-001", "GQ-002"}), store)  # 37/40 = 0.925, drop 0.075


def test_drop_of_exactly_five_points_passes(tmp_path: Path) -> None:
    store = BaselineStore(tmp_path / "baseline.json")
    store.save(1.0)
    assert run_quality_gate(questions(40), retriever({"GQ-000", "GQ-001"}), store).passed  # 38/40 = 0.95


def test_golden_set_under_thirty_questions_is_refused() -> None:
    with pytest.raises(GoldenSetTooSmall):
        evaluate(questions(29), retriever(), baseline=None)


def test_golden_questions_load_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "golden-questions.yaml"
    path.write_text(
        '- id: GQ-001\n  question: "How do I confirm a CSV column mapping?"\n'
        "  expected_paths:\n    - help/statements/csv-mapping.md\n"
    )
    assert load_golden_questions(path) == [
        GoldenQuestion("GQ-001", "How do I confirm a CSV column mapping?", ("help/statements/csv-mapping.md",))
    ]
