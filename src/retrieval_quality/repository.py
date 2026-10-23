"""Golden question set and last-accepted baseline, both files in the Knowledge repo.

__author__ = "Van Vo"
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.retrieval_quality.models import GoldenQuestion


def load_golden_questions(path: Path) -> list[GoldenQuestion]:
    entries = yaml.safe_load(path.read_text())
    return [GoldenQuestion(e["id"], e["question"], tuple(e["expected_paths"])) for e in entries]


class BaselineStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> float | None:
        if not self._path.exists():
            return None
        return float(json.loads(self._path.read_text())["recall_at_5"])

    def save(self, recall: float) -> None:
        self._path.write_text(json.dumps({"recall_at_5": recall}, indent=2) + "\n")
