"""KN-CHUNK-03: embedding model switch errors.

__author__ = "Van Vo"
"""

from __future__ import annotations


class ModelSwitchRefused(Exception):
    """The switch did not happen. Names doc paths only, never text."""

    def __init__(self, message: str, missing_docs: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing_docs = missing_docs or []
