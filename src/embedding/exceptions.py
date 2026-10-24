"""Embedding errors. Bedrock failing after retries is the shared `EmbeddingUnavailableError`
(core/exceptions.py), since retrieval returns it as is and indexing records it as an outcome.

__author__ = "Van Vo"
"""

from __future__ import annotations


class EmbeddingInputTooLong(ValueError):
    """The text exceeds the model's input limit; rejected before any call."""

    def __init__(self) -> None:
        super().__init__("Text is too long to embed.")
