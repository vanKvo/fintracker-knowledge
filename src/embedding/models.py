"""Embedding results and errors (KN-CHUNK-02).

__author__ = "Van Vo"
"""

from __future__ import annotations

from dataclasses import dataclass

from src.chunking.models import Chunk


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: Chunk
    embedding: tuple[float, ...]
    embedding_model: str


class EmbeddingUnavailable(Exception):
    """Bedrock failed after retries. The message is generic and safe to return outward."""

    code = "embedding_unavailable"

    def __init__(self) -> None:
        super().__init__("Embedding is temporarily unavailable.")


class EmbeddingInputTooLong(ValueError):
    """The text exceeds the model's input limit; rejected before any call."""

    def __init__(self) -> None:
        super().__init__("Text is too long to embed.")
