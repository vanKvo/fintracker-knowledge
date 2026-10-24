"""Embedding results and errors (KN-CHUNK-02).

__author__ = "Van Vo"
"""

from __future__ import annotations

from dataclasses import dataclass

from src.chunking.schemas import Chunk


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: Chunk
    embedding: tuple[float, ...]
    embedding_model: str
