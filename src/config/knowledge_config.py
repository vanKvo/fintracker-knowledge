"""Chunking and embedding settings (KN-CHUNK Appendix A).

Values are config, rules are spec: changing any default needs KN-CHUNK-04 to pass.

__author__ = "Van Vo"
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    split_headings: tuple[int, ...] = (1, 2, 3)
    max_chunk_tokens: int = 600
    min_chunk_tokens: int = 50
    overlap_tokens: int = 60  # paragraph splits only


@dataclass(frozen=True)
class EmbeddingConfig:
    # KN-CHUNK-03: ingestion and retrieval both read this; no code hard-codes a model ID.
    active_embedding_model: str = "amazon.titan-embed-text-v2:0"
    # Must match vector(1024) in db/migration V1; a change needs a new migration.
    embedding_dimensions: int = 1024
    embedding_normalize: bool = True
    max_input_tokens: int = 8192  # Titan Text Embeddings V2 input limit
    retry_backoff_seconds: tuple[float, ...] = (1, 2, 4)  # KN-CHUNK-02: 3 retries
