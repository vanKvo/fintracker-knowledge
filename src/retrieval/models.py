"""Rows the vector search reads from `kn_chunks` / `kn_documents` (app doc 9.1: stored records).

__author__ = "Van Vo"
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchHit:
    """One row from the vector search, with the fields the ownership re-check needs (never returned as-is)."""

    chunk_id: str
    document_id: str
    scope: str
    user_id: str | None
    source_path: str
    heading_path: str
    text: str
    distance: float


@dataclass(frozen=True)
class SearchResult:
    hits: list[SearchHit]
    index_commit: str | None
