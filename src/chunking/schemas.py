"""Chunk produced by KN-CHUNK-01.

__author__ = "Van Vo"
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    source_path: str
    heading_path: str  # e.g. "Statements > CSV mapping"
    text: str
    ordinal: int  # position in the document, from 0
    token_count: int
