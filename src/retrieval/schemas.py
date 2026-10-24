"""Retrieve request, hits and response (KN-RET Appendices A, B). Errors are the shared ones in
`core/exceptions.py` (Appendix C).

__author__ = "Van Vo"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.exceptions import InvalidArgumentError, ScopeNotEnabledError

MAX_QUERY_CHARS = 1000
DEFAULT_K = 5
MIN_K, MAX_K = 1, 10
_ALLOWED_FIELDS = frozenset({"query", "scope", "k"})

@dataclass(frozen=True)
class RetrieveRequest:
    query: str
    scope: str
    k: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    source_path: str
    heading_path: str
    score: float


@dataclass(frozen=True)
class RetrieveResponse:
    chunks: list[RetrievedChunk]
    embedding_model: str
    index_commit: str

    def to_dict(self) -> dict[str, Any]:
        """Appendix B. Built field by field so nothing else can leak into the body."""
        return {
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "source_path": c.source_path,
                    "heading_path": c.heading_path,
                    "score": c.score,
                }
                for c in self.chunks
            ],
            "embedding_model": self.embedding_model,
            "index_commit": self.index_commit,
        }


def parse_request(body: Any) -> RetrieveRequest:
    """Appendix A, strict: unknown fields, wrong types and out-of-range values are all rejected."""
    if not isinstance(body, dict):
        raise InvalidArgumentError.for_field("body")
    extra = set(body) - _ALLOWED_FIELDS
    if extra:
        raise InvalidArgumentError.for_field("request")  # the unknown field's name is input, so not echoed

    query = body.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
        raise InvalidArgumentError.for_field("query")

    scope = body.get("scope")
    if scope == "mine":
        raise ScopeNotEnabledError()  # until KN-UDOC ships
    if scope != "system":
        raise InvalidArgumentError.for_field("scope")

    k = body.get("k", DEFAULT_K)
    if type(k) is not int or not MIN_K <= k <= MAX_K:
        raise InvalidArgumentError.for_field("k")
    return RetrieveRequest(query, scope, k)
