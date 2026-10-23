"""Retrieve request, hits, response and errors (KN-RET Appendices A-C).

__author__ = "Van Vo"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_QUERY_CHARS = 1000
DEFAULT_K = 5
MIN_K, MAX_K = 1, 10
_ALLOWED_FIELDS = frozenset({"query", "scope", "k"})

# Appendix C: code -> (HTTP status, generic message). Messages never carry exception text.
_ERRORS: dict[str, tuple[int, str]] = {
    "invalid_argument": (400, "The request is not valid."),
    "scope_not_enabled": (400, "This scope is not available yet."),
    "user_context_missing": (401, "The request has no verified user context."),
    "caller_not_allowed": (403, "The caller is not allowed to do this."),
    "embedding_unavailable": (503, "Embedding is temporarily unavailable."),
    "database_unavailable": (503, "The database is temporarily unavailable."),
    "timeout": (504, "The request took too long."),
    "internal_error": (500, "Something went wrong."),
}


class RetrievalError(Exception):
    """A documented failure. `message` is generic and safe to return outward; `field` names the argument at fault."""

    def __init__(self, code: str, field: str | None = None) -> None:
        self.code = code
        self.status, base = _ERRORS[code]
        self.message = f"{base[:-1]}: {field}." if field else base
        super().__init__(self.message)


class DatabaseUnavailable(Exception):
    def __init__(self) -> None:
        super().__init__("The database is temporarily unavailable.")


@dataclass(frozen=True)
class RetrieveRequest:
    query: str
    scope: str
    k: int


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
        raise RetrievalError("invalid_argument", "body")
    extra = set(body) - _ALLOWED_FIELDS
    if extra:
        raise RetrievalError("invalid_argument", "request")  # the unknown field's name is input, so not echoed

    query = body.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
        raise RetrievalError("invalid_argument", "query")

    scope = body.get("scope")
    if scope == "mine":
        raise RetrievalError("scope_not_enabled")  # until KN-UDOC ships
    if scope != "system":
        raise RetrievalError("invalid_argument", "scope")

    k = body.get("k", DEFAULT_K)
    if type(k) is not int or not MIN_K <= k <= MAX_K:
        raise RetrievalError("invalid_argument", "k")
    return RetrieveRequest(query, scope, k)
