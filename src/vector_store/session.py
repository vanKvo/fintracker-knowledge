"""SQL that every Knowledge database session runs inside its own transaction.

KN-VS-03: the RDS Data API makes each call its own session, so the user context is only
visible to queries in the same transaction (set_config's third argument = transaction-local).
KN-VS-04: filtered HNSW searches need iterative scans, or the filter can leave fewer than k rows.

__author__ = "Van Vo"
"""

from __future__ import annotations

USER_CONTEXT_SETTING = "app.current_user_id"

# Named parameter :uid, as accepted by the Data API and SQLAlchemy.
SET_USER_CONTEXT_SQL = f"SELECT set_config('{USER_CONTEXT_SETTING}', :uid, true)"

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
HNSW_EF_SEARCH = 100
HNSW_ITERATIVE_SCAN = "relaxed_order"

SEARCH_SESSION_SQL: tuple[str, ...] = (
    f"SELECT set_config('hnsw.ef_search', '{HNSW_EF_SEARCH}', true)",
    f"SELECT set_config('hnsw.iterative_scan', '{HNSW_ITERATIVE_SCAN}', true)",
)
