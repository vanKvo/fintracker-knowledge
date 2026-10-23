"""Shared fixtures: one pgvector container per session, migrated once, rows cleared per test.

__author__ = "Van Vo"
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from testcontainers.postgres import PostgresContainer

from tests.support.migrate import apply_migrations

# Docker Desktop on macOS cannot mount the socket into the Ryuk reaper container; the
# session fixtures below already stop the containers they start.
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")

# pgvector >= 0.8.0 is required for hnsw.iterative_scan (KN-VS-04).
PGVECTOR_IMAGE = "pgvector/pgvector:0.8.0-pg16"


@pytest.fixture(scope="session")
def postgres() -> Iterator[PostgresContainer]:
    with PostgresContainer(PGVECTOR_IMAGE, driver="psycopg", dbname="knowledge") as container:
        yield container


@pytest.fixture(scope="session")
def admin_engine(postgres: PostgresContainer) -> Iterator[Engine]:
    """Superuser engine for migrations, seeding and cleanup. Never used for isolation checks."""
    engine = create_engine(postgres.get_connection_url(), isolation_level="AUTOCOMMIT")
    apply_migrations(engine)
    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def clean_tables(request: pytest.FixtureRequest) -> Iterator[None]:
    yield
    if "admin_engine" in request.fixturenames:
        engine: Engine = request.getfixturevalue("admin_engine")
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE kn_chunks, kn_documents, kn_index_runs CASCADE"))


@pytest.fixture(scope="session")
def app_engine(postgres: PostgresContainer, admin_engine: Engine) -> Iterator[Engine]:
    """Default-isolation engine for transactions that run as a Knowledge role (SET LOCAL ROLE)."""
    engine = create_engine(postgres.get_connection_url())
    yield engine
    engine.dispose()
