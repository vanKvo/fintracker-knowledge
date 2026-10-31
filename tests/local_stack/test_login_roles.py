"""KN-FAPI-07: the init script creates login roles that RLS applies to, in a database booted the way compose boots it.

The container is started with `local/init` mounted as the image's init directory, then migrated by the admin user.

__author__ = "Van Vo"
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError
from testcontainers.postgres import PostgresContainer

from tests.conftest import PGVECTOR_IMAGE
from tests.support.db import insert_chunk, insert_document, unit_vector
from tests.support.migrate import apply_migrations

INIT = Path(__file__).resolve().parents[2] / "local" / "init"
RETRIEVE_PASSWORD = "retrieve-test-password"
INGEST_PASSWORD = "ingest-test-password"
USER_A = "11111111-1111-4111-8111-111111111111"


@pytest.fixture(scope="module")
def stack() -> Iterator[PostgresContainer]:
    container = (
        PostgresContainer(PGVECTOR_IMAGE, driver="psycopg", dbname="knowledge")
        .with_volume_mapping(str(INIT), "/docker-entrypoint-initdb.d", "ro")
        .with_env("KNOWLEDGE_DB_RETRIEVE_PASSWORD", RETRIEVE_PASSWORD)
        .with_env("KNOWLEDGE_DB_INGEST_PASSWORD", INGEST_PASSWORD)
    )
    with container:
        admin = create_engine(container.get_connection_url(), isolation_level="AUTOCOMMIT")
        apply_migrations(admin)
        admin.dispose()
        yield container


@pytest.fixture
def admin(stack: PostgresContainer) -> Iterator[Engine]:
    engine = create_engine(stack.get_connection_url(), isolation_level="AUTOCOMMIT")
    yield engine
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE kn_chunks, kn_documents CASCADE"))
    engine.dispose()


def login(stack: PostgresContainer, user: str, password: str) -> Engine:
    host, port = stack.get_container_host_ip(), stack.get_exposed_port(5432)
    return create_engine(f"postgresql+psycopg://{user}:{password}@{host}:{port}/knowledge")


def test_the_login_roles_exist_cannot_bypass_rls_and_belong_to_their_nologin_roles(admin: Engine) -> None:
    with admin.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT r.rolname, r.rolsuper, r.rolbypassrls, r.rolcanlogin, "
                "(SELECT array_agg(g.rolname) FROM pg_auth_members m JOIN pg_roles g ON g.oid = m.roleid "
                " WHERE m.member = r.oid) "
                "FROM pg_roles r WHERE r.rolname IN ('kn_app_retrieve', 'kn_app_ingest') ORDER BY 1"
            )
        ).all()
    assert [(r[0], r[1], r[2], r[3], r[4]) for r in rows] == [
        ("kn_app_ingest", False, False, True, ["kn_ingest"]),
        ("kn_app_retrieve", False, False, True, ["kn_retrieve"]),
    ]


def test_the_retrieve_login_sees_no_user_rows_without_a_user_context_and_cannot_read_the_run_log(
    stack: PostgresContainer, admin: Engine
) -> None:
    system = insert_document(admin, "system", None, "help/a.md")
    insert_chunk(admin, system, "system", None, unit_vector(0), chunk_text="system chunk")
    private = insert_document(admin, "user", USER_A, "user/a.md")
    insert_chunk(admin, private, "user", USER_A, unit_vector(0), chunk_text="private chunk")
    engine = login(stack, "kn_app_retrieve", RETRIEVE_PASSWORD)

    with engine.connect() as conn:
        scopes = {r[0] for r in conn.execute(text("SELECT DISTINCT scope FROM kn_chunks"))}
    with engine.connect() as conn, pytest.raises(ProgrammingError):
        conn.execute(text("SELECT * FROM kn_index_runs"))

    assert scopes == {"system"}  # no app.current_user_id: the user row is invisible
    engine.dispose()


def test_the_ingest_login_can_write_system_documents(stack: PostgresContainer, admin: Engine) -> None:
    engine = login(stack, "kn_app_ingest", INGEST_PASSWORD)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO kn_documents (document_id, scope, user_id, source_path, content_hash, embedding_model, "
                "chunk_count, indexed_at) VALUES (gen_random_uuid(), 'system', NULL, 'help/x.md', repeat('b', 64), 'm', 0, now())"
            )
        )
    engine.dispose()


def test_the_init_script_refuses_to_create_roles_without_a_password() -> None:
    """A missing `.env.local` must not produce a role with an empty or default password."""
    import subprocess

    result = subprocess.run(
        ["sh", str(INIT / "10-roles.sh")],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "KNOWLEDGE_DB_RETRIEVE_PASSWORD" in result.stderr
