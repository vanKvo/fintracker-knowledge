"""KN-FAPI-09: checks the local stack through HTTP (and one direct database check for Row-Level Security).

    python scripts/smoke.py            # after `docker compose up -d` and the `knowledge-index` job

Reads the passwords from .env.local. Makes ONE Bedrock call (the query embedding) through the app. It stops with a clear
message, not a stack trace, when the stack is down or the host has no Bedrock access. The seeded row is removed at the end.

__author__ = "Van Vo"
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

BASE_URL = os.environ.get("KNOWLEDGE_URL", "http://localhost:8087")
CALLER = "arn:local:ai-assistant"  # INTERNAL_ALLOWED_CALLER_ARNS in docker-compose.yml
OTHER_CALLER = "arn:local:someone-else"
USER_A = "00000000-0000-4000-8000-00000000000a"
USER_B = "00000000-0000-4000-8000-00000000000b"
QUESTION = "How do I confirm a CSV column mapping?"
EXPECTED_SOURCE = "help/statements/csv-mapping.md"
HEADER_CALLER = "X-Internal-Caller-Arn"
HEADER_USER = "X-Internal-User-Id"


class Stop(Exception):
    """A check failed or the stack cannot be checked. The message is for the person running the script."""


def call(method: str, path: str, *, caller: str | None = CALLER, user: str | None = USER_B, body: Any = None) -> tuple[int, Any]:
    headers = {"content-type": "application/json"}
    if caller:
        headers[HEADER_CALLER] = caller
    if user:
        headers[HEADER_USER] = user
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(BASE_URL + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"null")
    except (urllib.error.URLError, OSError):
        raise Stop(f"Cannot reach {BASE_URL}. Is the stack up? (docker compose up -d)") from None


def expect(label: str, ok: bool) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {label}")
    if not ok:
        raise Stop(f"Check failed: {label}")


def retrieve(user: str) -> tuple[int, Any]:
    return call("POST", "/internal/retrieve", user=user, body={"query": QUESTION, "scope": "system", "k": 5})


def main() -> int:
    try:
        status, _ = call("GET", "/health", caller=None, user=None)
        expect("/health is 200", status == 200)
        status, body = call("GET", "/ready", caller=None, user=None)
        if status == 503 and (body or {}).get("code") == "embedding_unavailable":
            raise Stop("embedding_unavailable: the app has no AWS credentials for Bedrock. Mount a profile with access to Titan Text Embeddings V2.")
        expect("/ready is 200 (database and embedding client)", status == 200)

        status, body = call("POST", "/internal/retrieve", caller=None, body={"query": "x", "scope": "system"})
        expect("no caller is 401 unauthorized", (status, (body or {}).get("code")) == (401, "unauthorized"))
        status, body = call("POST", "/internal/retrieve", caller=OTHER_CALLER, body={"query": "x", "scope": "system"})
        expect("a caller off the list is 403 caller_not_allowed", (status, (body or {}).get("code")) == (403, "caller_not_allowed"))

        seeded = seed_user_row()
        try:
            status, body = retrieve(USER_B)
            if status == 503 and (body or {}).get("code") == "embedding_unavailable":
                raise Stop("embedding_unavailable: the query could not be embedded. Check the host's Bedrock access and region.")
            expect("retrieval is 200", status == 200)
            chunks = body["chunks"]
            expect(f"the first source is {EXPECTED_SOURCE} (run the knowledge-index job first)",
                   bool(chunks) and chunks[0]["source_path"] == EXPECTED_SOURCE)
            expect("user A's seeded row is absent from user B's results", all(seeded["marker"] not in c["text"] for c in chunks))
            rows_for_b, rows_for_a = visible_user_rows(USER_B), visible_user_rows(USER_A)
            expect("as the app's database role, user B sees none of user A's rows and user A sees their own",
                   (rows_for_b, rows_for_a) == (0, 1))
        finally:
            remove_seeded_rows(seeded["document_id"])
    except Stop as stop:
        print(f"\n{stop}", file=sys.stderr)
        return 1
    print("\nAll checks passed.")
    return 0


# --- direct database access: the admin seeds, the app's own login role reads --------------------------------------


def env_value(name: str, *, allow_empty: bool = False) -> str:
    if name in os.environ:
        return os.environ[name]
    file = Path(__file__).resolve().parents[1] / ".env.local"
    if file.is_file():
        for line in file.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and (value.strip() or allow_empty):
                return value.strip()
    raise Stop(f"{name} is not set (put it in .env.local).")


def connect(user: str, password: str) -> Any:
    try:
        import psycopg
    except ImportError:
        raise Stop("psycopg is not installed: run this with `poetry run python scripts/smoke.py`.") from None
    try:
        return psycopg.connect(
            host=os.environ.get("KNOWLEDGE_SMOKE_DB_HOST", "127.0.0.1"),
            port=int(os.environ.get("KNOWLEDGE_SMOKE_DB_PORT", "5432")),
            dbname="knowledge", user=user, password=password, connect_timeout=5, autocommit=True,
        )
    except psycopg.OperationalError:
        raise Stop("Cannot connect to the local database on port 5432 (see docker-compose.yml).") from None


def seed_user_row() -> dict[str, str]:
    """A user-scope chunk for user A with the same vector as an indexed system chunk, so it would rank first if it leaked."""
    document_id, marker = str(uuid.uuid4()), f"SMOKE-{uuid.uuid4().hex[:8]}"
    with connect(env_value("KNOWLEDGE_DB_ADMIN_USER"), env_value("KNOWLEDGE_DB_ADMIN_PASSWORD", allow_empty=True)) as conn:
        row = conn.execute("SELECT embedding, embedding_model FROM kn_chunks WHERE scope = 'system' LIMIT 1").fetchone()
        if row is None:
            raise Stop("No indexed help docs. Run: docker compose --profile index run --rm knowledge-index")
        conn.execute(
            "INSERT INTO kn_documents (document_id, scope, user_id, source_path, content_hash, embedding_model, chunk_count, indexed_at) "
            "VALUES (%s, 'user', %s, 'smoke/user-a.md', repeat('c', 64), %s, 1, now())", (document_id, USER_A, row[1]))
        conn.execute(
            "INSERT INTO kn_chunks (chunk_id, document_id, scope, user_id, ordinal, heading_path, text, token_count, embedding, embedding_model) "
            "VALUES (gen_random_uuid(), %s, 'user', %s, 0, 'Smoke', %s, 5, %s, %s)", (document_id, USER_A, marker, row[0], row[1]))
    return {"document_id": document_id, "marker": marker}


def visible_user_rows(user_id: str) -> int:
    with connect("kn_app_retrieve", env_value("KNOWLEDGE_DB_RETRIEVE_PASSWORD")) as conn, conn.transaction():
        conn.execute("SELECT set_config('app.current_user_id', %s, true)", (user_id,))
        row = conn.execute("SELECT count(*) FROM kn_chunks WHERE scope = 'user' AND text LIKE 'SMOKE-%%'").fetchone()
    return int(row[0]) if row else -1


def remove_seeded_rows(document_id: str) -> None:
    with connect(env_value("KNOWLEDGE_DB_ADMIN_USER"), env_value("KNOWLEDGE_DB_ADMIN_PASSWORD", allow_empty=True)) as conn:
        conn.execute("DELETE FROM kn_documents WHERE document_id = %s", (document_id,))  # chunks cascade


if __name__ == "__main__":
    raise SystemExit(main())
