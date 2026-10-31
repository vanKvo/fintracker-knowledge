"""KN-FAPI-07, -09: the local stack keeps secrets out of git and wires the one-shot steps in order.

__author__ = "Van Vo"
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

SERVICE = Path(__file__).resolve().parents[2]
MONOREPO = SERVICE.parents[1]
COMPOSE = SERVICE / "docker-compose.yml"
# Files that describe the stack. A password written into one of them would be committed.
STACK_FILES = [COMPOSE, SERVICE / "Dockerfile", *SERVICE.glob("db/**/*.sql"), *SERVICE.glob("local/**/*.sh"), *SERVICE.glob("local/**/*.sql")]
LITERAL_PASSWORD = re.compile(r"""(?ix)
    (password\w*\s*[:=]\s*['"]?[^\s'"$?-]     # PASSWORD: value / PASSWORD=value (not ${...}, ${X:?msg} or empty)
    |password\s+'[^']+')                     # SQL: PASSWORD 'value'
    """)


def compose() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    return loaded


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=SERVICE, capture_output=True, text=True)


def test_the_local_env_file_is_ignored_and_not_tracked() -> None:
    if git("rev-parse", "--is-inside-work-tree").returncode != 0:
        pytest.skip("not a git checkout")
    assert git("check-ignore", "-q", ".env.local").returncode == 0
    assert git("ls-files", ".env.local").stdout.strip() == ""


def test_no_password_literal_appears_in_a_compose_sql_script_or_dockerfile() -> None:
    offenders = [
        f"{path.relative_to(SERVICE)}:{number}"
        for path in STACK_FILES
        if path.is_file()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if not line.lstrip().startswith(("#", "--")) and LITERAL_PASSWORD.search(line)
    ]
    assert offenders == []


def test_an_example_env_file_documents_every_secret_the_stack_reads() -> None:
    example = (SERVICE / ".env.local.example").read_text(encoding="utf-8")
    for name in ("KNOWLEDGE_DB_ADMIN_PASSWORD", "FLYWAY_PASSWORD", "KNOWLEDGE_DB_RETRIEVE_PASSWORD", "KNOWLEDGE_DB_INGEST_PASSWORD"):
        assert re.search(rf"^{name}=", example, re.MULTILINE), name


def test_the_app_and_the_index_command_wait_for_the_migration_step_to_finish_successfully() -> None:
    services = compose()["services"]
    for name in ("knowledge-api", "knowledge-index"):
        assert services[name]["depends_on"]["knowledge-migrate"]["condition"] == "service_completed_successfully"
    assert "depends_on" not in services["knowledge-migrate"]  # the database is the host's, not a container
    assert services["knowledge-migrate"].get("restart", "no") == "no"


def test_the_index_command_is_on_demand_and_reads_the_recorded_snapshot_read_only() -> None:
    index = compose()["services"]["knowledge-index"]
    assert "index" in index.get("profiles", [])
    assert any(str(v).endswith(":/rag:ro") for v in index["volumes"])
    assert "/rag/corpus" in index["command"]  # the snapshot corpus, not the live docs


def test_the_stack_holds_no_aws_credentials_and_embeds_from_the_recording() -> None:
    """KN-CHUNK-05: nothing in the local stack can start a paid Bedrock call."""
    services = compose()["services"]
    for name in ("knowledge-api", "knowledge-index"):
        spec = services[name]
        environment = spec["environment"]
        assert environment["KNOWLEDGE_EMBEDDER"] == "replay" and environment["KNOWLEDGE_REPLAY_DIR"] == "/rag"
        assert not any(".aws" in str(v) for v in spec["volumes"]), "no host AWS profile in the container"
        assert not set(environment) & {"AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"}
        assert "AWS_ENDPOINT_URL" not in environment  # it would send Bedrock calls to a local stand-in
    assert ".aws" not in COMPOSE.read_text(encoding="utf-8").replace("~/.aws", "")


def test_local_auth_mode_and_the_caller_allow_list_are_set_in_the_compose_file_only() -> None:
    environment = compose()["services"]["knowledge-api"]["environment"]
    assert environment["INTERNAL_AUTH_MODE"] == "local"
    assert environment["INTERNAL_ALLOWED_CALLER_ARNS"]  # so the 403 path stays testable
    dockerfile = [line for line in (SERVICE / "Dockerfile").read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    assert not any("INTERNAL_AUTH_MODE" in line for line in dockerfile)  # the image never sets it
    setters = re.compile(r"""environ(\[|\.setdefault\(|\.update\()[^\n]*INTERNAL_AUTH_MODE""")
    assert [p for p in SERVICE.glob("src/**/*.py") if setters.search(p.read_text(encoding="utf-8"))] == []


def test_the_stack_uses_the_hosts_postgres_and_starts_no_postgres_container() -> None:
    services = compose()["services"]
    assert not any("postgres" in (spec.get("image") or "") and "flyway" not in spec["image"] for spec in services.values())
    assert "host.docker.internal:5432/knowledge" in " ".join(services["knowledge-migrate"]["command"])
    for name in ("knowledge-api", "knowledge-index"):
        assert services[name]["environment"]["KNOWLEDGE_DB_HOST"] == "host.docker.internal"
        assert "host.docker.internal:host-gateway" in services[name]["extra_hosts"]


def test_the_one_time_database_setup_script_exists_and_is_executable() -> None:
    script = SERVICE / "local" / "setup-local-database.sh"
    assert script.is_file() and script.stat().st_mode & 0o111


def test_the_root_compose_file_includes_this_stack() -> None:
    root = MONOREPO / "docker-compose.yml"
    if not root.is_file():
        pytest.skip("not inside the monorepo")
    assert "services/fintracker-knowledge/docker-compose.yml" in root.read_text(encoding="utf-8")
