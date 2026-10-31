"""KN-FAPI-08: the image builds, runs as a non-root user, holds no secrets or test data, answers `/health`, and
refuses `INTERNAL_AUTH_MODE=local` inside an AWS runtime.

Builds from the monorepo root. Slow, so it runs only where `KNOWLEDGE_IMAGE_TESTS=1` is set (CI, and before a release).

__author__ = "Van Vo"
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[2]
MONOREPO = SERVICE.parents[1]
IMAGE = "fintracker-knowledge-image-test"
ENVIRONMENT = {
    "AWS_REGION": "us-east-1",
    "KNOWLEDGE_DB_MODE": "postgres",
    "KNOWLEDGE_DB_HOST": "127.0.0.1",
    "KNOWLEDGE_DB_PORT": "1",  # nothing listens: the app must still start (`/ready` is what says no)
    "KNOWLEDGE_DB_NAME": "knowledge",
    "KNOWLEDGE_DB_RETRIEVE_USER": "kn_app_retrieve",
    "KNOWLEDGE_DB_RETRIEVE_PASSWORD": "not-a-real-password",
}

pytestmark = pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_IMAGE_TESTS") != "1" or shutil.which("docker") is None or not (MONOREPO / "libs").is_dir(),
    reason="set KNOWLEDGE_IMAGE_TESTS=1 with Docker, inside the monorepo, to build the image",
)


def docker(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def env_flags(extra: dict[str, str] | None = None) -> list[str]:
    flags: list[str] = []
    for name, value in {**ENVIRONMENT, **(extra or {})}.items():
        flags += ["-e", f"{name}={value}"]
    return flags


@pytest.fixture(scope="module", autouse=True)
def image() -> Iterator[str]:
    built = docker("build", "-f", str(SERVICE / "Dockerfile"), "-t", IMAGE, str(MONOREPO), timeout=1200)
    assert built.returncode == 0, built.stderr[-2000:]
    yield IMAGE
    docker("rmi", "-f", IMAGE)


def test_the_process_runs_as_a_non_root_user() -> None:
    result = docker("run", "--rm", "--entrypoint", "id", IMAGE, "-u")
    assert result.returncode == 0 and result.stdout.strip() != "0"


def test_the_image_holds_no_env_file_credential_file_or_test_data() -> None:
    result = docker(
        "run", "--rm", "--entrypoint", "sh", IMAGE, "-c",
        r"find /repo /home /root -name '.env*' -o -name credentials -o -name tests -o -name 'golden*' 2>/dev/null",
    )
    assert result.returncode == 0 and result.stdout.strip() == ""


def test_health_answers_200_within_ten_seconds_even_with_no_database() -> None:
    started = docker("run", "-d", "--rm", "-p", "127.0.0.1::8087", *env_flags(), IMAGE)
    assert started.returncode == 0, started.stderr
    container = started.stdout.strip()
    try:
        port = docker("port", container, "8087/tcp").stdout.strip().rsplit(":", 1)[-1]
        deadline = time.monotonic() + 10
        status = None
        while time.monotonic() < deadline and status != 200:
            try:
                status = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).status
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.3)
        assert status == 200
        with pytest.raises(urllib.error.HTTPError) as not_ready:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=5)
        assert not_ready.value.code == 503
    finally:
        docker("rm", "-f", container)


def test_the_default_command_refuses_local_auth_mode_inside_an_aws_runtime() -> None:
    result = docker("run", "--rm", *env_flags({"INTERNAL_AUTH_MODE": "local", "AWS_LAMBDA_FUNCTION_NAME": "knowledge"}), IMAGE)
    assert result.returncode != 0
    assert "AWS_LAMBDA_FUNCTION_NAME" in result.stderr + result.stdout
