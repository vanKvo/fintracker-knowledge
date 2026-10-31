"""Fixtures for the retrieval tests.

__author__ = "Van Vo"
"""

from __future__ import annotations

import pytest

from tests.retrieval.fakes import CALLER_ARN


@pytest.fixture(autouse=True)
def internal_http_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """LB-IHTTP-01: strict mode, with the AI assistant's role as the one allowed caller."""
    for name in ("INTERNAL_AUTH_MODE", "AWS_EXECUTION_ENV", "AWS_LAMBDA_FUNCTION_NAME", "ECS_CONTAINER_METADATA_URI_V4"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("INTERNAL_ALLOWED_CALLER_ARNS", CALLER_ARN)
