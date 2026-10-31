"""KN-FAPI-05: the Bedrock client is built from configuration, and a missing credential is a fast 503.

__author__ = "Van Vo"
"""

from __future__ import annotations

import time

import pytest
from botocore.exceptions import NoCredentialsError

from src.app_factory import create_app
from src.config.knowledge_config import EmbeddingConfig
from src.core.config import load_knowledge_settings
from src.core.container import Container
from tests.embedding.test_embedder import FakeBedrock, unit
from tests.retrieval.app_support import BODY, TEST_ENVIRONMENT, AppClient
from tests.retrieval.fakes import CALLER_ARN


@pytest.fixture(autouse=True)
def isolated_aws(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    empty = tmp_path_factory.mktemp("aws")
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(empty / "credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(empty / "config"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("INTERNAL_ALLOWED_CALLER_ARNS", CALLER_ARN)
    monkeypatch.delenv("INTERNAL_AUTH_MODE", raising=False)


def test_the_client_uses_the_configured_region_short_timeouts_and_no_sdk_retries() -> None:
    client = Container(load_knowledge_settings({**TEST_ENVIRONMENT, "AWS_REGION": "eu-west-1"})).bedrock_client()
    config = client.meta.config  # type: ignore[attr-defined]
    assert client.meta.region_name == "eu-west-1"  # type: ignore[attr-defined]
    assert config.connect_timeout + config.read_timeout <= 3.0  # fits KN-RET-07's budget
    assert config.retries["total_max_attempts"] == 1  # KN-CHUNK-02 owns the retries


def test_the_model_id_sent_to_bedrock_is_the_configured_one_never_a_container_constant() -> None:
    fake = FakeBedrock(unit(0))
    config = EmbeddingConfig(active_embedding_model="test.model:9")
    container = Container(
        load_knowledge_settings(TEST_ENVIRONMENT), bedrock_client_factory=lambda *_: fake, embedding_config=config  # type: ignore[arg-type]
    )
    container.embedder().embed_query("How do I confirm a CSV mapping?")
    assert fake.calls[0]["modelId"] == "test.model:9"


def test_without_credentials_the_first_query_is_a_503_embedding_unavailable_and_does_not_wait_for_retries() -> None:
    fake = FakeBedrock(NoCredentialsError())
    container = Container(load_knowledge_settings(TEST_ENVIRONMENT), bedrock_client_factory=lambda *_: fake)  # type: ignore[arg-type]
    app = create_app(load_knowledge_settings(TEST_ENVIRONMENT), container=container)

    started = time.monotonic()
    response = AppClient(app).retrieve(BODY)

    assert response.status_code == 503 and response.json()["code"] == "embedding_unavailable"
    assert time.monotonic() - started < 1.0 and len(fake.calls) == 1  # retrying a missing credential is pointless


def test_the_embedding_client_is_not_configured_without_credentials() -> None:
    assert Container(load_knowledge_settings(TEST_ENVIRONMENT)).embedding_configured() is False


def test_the_embedding_client_is_configured_with_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    assert Container(load_knowledge_settings(TEST_ENVIRONMENT)).embedding_configured() is True

