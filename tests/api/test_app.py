"""KN-FAPI-01, -03: one FastAPI app, its problem+json answers, and configuration that stops the start.

__author__ = "Van Vo"
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping

import pytest
from fintracker_internal_http import InternalConfigError

from src.app_factory import create_app
from src.core.config import ConfigError, load_knowledge_settings
from src.core.container import Container
from src.core.dependencies import get_retrieval_service
from tests.retrieval.app_support import BODY, TEST_ENVIRONMENT, AppClient, app_settings, build
from tests.retrieval.fakes import FakeEmbedder, FakeSearch, RecordingMetrics

AWS_MARKERS = ("AWS_EXECUTION_ENV", "AWS_LAMBDA_FUNCTION_NAME", "ECS_CONTAINER_METADATA_URI", "ECS_CONTAINER_METADATA_URI_V4")


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("INTERNAL_AUTH_MODE", "AWS_ENDPOINT_URL", *AWS_MARKERS, *TEST_ENVIRONMENT):
        monkeypatch.delenv(name, raising=False)


def without(name: str) -> dict[str, str]:
    return {key: value for key, value in TEST_ENVIRONMENT.items() if key != name}


# --- KN-FAPI-01 ---


def test_an_unknown_path_is_a_404_not_found_problem() -> None:
    client, _, _ = build()
    response = client.request("GET", "/internal/nothing")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "not_found"


def test_a_known_path_with_another_method_is_a_405_method_not_allowed_problem() -> None:
    client, _, _ = build()
    response = client.request("GET", "/internal/retrieve")
    assert response.status_code == 405
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "method_not_allowed"


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_the_api_docs_are_not_served_in_the_default_strict_mode(path: str) -> None:
    client, _, _ = build()
    assert client.request("GET", path).status_code == 404


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_the_api_docs_are_served_only_in_local_mode(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERNAL_AUTH_MODE", "local")
    client, _, _ = build()
    assert client.request("GET", path, caller=None).status_code == 200


def test_the_not_found_answer_leaks_neither_the_path_nor_the_method() -> None:
    client, _, _ = build()
    response = client.request("DELETE", "/internal/secret-path-4471")
    assert "secret-path-4471" not in response.text and "DELETE" not in response.text


# --- KN-FAPI-03 ---


def test_a_test_can_replace_the_service_and_no_client_or_connection_is_created() -> None:
    created: list[str] = []
    container = Container(
        load_knowledge_settings(TEST_ENVIRONMENT),
        bedrock_client_factory=lambda *_: created.append("bedrock"),  # type: ignore[arg-type,func-returns-value]
        engine_factory=lambda *_: created.append("engine"),  # type: ignore[arg-type,func-returns-value]
    )
    app = create_app(load_knowledge_settings(TEST_ENVIRONMENT), container=container)
    service = _service()
    app.dependency_overrides[get_retrieval_service] = lambda: service

    response = AppClient(app).retrieve()

    assert response.status_code == 200 and created == []


def test_two_requests_in_one_process_get_the_same_service_instance() -> None:
    container = Container(load_knowledge_settings(TEST_ENVIRONMENT))  # builds clients, opens no connection
    assert container.retrieval_service() is container.retrieval_service()


def test_startup_without_a_required_variable_stops_with_its_name_and_never_a_value() -> None:
    for name in ("AWS_REGION", "KNOWLEDGE_DB_MODE", "KNOWLEDGE_DB_HOST", "KNOWLEDGE_DB_NAME",
                 "KNOWLEDGE_DB_RETRIEVE_USER", "KNOWLEDGE_DB_RETRIEVE_PASSWORD"):
        with pytest.raises(ConfigError) as caught:
            load_knowledge_settings(without(name))
        assert name in str(caught.value)
        assert TEST_ENVIRONMENT["KNOWLEDGE_DB_RETRIEVE_PASSWORD"] not in str(caught.value)


def test_data_api_mode_requires_its_own_variables_not_the_postgres_ones() -> None:
    environment: Mapping[str, str] = {"AWS_REGION": "us-east-1", "KNOWLEDGE_DB_MODE": "data_api"}
    with pytest.raises(ConfigError, match="KNOWLEDGE_DB_CLUSTER_ARN"):
        load_knowledge_settings(environment)
    with pytest.raises(ConfigError, match="KNOWLEDGE_DB_RETRIEVE_SECRET_ARN"):
        load_knowledge_settings({**environment, "KNOWLEDGE_DB_CLUSTER_ARN": "arn:aws:rds:x"})
    settings = load_knowledge_settings(
        {**environment, "KNOWLEDGE_DB_CLUSTER_ARN": "arn:aws:rds:x", "KNOWLEDGE_DB_RETRIEVE_SECRET_ARN": "arn:aws:s:y"}
    )
    assert settings.database.mode == "data_api"


def test_an_unknown_database_mode_stops_the_start_naming_the_variable() -> None:
    with pytest.raises(ConfigError, match="KNOWLEDGE_DB_MODE"):
        load_knowledge_settings({**TEST_ENVIRONMENT, "KNOWLEDGE_DB_MODE": "sqlite"})


def test_iam_mode_with_no_allow_list_still_starts_and_every_call_is_a_403(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed: the unset list is not a startup error (LB-IHTTP-01)."""
    monkeypatch.delenv("INTERNAL_ALLOWED_CALLER_ARNS")
    client, search, _ = build()
    response = client.retrieve()
    assert response.status_code == 403 and response.json()["code"] == "caller_not_allowed" and search.calls == []


def test_local_mode_with_an_aws_runtime_marker_stops_the_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERNAL_AUTH_MODE", "local")
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "knowledge")
    with pytest.raises(InternalConfigError, match="AWS_LAMBDA_FUNCTION_NAME"):
        app_settings()


def test_a_global_endpoint_override_stops_the_start_naming_the_variable() -> None:
    """KN-FAPI-05: AWS_ENDPOINT_URL would send Bedrock calls to a local stand-in."""
    with pytest.raises(ConfigError, match="AWS_ENDPOINT_URL") as caught:
        load_knowledge_settings({**TEST_ENVIRONMENT, "AWS_ENDPOINT_URL": "http://dynamodb-local:8000"})
    assert "dynamodb-local" not in str(caught.value)


def test_a_service_specific_endpoint_override_is_allowed() -> None:
    load_knowledge_settings({**TEST_ENVIRONMENT, "AWS_ENDPOINT_URL_BEDROCK_RUNTIME": "http://localhost:9999"})


def test_importing_the_entry_point_with_a_missing_variable_raises_naming_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The process (uvicorn or Lambda init) fails at start, not on the first request."""
    for name, value in without("KNOWLEDGE_DB_HOST").items():
        monkeypatch.setenv(name, value)
    sys.modules.pop("src.main", None)
    with pytest.raises(ConfigError, match="KNOWLEDGE_DB_HOST"):
        importlib.import_module("src.main")
    sys.modules.pop("src.main", None)


def test_the_entry_point_exposes_the_app_and_the_lambda_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in TEST_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("src.core.container.Container.verify_database", lambda self: None)
    sys.modules.pop("src.main", None)
    main = importlib.import_module("src.main")
    assert main.app.title and callable(main.handler)
    sys.modules.pop("src.main", None)


def _service():  # type: ignore[no-untyped-def]
    from src.retrieval.service import RetrievalService

    return RetrievalService(FakeEmbedder(), FakeSearch(), RecordingMetrics(), sleep=lambda _s: None)
