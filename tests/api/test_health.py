"""KN-FAPI-08: `/health` touches nothing; `/ready` needs the database as kn_retrieve and the embedding client.

__author__ = "Van Vo"
"""

from __future__ import annotations

import pytest

from src.app_factory import create_app
from src.core.dependencies import get_container
from src.core.exceptions import DatabaseUnavailableError
from tests.retrieval.app_support import AppClient, app_settings


class FakeContainer:
    def __init__(self, database: bool = True, embedding: bool = True) -> None:
        self.database = database
        self.embedding = embedding
        self.touched: list[str] = []

    def check_database(self) -> None:
        self.touched.append("database")
        if not self.database:
            raise DatabaseUnavailableError()

    def embedding_configured(self) -> bool:
        self.touched.append("embedding")
        return self.embedding


def client(container: FakeContainer) -> AppClient:
    app = create_app(app_settings())
    app.dependency_overrides[get_container] = lambda: container
    return AppClient(app)


def probe(container: FakeContainer, path: str):  # type: ignore[no-untyped-def]
    # No caller and no user: the probes are not internal routes and carry no user data.
    return client(container).request("GET", path, caller=None, user=None)


def test_health_is_200_without_touching_any_dependency() -> None:
    container = FakeContainer(database=False, embedding=False)
    response = probe(container, "/health")
    assert response.status_code == 200 and container.touched == []


def test_ready_is_200_when_the_database_answers_and_the_embedding_client_is_configured() -> None:
    assert probe(FakeContainer(), "/ready").status_code == 200


def test_ready_is_503_database_unavailable_when_the_database_is_stopped_while_health_stays_200() -> None:
    container = FakeContainer(database=False)
    ready = probe(container, "/ready")
    assert ready.status_code == 503 and ready.json()["code"] == "database_unavailable"
    assert ready.headers["content-type"].startswith("application/problem+json")
    assert probe(container, "/health").status_code == 200


def test_ready_is_503_embedding_unavailable_without_credentials_while_health_stays_200() -> None:
    """KN-FAPI-05: no credentials leaves the app started and not ready."""
    container = FakeContainer(embedding=False)
    ready = probe(container, "/ready")
    assert ready.status_code == 503 and ready.json()["code"] == "embedding_unavailable"
    assert probe(container, "/health").status_code == 200


@pytest.mark.parametrize("path", ["/health", "/ready"])
def test_the_probes_return_no_user_data_and_need_no_headers(path: str) -> None:
    response = probe(FakeContainer(), path)
    assert response.status_code == 200 and set(response.json()) <= {"status"}
