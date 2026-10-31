"""Builds the FastAPI app around fakes and calls it the way API Gateway does (KN-FAPI-01, -03, -10).

The app runs for real: routing, the caller check, the user context, the error handlers. Only its edges are
fake: the embedder, the search and the metrics sink, replaced through `app.dependency_overrides`.

__author__ = "Van Vo"
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

import httpx
from fastapi import FastAPI
from fintracker_internal_http.testing import api_gateway_event
from starlette.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send

from src.app_factory import create_app
from src.core.config import KnowledgeSettings, load_knowledge_settings
from src.core.dependencies import get_metrics, get_retrieval_service
from src.retrieval.models import SearchResult
from src.retrieval.service import RetrievalService
from tests.retrieval.fakes import (
    CALLER_ARN,
    REQUEST_ID,
    USER_ID,
    FakeEmbedder,
    FakeSearch,
    RecordingMetrics,
    hit,
)

BODY = {"query": "How do I confirm a CSV mapping?", "scope": "system", "k": 5}

TEST_ENVIRONMENT = {
    "AWS_REGION": "us-east-1",
    "KNOWLEDGE_DB_MODE": "postgres",
    "KNOWLEDGE_DB_HOST": "localhost",
    "KNOWLEDGE_DB_PORT": "5432",
    "KNOWLEDGE_DB_NAME": "knowledge",
    "KNOWLEDGE_DB_RETRIEVE_USER": "kn_app_retrieve",
    "KNOWLEDGE_DB_RETRIEVE_PASSWORD": "not-a-real-password",
}


def app_settings(**extra: str) -> KnowledgeSettings:
    """Settings as the process reads them: the real environment (INTERNAL_AUTH_MODE, the caller allow-list) plus the
    test's own database and region."""
    return load_knowledge_settings({**os.environ, **TEST_ENVIRONMENT, **extra})


class _EventInjector:
    """What Mangum does in AWS: puts the API Gateway event in the ASGI scope."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.event: dict[str, Any] | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self.event is not None:
            scope["aws.event"] = self.event
        await self.app(scope, receive, send)


class AppClient:
    def __init__(self, app: FastAPI) -> None:
        self.app = app

    def request(
        self,
        method: str,
        path: str,
        *,
        content: str | bytes | None = None,
        caller: str | None = CALLER_ARN,
        user: str | None = USER_ID,
        headers: Mapping[str, str] | None = None,
        request_id: str = REQUEST_ID,
    ) -> httpx.Response:
        sent = {"content-type": "application/json", **(headers or {})}
        if user is not None:
            sent["x-internal-user-id"] = user
        event = api_gateway_event(caller_arn=caller, headers=sent, method=method, path=path)
        event["requestContext"]["requestId"] = request_id
        injector = _EventInjector(self.app)
        injector.event = event
        with TestClient(injector, raise_server_exceptions=False) as client:
            return client.request(method, path, content=content, headers=sent)

    def retrieve(self, body: Any = BODY, **kwargs: Any) -> httpx.Response:
        content = body if isinstance(body, (str, bytes)) else json.dumps(body)
        return self.request("POST", "/internal/retrieve", content=content, **kwargs)


def build(
    search: FakeSearch | None = None, embedder: FakeEmbedder | None = None
) -> tuple[AppClient, FakeSearch, RecordingMetrics]:
    search = search or FakeSearch(SearchResult([hit()], "9f2c1ab"))
    metrics = RecordingMetrics()
    service = RetrievalService(embedder or FakeEmbedder(), search, metrics, sleep=lambda _s: None)
    app = create_app(app_settings())
    app.dependency_overrides[get_retrieval_service] = lambda: service
    app.dependency_overrides[get_metrics] = lambda: metrics
    return AppClient(app), search, metrics
