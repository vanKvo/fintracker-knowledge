"""KN-FAPI-03, -05, -04: builds each part of the service once per process, on first use.

Nothing here connects to anything at import or construction: the Bedrock client, the database engine and the
services are created when a route first asks for them, so a test (or a failing dependency) never prevents the
app from starting. Tests replace parts with `app.dependency_overrides` or the factories below.

__author__ = "Van Vo"
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

import boto3
from botocore.config import Config

from src.config.knowledge_config import EmbeddingConfig
from src.core.config import DatabaseSettings, KnowledgeSettings
from src.core.exceptions import DatabaseUnavailableError
from src.embedding.service import BedrockRuntime, Embedder
from src.retrieval.metrics import LogMetrics, RetrievalMetrics
from src.retrieval.repository import PostgresChunkSearch
from src.retrieval.service import RetrievalService
from src.vector_store.connections import (
    Connections,
    DataApiConnections,
    PostgresConnections,
    postgres_engine,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# KN-RET-07's budget is 3 s, so the embedding call may not outlive it; KN-CHUNK-02 owns the retries, not the SDK.
BEDROCK_CONNECT_TIMEOUT_SECONDS = 1
BEDROCK_READ_TIMEOUT_SECONDS = 2

BedrockFactory = Callable[[KnowledgeSettings], BedrockRuntime]
EngineFactory = Callable[[DatabaseSettings], Any]


def build_bedrock_client(settings: KnowledgeSettings) -> BedrockRuntime:
    """Region from configuration and credentials from the default chain. Only service-specific endpoint overrides
    (AWS_ENDPOINT_URL_BEDROCK_RUNTIME) apply: a global AWS_ENDPOINT_URL is refused at startup."""
    client: BedrockRuntime = boto3.client(
        "bedrock-runtime",
        region_name=settings.region,
        config=Config(
            connect_timeout=BEDROCK_CONNECT_TIMEOUT_SECONDS,
            read_timeout=BEDROCK_READ_TIMEOUT_SECONDS,
            retries={"total_max_attempts": 1, "mode": "standard"},
        ),
    )
    return client


def build_replay_client(settings: KnowledgeSettings) -> BedrockRuntime:
    """KN-CHUNK-05: answers from the recording in `replay_dir`. No AWS code runs and no credentials are read."""
    from pathlib import Path

    from src.rag_eval.recording import RecordedBedrock

    return RecordedBedrock.from_directory(Path(settings.replay_dir))


def build_embedding_client(settings: KnowledgeSettings) -> BedrockRuntime:
    """The client the container uses unless a test passes its own: replay when configured, else the real one."""
    return build_replay_client(settings) if settings.embedder == "replay" else build_bedrock_client(settings)


def build_engine(database: DatabaseSettings) -> Any:
    return postgres_engine(database.host, database.port, database.name, database.user, database.password)


class Container:
    def __init__(
        self,
        settings: KnowledgeSettings,
        *,
        bedrock_client_factory: BedrockFactory = build_embedding_client,
        engine_factory: EngineFactory = build_engine,
        embedding_config: EmbeddingConfig = EmbeddingConfig(),
        metrics: RetrievalMetrics | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._bedrock_client_factory = bedrock_client_factory
        self._engine_factory = engine_factory
        self._embedding_config = embedding_config
        self._metrics = metrics
        self._sleep = sleep
        self._lock = threading.RLock()
        self._cache: dict[str, Any] = {}

    @property
    def settings(self) -> KnowledgeSettings:
        return self._settings

    def bedrock_client(self) -> BedrockRuntime:
        return self._once("bedrock", lambda: self._bedrock_client_factory(self._settings))

    def embedder(self) -> Embedder:
        return self._once("embedder", lambda: Embedder(self.bedrock_client(), self._embedding_config))

    def connections(self) -> Connections:
        return self._once("connections", self._build_connections)

    def metrics(self) -> RetrievalMetrics:
        return self._once("metrics", lambda: self._metrics or LogMetrics())

    def retrieval_service(self) -> RetrievalService:
        return self._once(
            "retrieval",
            lambda: RetrievalService(
                self.embedder(),
                PostgresChunkSearch(self.connections().connection),
                self.metrics(),
                self._embedding_config,
                sleep=self._sleep,
            ),
        )

    # --- readiness and startup checks ---

    def check_database(self) -> None:
        """Raises DatabaseUnavailableError unless the database answers a trivial query as the retrieve role."""
        self.connections().ping()

    def embedding_configured(self) -> bool:
        """True when the embedding client can work: the recording loads (replay), or the default credential chain yields
        credentials (bedrock). Calls nothing billable."""
        if self._settings.embedder == "replay":
            try:
                self.bedrock_client()
            except Exception:  # noqa: BLE001 - a missing or unreadable recording means not ready, not a crash
                return False
            return True
        return boto3.session.Session(region_name=self._settings.region).get_credentials() is not None

    def verify_database(self) -> None:
        """KN-FAPI-04: in `postgres` mode, refuse a role that bypasses Row-Level Security.

        An unreachable database is not fatal here (the app starts and `/ready` says no); an unsafe role is.

        Raises:
            UnsafeDatabaseRoleError: the connected role is a superuser or has BYPASSRLS.
        """
        connections = self.connections()
        if not isinstance(connections, PostgresConnections):
            return
        try:
            connections.assert_role_is_safe()
        except DatabaseUnavailableError:
            logger.warning("Database not reachable at startup; the role check runs again on the first call")
            return
        logger.info("Database role verified: no superuser, no BYPASSRLS")

    def _build_connections(self) -> Connections:
        database = self._settings.database
        if database.mode == "data_api":
            client = boto3.client("rds-data", region_name=self._settings.region)
            return DataApiConnections(
                client, cluster_arn=database.cluster_arn, secret_arn=database.secret_arn, database=database.name or None
            )
        return PostgresConnections(self._engine_factory(database))

    def _once(self, key: str, build: Callable[[], T]) -> T:
        with self._lock:
            if key not in self._cache:
                self._cache[key] = build()
            cached: T = self._cache[key]
            return cached
