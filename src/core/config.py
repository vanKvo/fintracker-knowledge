"""KN-FAPI-03, -04, -05: the service's settings, read once at startup (Appendix B).

A missing or unsafe setting stops the process with the variable's name and never its value. The internal-call
settings (INTERNAL_AUTH_MODE, the caller allow-list) are the shared library's (LB-IHTTP-01/02).

`role` picks which database login the settings describe: the API runs as `retrieve`, the index command as `ingest`.

__author__ = "Van Vo"
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from fintracker_internal_http import InternalSettings
from fintracker_internal_http import load_settings as load_internal_settings

DbMode = Literal["postgres", "data_api"]
Role = Literal["retrieve", "ingest"]
EmbedderMode = Literal["bedrock", "replay"]

GLOBAL_ENDPOINT_VARIABLE = "AWS_ENDPOINT_URL"
# Markers of an AWS runtime. Replay mode answers from a saved file, so it must never run where real traffic is served.
AWS_RUNTIME_MARKERS = ("AWS_LAMBDA_FUNCTION_NAME", "AWS_EXECUTION_ENV", "ECS_CONTAINER_METADATA_URI", "ECS_CONTAINER_METADATA_URI_V4")
DEFAULT_PORT = 5432


class ConfigError(RuntimeError):
    """Invalid configuration. The message names the variable, never a value."""


@dataclass(frozen=True)
class DatabaseSettings:
    mode: DbMode
    role: Role
    host: str = ""
    port: int = DEFAULT_PORT
    name: str = ""
    user: str = ""
    password: str = field(default="", repr=False)
    cluster_arn: str = ""
    secret_arn: str = ""


@dataclass(frozen=True)
class KnowledgeSettings:
    internal: InternalSettings
    database: DatabaseSettings
    region: str
    # KN-CHUNK-05: `bedrock` calls the real embedding model (production). `replay` answers from the recording in
    # `replay_dir` and needs no AWS credentials; it serves the golden and negative questions only.
    embedder: EmbedderMode = "bedrock"
    replay_dir: str = ""


def load_knowledge_settings(environ: Mapping[str, str] | None = None, *, role: Role = "retrieve") -> KnowledgeSettings:
    """Reads and validates the settings.

    Raises:
        ConfigError: a variable is missing or invalid, or AWS_ENDPOINT_URL is set.
        InternalConfigError: invalid internal-call settings, or `local` auth mode inside an AWS runtime.
    """
    env = os.environ if environ is None else environ
    internal = load_internal_settings(env)
    if env.get(GLOBAL_ENDPOINT_VARIABLE):
        raise ConfigError(f"{GLOBAL_ENDPOINT_VARIABLE} must not be set: it would redirect Bedrock calls too")
    region = _required(env, "AWS_REGION", fallback="AWS_DEFAULT_REGION")
    embedder, replay_dir = _embedder(env)
    return KnowledgeSettings(
        internal=internal, database=_database(env, role), region=region, embedder=embedder, replay_dir=replay_dir
    )


def _embedder(env: Mapping[str, str]) -> tuple[EmbedderMode, str]:
    mode = env.get("KNOWLEDGE_EMBEDDER", "").strip() or "bedrock"
    if mode == "bedrock":
        return "bedrock", ""
    if mode != "replay":
        raise ConfigError("KNOWLEDGE_EMBEDDER must be one of: bedrock, replay")
    marker = next((name for name in AWS_RUNTIME_MARKERS if env.get(name)), None)
    if marker:
        raise ConfigError(f"KNOWLEDGE_EMBEDDER=replay is refused in an AWS runtime ({marker} is set)")
    return "replay", _required(env, "KNOWLEDGE_REPLAY_DIR")


def _database(env: Mapping[str, str], role: Role) -> DatabaseSettings:
    mode = _required(env, "KNOWLEDGE_DB_MODE")
    if mode == "postgres":
        prefix = f"KNOWLEDGE_DB_{role.upper()}"
        return DatabaseSettings(
            mode="postgres",
            role=role,
            host=_required(env, "KNOWLEDGE_DB_HOST"),
            port=_port(env),
            name=_required(env, "KNOWLEDGE_DB_NAME"),
            user=_required(env, f"{prefix}_USER"),
            password=_required(env, f"{prefix}_PASSWORD"),
        )
    if mode == "data_api":
        if role != "retrieve":
            raise ConfigError("KNOWLEDGE_DB_MODE must be postgres for the index command")
        return DatabaseSettings(
            mode="data_api",
            role=role,
            name=env.get("KNOWLEDGE_DB_NAME", "").strip(),
            cluster_arn=_required(env, "KNOWLEDGE_DB_CLUSTER_ARN"),
            secret_arn=_required(env, "KNOWLEDGE_DB_RETRIEVE_SECRET_ARN"),
        )
    raise ConfigError("KNOWLEDGE_DB_MODE must be one of: postgres, data_api")


def _required(env: Mapping[str, str], name: str, *, fallback: str | None = None) -> str:
    value = env.get(name, "").strip() or (env.get(fallback, "").strip() if fallback else "")
    if not value:
        raise ConfigError(f"{name} is not set")
    return value


def _port(env: Mapping[str, str]) -> int:
    raw = env.get("KNOWLEDGE_DB_PORT", "").strip()
    if not raw:
        return DEFAULT_PORT
    if not raw.isdigit() or not 0 < int(raw) < 65536:
        raise ConfigError("KNOWLEDGE_DB_PORT must be a port number")
    return int(raw)
