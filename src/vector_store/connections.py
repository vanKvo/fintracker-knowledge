"""KN-FAPI-04: one connection factory per database role, behind the one interface the repositories use.

`connection()` opens one transaction per call, already running as the role's login, and yields something that
runs `execute(text(...), params)`. Two implementations:

- `PostgresConnections`: a direct TCP connection with a login role that is a member of the NOLOGIN role. Used
  locally and in CI. At startup the role must not be a superuser or have BYPASSRLS, or Row-Level Security would
  silently not apply.
- `DataApiConnections`: the RDS Data API with the role's secret, used in AWS. Each call is its own session, so the
  user context is set inside the same transaction (KN-VS-03).

Every database failure is a `SQLAlchemyError` (the Data API's is `DataApiError`) so the repositories map both to
`DatabaseUnavailableError`. Driver and AWS messages (hosts, SQL, ARNs) are never passed on.

__author__ = "Van Vo"
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol

from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from src.core.exceptions import DatabaseUnavailableError

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_SECONDS = 2
POOL_SIZE = 5


class UnsafeDatabaseRoleError(RuntimeError):
    """The connected role would bypass Row-Level Security. Names the role, never the password."""


class DataApiError(SQLAlchemyError):
    """An RDS Data API call failed. Carries the exception type only."""


class Connections(Protocol):
    def connection(self) -> AbstractContextManager[Any]: ...
    def ping(self) -> None: ...


def postgres_engine(host: str, port: int, name: str, user: str, password: str) -> Engine:
    # The pool's DEBUG line on a failed connect quotes the driver's message (host, user), so keep it quiet.
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    url = URL.create("postgresql+psycopg", username=user, password=password, host=host, port=port, database=name)
    return create_engine(
        url,
        pool_size=POOL_SIZE,
        max_overflow=POOL_SIZE,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={"connect_timeout": CONNECT_TIMEOUT_SECONDS},
    )


class PostgresConnections:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        with self._engine.begin() as conn:
            yield conn

    def ping(self) -> None:
        """A trivial query as this role.

        Raises:
            DatabaseUnavailableError: the database did not answer.
        """
        try:
            with self.connection() as conn:
                conn.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            raise DatabaseUnavailableError(log_message=f"database ping failed: {type(exc).__name__}") from exc

    def assert_role_is_safe(self) -> None:
        """Refuses a superuser or BYPASSRLS role.

        Raises:
            UnsafeDatabaseRoleError: the role bypasses Row-Level Security.
            DatabaseUnavailableError: the database did not answer (the caller decides whether that is fatal).
        """
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    text("SELECT current_user, rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
                ).one()
        except SQLAlchemyError as exc:
            raise DatabaseUnavailableError(log_message=f"role check failed: {type(exc).__name__}") from exc
        role, superuser, bypass = str(row[0]), bool(row[1]), bool(row[2])
        if superuser:
            raise UnsafeDatabaseRoleError(f"Database role {role} is a superuser, which bypasses Row-Level Security.")
        if bypass:
            raise UnsafeDatabaseRoleError(f"Database role {role} has BYPASSRLS, which bypasses Row-Level Security.")


# --- RDS Data API ---------------------------------------------------------------------------------------------


class RdsData(Protocol):
    """The four rds-data calls a transaction uses."""

    def begin_transaction(self, **kwargs: Any) -> dict[str, Any]: ...
    def execute_statement(self, **kwargs: Any) -> dict[str, Any]: ...
    def commit_transaction(self, **kwargs: Any) -> dict[str, Any]: ...
    def rollback_transaction(self, **kwargs: Any) -> dict[str, Any]: ...


class DataApiResult:
    def __init__(self, rows: list[tuple[Any, ...]], rowcount: int) -> None:
        self._rows = rows
        self.rowcount = rowcount

    def all(self) -> list[tuple[Any, ...]]:
        return list(self._rows)

    def first(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None

    def scalar(self) -> Any:
        return self._rows[0][0] if self._rows else None


class DataApiConnection:
    """One open Data API transaction; `execute` takes the same text statements and named parameters as SQLAlchemy."""

    def __init__(self, client: RdsData, target: dict[str, str], transaction_id: str) -> None:
        self._client = client
        self._target = target
        self._transaction_id = transaction_id

    def execute(self, statement: Any, params: dict[str, Any] | Sequence[Any] | None = None) -> DataApiResult:
        if params is not None and not isinstance(params, dict):
            raise TypeError("the Data API connection runs one parameter set per statement")
        sql: str = statement if isinstance(statement, str) else statement.text
        try:
            response = self._client.execute_statement(
                **self._target,
                transactionId=self._transaction_id,
                sql=sql,
                parameters=[{"name": name, **_field(value)} for name, value in (params or {}).items()],
            )
        except (BotoCoreError, ClientError) as exc:
            raise DataApiError(f"data api call failed: {type(exc).__name__}") from None
        rows = [tuple(_value(item) for item in record) for record in response.get("records", [])]
        return DataApiResult(rows, int(response.get("numberOfRecordsUpdated", 0)))


class DataApiConnections:
    def __init__(self, client: RdsData, *, cluster_arn: str, secret_arn: str, database: str | None = None) -> None:
        self._client = client
        self._target = {"resourceArn": cluster_arn, "secretArn": secret_arn}
        if database:
            self._target["database"] = database

    @contextmanager
    def connection(self) -> Iterator[DataApiConnection]:
        try:
            transaction_id = self._client.begin_transaction(**self._target)["transactionId"]
        except (BotoCoreError, ClientError) as exc:
            raise DataApiError(f"data api call failed: {type(exc).__name__}") from None
        try:
            yield DataApiConnection(self._client, self._target, transaction_id)
        except BaseException:
            self._finish(self._client.rollback_transaction, transaction_id)
            raise
        try:
            self._client.commit_transaction(**self._target_for_end(), transactionId=transaction_id)
        except (BotoCoreError, ClientError) as exc:
            raise DataApiError(f"data api call failed: {type(exc).__name__}") from None

    def ping(self) -> None:
        try:
            with self.connection() as conn:
                conn.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            raise DatabaseUnavailableError(log_message=f"database ping failed: {type(exc).__name__}") from exc

    def _finish(self, call: Any, transaction_id: str) -> None:
        try:
            call(**self._target_for_end(), transactionId=transaction_id)
        except (BotoCoreError, ClientError) as exc:
            logger.error("Data API rollback failed: %s", type(exc).__name__)  # the transaction times out by itself

    def _target_for_end(self) -> dict[str, str]:
        # commit and rollback take the cluster and secret, not the database name
        return {key: value for key, value in self._target.items() if key != "database"}


def _field(value: Any) -> dict[str, Any]:
    if value is None:
        return {"value": {"isNull": True}}
    if isinstance(value, bool):
        return {"value": {"booleanValue": value}}
    if isinstance(value, int):
        return {"value": {"longValue": value}}
    if isinstance(value, float):
        return {"value": {"doubleValue": value}}
    if isinstance(value, uuid.UUID):
        return {"value": {"stringValue": str(value)}, "typeHint": "UUID"}
    return {"value": {"stringValue": str(value)}}


def _value(item: dict[str, Any]) -> Any:
    if item.get("isNull"):
        return None
    for key in ("stringValue", "longValue", "doubleValue", "booleanValue"):
        if key in item:
            return item[key]
    raise DataApiError("data api returned a value type that is not supported")
