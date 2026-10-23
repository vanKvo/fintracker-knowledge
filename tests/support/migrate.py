"""Applies the Flyway-style migrations in db/migration to a test database.

Production runs Flyway as a deploy step (never at startup); this mirrors its ordering and
one-transaction-per-file behaviour so tests exercise the same SQL.

__author__ = "Van Vo"
"""

from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy.engine import Engine

MIGRATION_DIR = Path(__file__).resolve().parents[2] / "db" / "migration"
_VERSION = re.compile(r"^V(\d+)__.+\.sql$")


class MigrationError(Exception):
    """A migration failed; later migrations were not attempted."""


def migration_files(directory: Path = MIGRATION_DIR) -> list[Path]:
    versioned = [(int(m.group(1)), p) for p in directory.glob("V*.sql") if (m := _VERSION.match(p.name))]
    return [p for _, p in sorted(versioned)]


def apply_migrations(engine: Engine, directory: Path = MIGRATION_DIR) -> None:
    for path in migration_files(directory):
        try:
            _apply_file(engine, path)
        except Exception as exc:
            raise MigrationError(f"{path.name} failed") from exc


def _apply_file(engine: Engine, path: Path) -> None:
    # Raw DBAPI cursor with no parameters, so "%" in format() calls is left alone. The whole
    # file commits or rolls back together.
    raw = engine.raw_connection()
    try:
        raw.cursor().execute(path.read_text())
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()
