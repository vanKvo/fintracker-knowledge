"""KN-SDOC-06: index freshness for the status route and the alarms.

__author__ = "Van Vo"
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime, timedelta

from src.system_docs.access import authorize_caller
from src.system_docs.models import IndexStatus
from src.system_docs.repository import DocumentIndex
from src.system_docs.schemas import RunStatus

STALE_AFTER = timedelta(minutes=30)


def get_index_status(index: DocumentIndex, caller_arn: str | None, allowed_arns: Collection[str]) -> IndexStatus:
    """Body of GET /internal/index-status: the last fully successful run's commit and finish time."""
    authorize_caller(caller_arn, allowed_arns)
    return index.last_succeeded_run()


def run_alarm(status: RunStatus) -> bool:
    return status != "succeeded"


def stale_alarm(manifest_at: datetime, last_success_at: datetime | None, now: datetime) -> bool:
    """True when a manifest is over 30 minutes old and no successful run has finished since it."""
    followed = last_success_at is not None and last_success_at >= manifest_at
    return not followed and now - manifest_at > STALE_AFTER
