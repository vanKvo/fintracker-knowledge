"""Caller allow-list for Knowledge-internal operations (rebuild, index status). Fails closed.

__author__ = "Van Vo"
"""

from __future__ import annotations

from collections.abc import Collection

from src.core.exceptions import CallerNotAllowedError


def authorize_caller(caller_arn: str | None, allowed_arns: Collection[str]) -> None:
    if not caller_arn or caller_arn not in allowed_arns:
        raise CallerNotAllowedError(log_message="caller not on the allow-list")
