"""Caller allow-list for Knowledge-internal operations (rebuild, index status). Fails closed.

__author__ = "Van Vo"
"""

from __future__ import annotations

from collections.abc import Collection


class CallerNotAllowed(Exception):
    def __init__(self) -> None:
        super().__init__("The caller is not allowed to do this.")


def authorize_caller(caller_arn: str | None, allowed_arns: Collection[str]) -> None:
    if not caller_arn or caller_arn not in allowed_arns:
        raise CallerNotAllowed()
