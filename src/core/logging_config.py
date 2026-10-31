"""One JSON line per log record on stdout, with the card-number mask (XS-ACCT-06; app doc 4.7).

Standard-library loggers propagate to the root logger, so every module that logs through `logging.getLogger` is covered.
Call once at process start.

__author__ = "Van Vo"
"""

from __future__ import annotations

import logging
import os
import sys

from src.core.log_masking import MaskingFormatter

_HANDLER_NAME = "knowledge-json"


def configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    if any(handler.get_name() == _HANDLER_NAME for handler in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(MaskingFormatter(service="knowledge-service"))
    root.addHandler(handler)
