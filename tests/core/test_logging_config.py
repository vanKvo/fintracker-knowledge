"""XS-ACCT-06: the app's log output is one JSON line per record, with the card-number mask, for every module's logger.

__author__ = "Van Vo"
"""

import json
import logging

import pytest

from src.core.logging_config import configure_logging


def test_a_standard_library_logger_writes_a_json_line_with_card_numbers_masked(capsys: pytest.CaptureFixture[str]) -> None:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    try:
        configure_logging()
        configure_logging()  # idempotent: one handler, so one line per record
        logging.getLogger("src.some_module").warning("charged card 4111 1111 1111 1111")
        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)

    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["message"] == "charged card ****1111" and record["service"] == "knowledge-service"
