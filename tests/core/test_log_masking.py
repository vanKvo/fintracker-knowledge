"""XS-ACCT-06: every JSON log line passes through the card-number mask (Appendix B).

__author__ = "Van Vo"
"""

import io
import json
from collections.abc import Iterator
import logging

import pytest

from src.core.log_masking import mask_card_numbers
from src.core.observability import logger


@pytest.fixture
def log_output() -> Iterator[io.StringIO]:
    """What the service's logger writes. The handler's stream is swapped for a buffer, formatter unchanged."""
    buffer = io.StringIO()
    handler = logger.registered_handler
    previous = handler.setStream(buffer)
    yield buffer
    handler.setStream(previous)


@pytest.mark.parametrize("card", ["4111111111111111", "4111-1111-1111-1111", "4111 1111 1111 1111", "5500 0000 0000 0004"])
def test_a_luhn_valid_run_is_replaced_with_stars_and_its_last_four(card: str) -> None:
    assert mask_card_numbers(f"PAYMENT TO CARD {card}") == f"PAYMENT TO CARD ****{card[-4:]}"


@pytest.mark.parametrize("text", ["1234567812345678", "202610291430", "CALL 4155550123", "ORDER 1234567890123", "41111111111111119999"])
def test_other_digit_runs_are_left_alone(text: str) -> None:
    assert mask_card_numbers(text) == text


def test_a_number_followed_by_an_amount_is_still_masked() -> None:
    assert mask_card_numbers("CARD 4111111111111111 25.00") == "CARD ****1111 25.00"
    assert mask_card_numbers("A 4222222222222 B 6759649826438453003") == "A ****2222 B ****3003"


def test_a_log_call_whose_message_includes_a_card_number_writes_it_masked(log_output: io.StringIO) -> None:
    logger.info("Payment to card 4111111111111111 failed")

    out = log_output.getvalue()
    assert json.loads(out.strip().splitlines()[-1])["message"] == "Payment to card ****1111 failed"
    assert "4111111111111111" not in out




def test_a_standard_library_logger_is_masked_before_any_handler_sees_it(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        logging.getLogger("src.downloads.repository").info("card %s declined", "4111111111111111")

    assert caplog.messages == ["card ****1111 declined"]
