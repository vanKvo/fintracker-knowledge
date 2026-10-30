"""XS-ACCT-06: every JSON log line passes through a mask that replaces card-number-like runs with
``****`` plus their last four digits. It is the last line of defense behind the rule that logs carry
no financial values (app doc 4.7).

A run is 13 to 19 digits, optionally split by single spaces or hyphens, that passes the Luhn check and
starts and ends on a word boundary (XS-ACCT Appendix B). The same rule is in the Ledger
(CardNumberMasker), in each Python service's own copy of this module, and in the Ledger's log encoder.

__author__ = "Van Vo"
"""

from __future__ import annotations

import logging
from typing import Any

from aws_lambda_powertools.logging.formatter import LambdaPowertoolsFormatter

_MIN_DIGITS = 13
_MAX_DIGITS = 19
_DIGITS = frozenset("0123456789")


def mask_card_numbers(text: str) -> str:
    """Return ``text`` with every card-number-like run replaced by ``****`` and its last four digits."""
    if len(text) < _MIN_DIGITS:
        return text
    out: list[str] = []
    copied = 0
    i = 0
    while i < len(text):
        if text[i] in _DIGITS and (i == 0 or not _is_word_char(text[i - 1])):
            end = _card_end(text, i)
            if end > 0:
                out.append(text[copied:i])
                out.append("****" + _digits(text, i, end)[-4:])
                copied = i = end
                continue
        i += 1
    if not out:
        return text
    out.append(text[copied:])
    return "".join(out)


def mask_value(value: Any) -> Any:
    """Mask card numbers in a log field: text and any nested dict, list or tuple."""
    if isinstance(value, str):
        return mask_card_numbers(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return {key: mask_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [mask_value(item) for item in value]
    if value is None or isinstance(value, float):
        return value
    # Any other object (such as the exception in a stack trace) is written with str(), so mask that.
    text = str(value)
    masked = mask_card_numbers(text)
    return value if masked == text else masked


def install_record_masking() -> None:
    """Mask the message and arguments of every standard-library log record as it is created, so the
    modules that log through ``logging.getLogger`` are covered too, whichever handler formats them."""
    previous = logging.getLogRecordFactory()
    if getattr(previous, "masks_card_numbers", False):
        return

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        if isinstance(record.msg, str):
            record.msg = mask_card_numbers(record.msg)
        if isinstance(record.args, dict):
            record.args = {key: mask_value(item) for key, item in record.args.items()}
        elif record.args:
            record.args = tuple(mask_value(item) for item in record.args)
        return record

    factory.masks_card_numbers = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)


class MaskingFormatter(LambdaPowertoolsFormatter):
    """Powertools JSON formatter that masks card numbers in every field before the line is written."""

    def serialize(self, log: dict[str, Any]) -> str:
        return super().serialize(mask_value(log))


def _card_end(text: str, start: int) -> int:
    """End (exclusive) of the longest card-number-like run starting at ``start``, or -1.

    A candidate ends after a digit that is followed by a non-word character or the end of the text.
    """
    ends: dict[int, int] = {}
    count = 0
    i = start
    while i < len(text) and count < _MAX_DIGITS:
        ch = text[i]
        if ch in _DIGITS:
            count += 1
            if i + 1 == len(text) or not _is_word_char(text[i + 1]):
                ends[count] = i + 1
            i += 1
        elif ch in " -" and i + 1 < len(text) and text[i + 1] in _DIGITS:
            i += 1
        else:
            break
    for length in range(count, _MIN_DIGITS - 1, -1):
        end = ends.get(length)
        if end is not None and _luhn_valid(_digits(text, start, end)):
            return end
    return -1


def _digits(text: str, start: int, end: int) -> str:
    return "".join(ch for ch in text[start:end] if ch in _DIGITS)


def _luhn_valid(digits: str) -> bool:
    total = 0
    for position, ch in enumerate(reversed(digits)):
        d = int(ch)
        if position % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"
