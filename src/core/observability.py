"""Lambda Powertools logger for the Knowledge.

XS-ACCT-06: the logger's JSON lines are masked, and importing this module also masks every
standard-library log record, so the modules that log through `logging.getLogger` are covered too.

__author__ = "Van Vo"
"""

from aws_lambda_powertools import Logger

from .log_masking import MaskingFormatter, install_record_masking

logger = Logger(service="knowledge-service", logger_formatter=MaskingFormatter())
install_record_masking()

__all__ = ["logger"]
