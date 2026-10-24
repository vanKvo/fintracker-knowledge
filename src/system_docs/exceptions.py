"""System-docs indexing errors (KN-SDOC). Generic messages only, never doc text or the underlying
error. They become branch outcomes (Appendix B), never HTTP responses; a database failure is the
shared `DatabaseUnavailableError`.

__author__ = "Van Vo"
"""

from __future__ import annotations


class InvalidManifest(ValueError):
    def __init__(self) -> None:
        super().__init__("The manifest is not valid.")


class SourceMissing(Exception):
    def __init__(self) -> None:
        super().__init__("The source file is not available.")


class InvalidSource(Exception):
    def __init__(self) -> None:
        super().__init__("The source file is not valid.")
