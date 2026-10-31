"""System-docs copies in S3, uploaded by CI under `system-docs/` (KN-SDOC-01). Read-only here.

__author__ = "Van Vo"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from botocore.exceptions import ClientError

from src.system_docs.exceptions import SourceMissing

PREFIX = "system-docs/"
MANIFEST_PREFIX = f"{PREFIX}_manifests/"


class DocSource(Protocol):
    def read(self, path: str) -> bytes: ...
    def list_paths(self) -> list[str]: ...


class S3DocSource:
    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def read(self, path: str) -> bytes:
        try:
            body: bytes = self._client.get_object(Bucket=self._bucket, Key=PREFIX + path)["Body"].read()
            return body
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                raise SourceMissing() from None
            raise

    def list_paths(self) -> list[str]:
        paths: list[str] = []
        for page in self._client.get_paginator("list_objects_v2").paginate(Bucket=self._bucket, Prefix=PREFIX):
            for item in page.get("Contents", []):
                key: str = item["Key"]
                if key.endswith(".md") and not key.startswith(MANIFEST_PREFIX):
                    paths.append(key[len(PREFIX) :])
        return sorted(paths)


HELP_DIRECTORY = "help"


class LocalDocSource:
    """KN-FAPI-06: help docs from a folder, in the path form of the S3 keys and the golden set (`help/statements/x.md`).

    Only `<root>/help/**/*.md` is read. A path with `..`, outside `help/`, not Markdown, or whose real location (after
    symlinks) is outside `<root>/help/` is reported as missing, as the publisher refuses the same paths.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._help = (self._root / HELP_DIRECTORY).resolve()

    def read(self, path: str) -> bytes:
        resolved = self._resolve(path)
        if resolved is None:
            raise SourceMissing()
        try:
            return resolved.read_bytes()
        except OSError:
            raise SourceMissing() from None

    def list_paths(self) -> list[str]:
        if not self._help.is_dir():
            return []
        found = (p.relative_to(self._root).as_posix() for p in (self._root / HELP_DIRECTORY).rglob("*.md"))
        return sorted(path for path in found if self._resolve(path) is not None)

    def _resolve(self, path: str) -> Path | None:
        parts = path.split("/")
        if ".." in parts or path.startswith("/") or parts[0] != HELP_DIRECTORY or not path.endswith(".md"):
            return None
        candidate = (self._root / path).resolve()
        return candidate if candidate.is_file() and candidate.is_relative_to(self._help) else None
