"""System-docs copies in S3, uploaded by CI under `system-docs/` (KN-SDOC-01). Read-only here.

__author__ = "Van Vo"
"""

from __future__ import annotations

from typing import Any, Protocol

from botocore.exceptions import ClientError

from src.system_docs.models import SourceMissing

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
