"""KN-SDOC-02: reading the system-docs copies from S3.

__author__ = "Van Vo"
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from botocore.exceptions import ClientError

from src.system_docs.models import SourceMissing
from src.system_docs.source import S3DocSource


class FakeS3:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803 - boto3 signature
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "x"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def get_paginator(self, name: str) -> Any:
        keys = sorted(self.objects)

        class Paginator:
            def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
                return [{"Contents": [{"Key": k} for k in keys[:2]]}, {"Contents": [{"Key": k} for k in keys[2:]]}]

        return Paginator()


def test_source_reads_a_doc_by_its_repo_path() -> None:
    source = S3DocSource(FakeS3({"system-docs/help/a.md": b"# A"}), "bucket")

    assert source.read("help/a.md") == b"# A"


def test_missing_object_is_reported_as_source_missing() -> None:
    with pytest.raises(SourceMissing):
        S3DocSource(FakeS3({}), "bucket").read("help/a.md")


def test_listing_returns_markdown_docs_without_manifests() -> None:
    objects = {
        "system-docs/help/a.md": b"",
        "system-docs/help/b.md": b"",
        "system-docs/help/c.png": b"",
        "system-docs/_manifests/9f2c1ab.json": b"",
    }

    assert S3DocSource(FakeS3(objects), "bucket").list_paths() == ["help/a.md", "help/b.md"]
