"""KN-SDOC-01: CI upload of changed and deleted docs.

__author__ = "Van Vo"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.system_docs.publisher import MAX_DOC_BYTES, UploadFailed, parse_name_status, publish_system_docs

BUCKET = "knowledge-bucket"
COMMIT = "9f2c1ab"
MERGED_AT = "2026-10-20T18:04:00Z"


class FakeS3:
    def __init__(self, fail_on: str | None = None) -> None:
        self.puts: list[dict[str, Any]] = []
        self.fail_on = fail_on

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        if self.fail_on and kwargs["Key"].endswith(self.fail_on):
            raise RuntimeError("S3 says no: secret-internal-detail")
        self.puts.append(kwargs)
        return {}


def write(root: Path, relative: str, content: bytes = b"# Title\n\nBody\n") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def publish(root: Path, s3: FakeS3, changed: list[str], deleted: list[str]):  # type: ignore[no-untyped-def]
    return publish_system_docs(
        root=root, changed=changed, deleted=deleted, commit=COMMIT, merged_at=MERGED_AT, s3=s3, bucket=BUCKET
    )


def test_two_changed_and_one_deleted_upload_two_files_and_one_manifest(tmp_path: Path) -> None:
    write(tmp_path, "help/a.md")
    write(tmp_path, "help/sub/b.md")
    s3 = FakeS3()

    result = publish(tmp_path, s3, ["help/a.md", "help/sub/b.md"], ["help/old.md"])

    keys = [put["Key"] for put in s3.puts]
    assert keys == [
        "system-docs/help/a.md",
        "system-docs/help/sub/b.md",
        "system-docs/_manifests/9f2c1ab.json",
    ]
    manifest = json.loads(s3.puts[-1]["Body"])
    assert manifest == {
        "commit": COMMIT,
        "merged_at": MERGED_AT,
        "changed": ["help/a.md", "help/sub/b.md"],
        "deleted": ["help/old.md"],
    }
    assert result.manifest_key == "system-docs/_manifests/9f2c1ab.json"
    assert all(put["Bucket"] == BUCKET for put in s3.puts)


def test_merge_without_help_changes_uploads_nothing(tmp_path: Path) -> None:
    write(tmp_path, "src/readme.md")
    s3 = FakeS3()

    result = publish(tmp_path, s3, ["src/readme.md"], ["docs/old.md"])

    assert s3.puts == []
    assert result.manifest_key is None


def test_non_markdown_and_oversized_files_are_skipped_and_listed(tmp_path: Path) -> None:
    write(tmp_path, "help/ok.md")
    write(tmp_path, "help/diagram.png", b"\x89PNG")
    write(tmp_path, "help/huge.md", b"x" * (MAX_DOC_BYTES + 1))
    write(tmp_path, "help/limit.md", b"x" * MAX_DOC_BYTES)
    s3 = FakeS3()

    result = publish(tmp_path, s3, ["help/ok.md", "help/diagram.png", "help/huge.md", "help/limit.md"], [])

    manifest = json.loads(s3.puts[-1]["Body"])
    assert manifest["changed"] == ["help/ok.md", "help/limit.md"]
    assert sorted(result.skipped) == ["help/diagram.png", "help/huge.md"]


def test_path_escaping_help_is_skipped(tmp_path: Path) -> None:
    write(tmp_path, "secret.md")
    s3 = FakeS3()

    result = publish(tmp_path, s3, ["help/../secret.md"], [])

    assert s3.puts == []
    assert result.manifest_key is None


def test_failed_upload_writes_no_manifest_and_fails_the_job(tmp_path: Path) -> None:
    write(tmp_path, "help/a.md")
    write(tmp_path, "help/b.md")
    s3 = FakeS3(fail_on="help/b.md")

    with pytest.raises(UploadFailed) as raised:
        publish(tmp_path, s3, ["help/a.md", "help/b.md"], ["help/old.md"])

    assert not any("_manifests" in put["Key"] for put in s3.puts)
    assert "secret-internal-detail" not in str(raised.value)


def test_git_name_status_output_becomes_changed_and_deleted_lists() -> None:
    output = "A\thelp/new.md\nM\thelp/edited.md\nD\thelp/gone.md\n"

    assert parse_name_status(output) == (["help/new.md", "help/edited.md"], ["help/gone.md"])
