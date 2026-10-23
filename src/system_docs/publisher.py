"""KN-SDOC-01: CI uploads changed help docs, then a manifest, after a merge to main.

Runs in CI under its OIDC deploy role, the only role the bucket policy lets write `system-docs/`.
The manifest goes last, so a failed upload starts no ingestion.

__author__ = "Van Vo"
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.system_docs.source import MANIFEST_PREFIX, PREFIX

logger = logging.getLogger(__name__)

HELP_PREFIX = "help/"
MAX_DOC_BYTES = 200 * 1024


class UploadFailed(Exception):
    def __init__(self) -> None:
        super().__init__("Uploading the help docs failed; no manifest was written.")


@dataclass(frozen=True)
class PublishResult:
    uploaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    manifest_key: str | None = None


def parse_name_status(output: str) -> tuple[list[str], list[str]]:
    """`git diff --name-status --no-renames` output -> (changed, deleted)."""
    changed: list[str] = []
    deleted: list[str] = []
    for line in output.splitlines():
        status, _, path = line.partition("\t")
        if not path:
            continue
        (deleted if status.startswith("D") else changed).append(path)
    return changed, deleted


def _under_help(path: str) -> bool:
    return path.startswith(HELP_PREFIX)


def _safe(path: str) -> bool:
    return ".." not in path.split("/") and path.endswith(".md")


def publish_system_docs(
    *,
    root: Path,
    changed: list[str],
    deleted: list[str],
    commit: str,
    merged_at: str,
    s3: Any,
    bucket: str,
) -> PublishResult:
    skipped: list[str] = []
    to_upload: list[tuple[str, bytes]] = []
    for path in dict.fromkeys(p for p in changed if _under_help(p)):
        data = (root / path).read_bytes() if _safe(path) and (root / path).is_file() else None
        if data is None or len(data) > MAX_DOC_BYTES:
            skipped.append(path)
        else:
            to_upload.append((path, data))
    removed = [p for p in dict.fromkeys(deleted) if _under_help(p) and _safe(p)]
    for path in skipped:
        logger.warning("Skipped %s (not a markdown file under the size limit)", path)

    if not to_upload and not removed:
        return PublishResult(skipped=skipped)

    try:
        for path, data in to_upload:
            s3.put_object(Bucket=bucket, Key=PREFIX + path, Body=data, ContentType="text/markdown")
        manifest = {"commit": commit, "merged_at": merged_at, "changed": [p for p, _ in to_upload], "deleted": removed}
        key = f"{MANIFEST_PREFIX}{commit}.json"
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(manifest).encode(), ContentType="application/json")
    except Exception as exc:  # noqa: BLE001 - detail stays in the CI log; the raised error is generic
        logger.error("Help-doc upload failed: %s", type(exc).__name__)
        raise UploadFailed() from None
    return PublishResult(uploaded=[p for p, _ in to_upload], skipped=skipped, manifest_key=key)


def main() -> None:
    """CI entry point: python -m src.system_docs.publisher --bucket B --commit SHA --merged-at TS --base SHA"""
    import boto3

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--merged-at", required=True)
    parser.add_argument("--base", required=True, help="commit to diff against (the previous main)")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    diff = subprocess.run(
        ["git", "diff", "--name-status", "--no-renames", args.base, args.commit, "--", HELP_PREFIX],
        cwd=args.root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    changed, deleted = parse_name_status(diff)
    result = publish_system_docs(
        root=Path(args.root),
        changed=changed,
        deleted=deleted,
        commit=args.commit,
        merged_at=args.merged_at,
        s3=boto3.client("s3"),
        bucket=args.bucket,
    )
    print(f"uploaded={len(result.uploaded)} skipped={result.skipped} manifest={result.manifest_key}")


if __name__ == "__main__":
    main()
