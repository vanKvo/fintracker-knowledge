"""KN-SDOC-02: manifest parsing and the idempotent execution start.

__author__ = "Van Vo"
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from botocore.exceptions import ClientError

from src.system_docs.exceptions import InvalidManifest
from src.system_docs.trigger import parse_manifest, start_ingestion

MANIFEST = {
    "commit": "9f2c1ab",
    "merged_at": "2026-10-20T18:04:00Z",
    "changed": ["help/statements/csv-mapping.md"],
    "deleted": ["help/legacy/old-import.md"],
}


class FakeStepFunctions:
    def __init__(self) -> None:
        self.started: dict[str, dict[str, Any]] = {}

    def start_execution(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs["name"] in self.started:
            raise ClientError({"Error": {"Code": "ExecutionAlreadyExists", "Message": "x"}}, "StartExecution")
        self.started[kwargs["name"]] = kwargs
        return {"executionArn": "arn:aws:states:::execution:" + kwargs["name"]}


def test_manifest_is_parsed_into_changed_and_deleted_paths() -> None:
    manifest = parse_manifest(json.dumps(MANIFEST).encode())

    assert manifest.commit == "9f2c1ab"
    assert manifest.changed == ("help/statements/csv-mapping.md",)
    assert manifest.deleted == ("help/legacy/old-import.md",)


@pytest.mark.parametrize(
    "broken",
    [b"not json", b"[]", json.dumps({**MANIFEST, "commit": "../etc"}).encode(), json.dumps({"commit": "9f2c1ab"}).encode()],
    ids=["not-json", "not-an-object", "bad-commit", "missing-lists"],
)
def test_malformed_manifest_is_rejected(broken: bytes) -> None:
    with pytest.raises(InvalidManifest):
        parse_manifest(broken)


def test_manifest_event_starts_one_execution_named_after_the_commit() -> None:
    sfn = FakeStepFunctions()

    started = start_ingestion(parse_manifest(json.dumps(MANIFEST).encode()), sfn, "arn:state-machine")

    assert started is True
    call = sfn.started["9f2c1ab"]
    assert call["stateMachineArn"] == "arn:state-machine"
    assert json.loads(call["input"])["changed"] == ["help/statements/csv-mapping.md"]


def test_replayed_manifest_event_starts_no_second_execution() -> None:
    sfn = FakeStepFunctions()
    manifest = parse_manifest(json.dumps(MANIFEST).encode())

    assert start_ingestion(manifest, sfn, "arn:state-machine") is True
    assert start_ingestion(manifest, sfn, "arn:state-machine") is False
    assert len(sfn.started) == 1


def test_other_start_errors_are_not_swallowed() -> None:
    class Denied:
        def start_execution(self, **kwargs: Any) -> dict[str, Any]:
            raise ClientError({"Error": {"Code": "AccessDeniedException", "Message": "x"}}, "StartExecution")

    with pytest.raises(ClientError):
        start_ingestion(parse_manifest(json.dumps(MANIFEST).encode()), Denied(), "arn:state-machine")
