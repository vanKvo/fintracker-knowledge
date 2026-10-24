"""KN-SDOC-02: read a manifest and start its ingestion execution once.

The execution is named after the commit, so replaying the manifest's storage event finds the name
already used and starts nothing.

__author__ = "Van Vo"
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from botocore.exceptions import ClientError

from src.system_docs.exceptions import InvalidManifest
from src.system_docs.schemas import Manifest

logger = logging.getLogger(__name__)

_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")  # also a valid execution name


class StepFunctions(Protocol):
    def start_execution(self, **kwargs: Any) -> dict[str, Any]: ...


def parse_manifest(raw: bytes) -> Manifest:
    try:
        data = json.loads(raw)
        commit, merged_at = data["commit"], data["merged_at"]
        changed, deleted = data["changed"], data["deleted"]
    except (ValueError, KeyError, TypeError):
        raise InvalidManifest() from None
    lists_ok = all(isinstance(paths, list) and all(isinstance(p, str) for p in paths) for paths in (changed, deleted))
    if not (isinstance(commit, str) and _COMMIT.match(commit) and isinstance(merged_at, str) and lists_ok):
        raise InvalidManifest()
    return Manifest(commit, merged_at, tuple(changed), tuple(deleted))


def start_ingestion(manifest: Manifest, client: StepFunctions, state_machine_arn: str) -> bool:
    """True when a new execution started; False when this commit already has one."""
    payload = {
        "commit": manifest.commit,
        "merged_at": manifest.merged_at,
        "changed": list(manifest.changed),
        "deleted": list(manifest.deleted),
    }
    try:
        client.start_execution(stateMachineArn=state_machine_arn, name=manifest.commit, input=json.dumps(payload))
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ExecutionAlreadyExists":
            logger.info("Ingestion already started for commit %s", manifest.commit)
            return False
        raise
    return True
