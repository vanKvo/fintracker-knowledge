"""KN-CHUNK-03: switch the active embedding model only when a full rebuild is complete and good.

The rebuild (KN-SDOC-05) writes the new model's rows alongside the old; retrieval keeps using
the active model until the setting flips, so a query never meets two models' vectors.

__author__ = "Van Vo"
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from src.model_switch.models import ModelSwitchRefused
from src.model_switch.repository import ActiveModelSetting, ModelIndex
from src.retrieval_quality.models import QualityGateFailed

RebuildStatus = Literal["succeeded", "partial", "failed"]


def switch_embedding_model(
    target_model: str,
    *,
    rebuild_status: RebuildStatus,
    index: ModelIndex,
    setting: ActiveModelSetting,
    quality_gate: Callable[[], object],
) -> None:
    """`quality_gate` runs KN-CHUNK-04 against the new vectors and raises QualityGateFailed."""
    if target_model == setting.get():
        return
    if rebuild_status == "failed":
        index.delete_model_rows(target_model)
        raise ModelSwitchRefused("The rebuild failed; the embedding model was not switched.")

    # A partial rebuild lands here too: its failed docs have no chunks under the target model.
    missing = index.docs_missing_model(target_model)
    if missing:
        raise ModelSwitchRefused(
            f"{len(missing)} doc(s) have no chunks under the target model: {', '.join(missing)}", missing
        )

    try:
        quality_gate()
    except QualityGateFailed:
        index.delete_model_rows(target_model)
        raise

    setting.set(target_model)
    index.delete_other_models_rows(target_model)
