# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle run-state object persisted under the store.

The run-state object records per-stage progress for one collection
generation. ``verified`` is only assigned after a verifier re-checks the
typed receipt, every input identity, and the exact output inventory; the
presence of an output filename is never proof of completion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticLifecycleStage,
    DiagnosticStageStatus,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import (
    runs_dir,
)
from sol_execbench.core.data.base_model import (
    CurrentFrozenSchemaModel,
    FrozenArtifactModel,
)
from sol_execbench.core.integrity import SHA256Digest
from sol_execbench.core.integrity.schema_versions import SchemaVersion


class DiagnosticRunStageState(FrozenArtifactModel):
    """Progress recorded for one stage of a lifecycle run."""

    stage: DiagnosticLifecycleStage
    status: DiagnosticStageStatus
    attempts: int = Field(ge=0)
    receipt_path: str = ""
    outputs: tuple[DiagnosticLifecycleArtifact, ...] = ()


class DiagnosticRunManifest(CurrentFrozenSchemaModel):
    """One resumable lifecycle run-state object.

    The run-state carries the collection generation identity, the design it
    belongs to, and one immutable progress entry per DAG stage with bounded
    attempts and the typed receipt location. Resume and status re-verify each
    recorded entry rather than trusting file existence.
    """

    current_schema_version = SchemaVersion.DIAGNOSTIC_LIFECYCLE_RUN

    schema_version: Literal[SchemaVersion.DIAGNOSTIC_LIFECYCLE_RUN] = (
        SchemaVersion.DIAGNOSTIC_LIFECYCLE_RUN
    )
    run_id: str = Field(min_length=1)
    collection_run_id: SHA256Digest
    design_id: SHA256Digest
    generation: int = Field(ge=1)
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)
    design_manifest_path: str = ""
    inputs: dict[str, str] = Field(default_factory=dict)
    stages: tuple[DiagnosticRunStageState, ...] = ()

    def stage_state(
        self,
        stage: DiagnosticLifecycleStage,
    ) -> DiagnosticRunStageState | None:
        """Return the recorded progress for one stage, if present."""
        for item in self.stages:
            if item.stage is stage:
                return item
        return None

    def set_stage(
        self, state: DiagnosticRunStageState
    ) -> DiagnosticRunManifest:
        """Return a copy with one stage's progress replaced or appended."""
        remaining = tuple(
            item for item in self.stages if item.stage is not state.stage
        )
        return self.model_copy(
            update={"stages": (*remaining, state)},
        )


def run_state_path(
    collection_run_id: str,
    store_root_path: Path | None = None,
) -> Path:
    """Return the run-state file path for one collection generation."""
    return runs_dir(store_root_path) / collection_run_id / "run.json"


def stage_receipt_path(
    collection_run_id: str,
    stage: DiagnosticLifecycleStage,
    store_root_path: Path | None = None,
) -> Path:
    """Return the typed receipt file path for one stage of a run."""
    return (
        runs_dir(store_root_path)
        / collection_run_id
        / "receipts"
        / f"{stage.value}.json"
    )


__all__ = [
    "DiagnosticRunManifest",
    "DiagnosticRunStageState",
    "run_state_path",
    "stage_receipt_path",
]
