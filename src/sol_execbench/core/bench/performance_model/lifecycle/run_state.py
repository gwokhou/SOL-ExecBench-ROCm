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

from pydantic import Field, model_validator

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticAttemptFailureCode,
    DiagnosticAttemptStatus,
    DiagnosticEvidencePurpose,
    DiagnosticLifecycleStage,
    DiagnosticStageStatus,
)
from sol_execbench.core.bench.performance_model.lifecycle.identity import (
    collection_run_id as derive_collection_run_id,
)
from sol_execbench.core.bench.performance_model.lifecycle.schema_versions import (
    DiagnosticLifecycleSchema,
    DiagnosticLifecycleStateKind,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
    DiagnosticLifecycleParent,
    GpuLifecycleIdentity,
    require_complete_gpu_identity,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import (
    attempts_dir,
    orchestrations_dir,
)
from sol_execbench.core.data.base_model import (
    CurrentFrozenSchemaModel,
    FrozenArtifactModel,
)
from sol_execbench.core.integrity import SHA256Digest, stable_json_checksum


def _exclude_none(value: object) -> bool:
    return value is None


class DiagnosticRunStageState(FrozenArtifactModel):
    """Progress recorded for one stage of a lifecycle run."""

    stage: DiagnosticLifecycleStage
    status: DiagnosticStageStatus
    attempts: int = Field(ge=0)
    receipt_path: str = ""
    outputs: tuple[DiagnosticLifecycleArtifact, ...] = ()


class DiagnosticStageAttempt(CurrentFrozenSchemaModel):
    """One immutable append-only execution attempt for a lifecycle stage."""

    current_schema_version = (
        DiagnosticLifecycleSchema.DIAGNOSTIC_LIFECYCLE_STATE
    )
    current_artifact_kind = DiagnosticLifecycleStateKind.ATTEMPT

    schema_version: Literal[
        DiagnosticLifecycleSchema.DIAGNOSTIC_LIFECYCLE_STATE
    ] = DiagnosticLifecycleSchema.DIAGNOSTIC_LIFECYCLE_STATE
    artifact_kind: Literal[DiagnosticLifecycleStateKind.ATTEMPT] = (
        DiagnosticLifecycleStateKind.ATTEMPT
    )
    run_id: SHA256Digest
    stage: DiagnosticLifecycleStage
    attempt: int = Field(ge=1)
    status: DiagnosticAttemptStatus
    started_at: str = Field(min_length=1)
    finished_at: str = Field(min_length=1)
    failure_code: DiagnosticAttemptFailureCode | None = None
    detail: str = Field(default="", max_length=4096)

    @model_validator(mode="after")
    def _failure_fields_match_status(self) -> DiagnosticStageAttempt:
        if self.status is DiagnosticAttemptStatus.FAILED:
            if self.failure_code is None or not self.detail:
                raise ValueError("failed attempt requires code and detail")
        elif self.failure_code is not None or self.detail:
            raise ValueError("verified attempt cannot carry failure fields")
        return self


class DiagnosticLifecyclePlan(CurrentFrozenSchemaModel):
    """One immutable, reviewable set of lifecycle execution inputs."""

    current_schema_version = (
        DiagnosticLifecycleSchema.DIAGNOSTIC_LIFECYCLE_STATE
    )
    current_artifact_kind = DiagnosticLifecycleStateKind.PLAN

    schema_version: Literal[
        DiagnosticLifecycleSchema.DIAGNOSTIC_LIFECYCLE_STATE
    ] = DiagnosticLifecycleSchema.DIAGNOSTIC_LIFECYCLE_STATE
    artifact_kind: Literal[DiagnosticLifecycleStateKind.PLAN] = (
        DiagnosticLifecycleStateKind.PLAN
    )
    plan_id: SHA256Digest
    design: DiagnosticLifecycleParent
    development_snapshot: DiagnosticLifecycleParent
    collection_root: str = Field(min_length=1)
    collection_inventory: tuple[DiagnosticLifecycleArtifact, ...] = Field(
        min_length=1
    )
    collection_run_id: SHA256Digest
    gpu_identity: GpuLifecycleIdentity | None = None
    generation: int = Field(ge=1)
    roles: tuple[Literal["held_out"], ...] = ("held_out",)
    calibration_profile_path: str = Field(min_length=1)
    calibration_profile: DiagnosticLifecycleArtifact
    calibration_audit_path: str = Field(min_length=1)
    calibration_audit: DiagnosticLifecycleArtifact
    held_out_corpus_path: str = Field(min_length=1)
    held_out_corpus: DiagnosticLifecycleArtifact
    output_root: str = Field(min_length=1)
    source_revision: str = Field(min_length=40, max_length=64)
    purpose: DiagnosticEvidencePurpose
    model_version: str = Field(min_length=1)
    max_attempts: int = Field(ge=1, le=10)
    vram_policy_path: str | None = Field(default=None, exclude_if=_exclude_none)
    vram_policy: DiagnosticLifecycleArtifact | None = Field(
        default=None, exclude_if=_exclude_none
    )
    frozen_inference_profile_path: str | None = Field(
        default=None, exclude_if=_exclude_none
    )
    frozen_inference_profile: DiagnosticLifecycleArtifact | None = Field(
        default=None, exclude_if=_exclude_none
    )

    @model_validator(mode="after")
    def _identities_are_canonical(self) -> DiagnosticLifecyclePlan:
        if self.design.stage is not DiagnosticLifecycleStage.DESIGN:
            raise ValueError("lifecycle plan design reference has wrong stage")
        if (
            self.development_snapshot.stage
            is not DiagnosticLifecycleStage.CORPUS_SNAPSHOT
        ):
            raise ValueError(
                "lifecycle plan development reference has wrong stage"
            )
        if (
            self.design.purpose is not self.purpose
            or self.development_snapshot.purpose is not self.purpose
        ):
            raise ValueError("lifecycle plan references cross purpose domains")
        if self.roles != ("held_out",):
            raise ValueError("production lifecycle collection is held-out only")
        if (self.vram_policy_path is None) != (self.vram_policy is None):
            raise ValueError("VRAM policy path and artifact must be paired")
        if (self.frozen_inference_profile_path is None) != (
            self.frozen_inference_profile is None
        ):
            raise ValueError(
                "frozen inference path and artifact must be paired"
            )
        if self.purpose is DiagnosticEvidencePurpose.PRODUCTION:
            require_complete_gpu_identity(
                self.gpu_identity,
                stage=DiagnosticLifecycleStage.COLLECTION_RUN,
                require_pcie_topology=True,
            )
        paths = tuple(item.relative_path for item in self.collection_inventory)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("collection inventory must be sorted and unique")
        expected_run_id = derive_collection_run_id(
            design_id=self.design.stage_id,
            generation=self.generation,
            roles=self.roles,
            frozen_held_out_sha256=self.held_out_corpus.sha256,
            gpu_identity=self.gpu_identity,
            source_revision=self.source_revision,
            purpose=self.purpose,
        )
        if self.collection_run_id != expected_run_id:
            raise ValueError(
                "lifecycle plan collection_run_id is not canonical"
            )
        expected_plan_id = stable_json_checksum(
            diagnostic_lifecycle_plan_payload(self)
        )
        if self.plan_id != expected_plan_id:
            raise ValueError("lifecycle plan plan_id is not canonical")
        return self


def diagnostic_lifecycle_plan_payload(
    plan: DiagnosticLifecyclePlan,
) -> dict[str, object]:
    """Return the canonical plan-ID payload with additive identity support."""
    payload = plan.model_dump(mode="json", exclude={"plan_id"})
    if plan.gpu_identity is None:
        payload.pop("gpu_identity", None)
    for field in (
        "vram_policy_path",
        "vram_policy",
        "frozen_inference_profile_path",
        "frozen_inference_profile",
    ):
        if getattr(plan, field) is None:
            payload.pop(field, None)
    return payload


class DiagnosticRunManifest(CurrentFrozenSchemaModel):
    """One resumable lifecycle run-state object.

    The run-state carries the collection generation identity, the design it
    belongs to, and one immutable progress entry per DAG stage with bounded
    attempts and the typed receipt location. Resume and status re-verify each
    recorded entry rather than trusting file existence.
    """

    current_schema_version = (
        DiagnosticLifecycleSchema.DIAGNOSTIC_LIFECYCLE_STATE
    )
    current_artifact_kind = DiagnosticLifecycleStateKind.RUN

    schema_version: Literal[
        DiagnosticLifecycleSchema.DIAGNOSTIC_LIFECYCLE_STATE
    ] = DiagnosticLifecycleSchema.DIAGNOSTIC_LIFECYCLE_STATE
    artifact_kind: Literal[DiagnosticLifecycleStateKind.RUN] = (
        DiagnosticLifecycleStateKind.RUN
    )
    run_id: str = Field(min_length=1)
    collection_run_id: SHA256Digest
    design_id: SHA256Digest
    generation: int = Field(ge=1)
    purpose: DiagnosticEvidencePurpose = DiagnosticEvidencePurpose.PRODUCTION
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)
    plan_id: SHA256Digest
    plan_sha256: SHA256Digest
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
    return orchestrations_dir(store_root_path) / collection_run_id / "run.json"


def lifecycle_plan_path(
    collection_run_id: str,
    store_root_path: Path | None = None,
) -> Path:
    """Return the immutable reviewed plan copy for one orchestration."""
    return orchestrations_dir(store_root_path) / collection_run_id / "plan.json"


def stage_receipt_path(
    collection_run_id: str,
    stage: DiagnosticLifecycleStage,
    store_root_path: Path | None = None,
) -> Path:
    """Return the typed receipt file path for one stage of a run."""
    return (
        orchestrations_dir(store_root_path)
        / collection_run_id
        / "receipts"
        / f"{stage.value}.json"
    )


def stage_attempt_path(
    collection_run_id: str,
    stage: DiagnosticLifecycleStage,
    attempt: int,
    store_root_path: Path | None = None,
) -> Path:
    """Return the immutable event path for one numbered stage attempt."""
    return (
        attempts_dir(store_root_path)
        / collection_run_id
        / stage.value
        / f"{attempt:04d}.json"
    )


__all__ = [
    "DiagnosticLifecyclePlan",
    "DiagnosticRunManifest",
    "DiagnosticRunStageState",
    "DiagnosticStageAttempt",
    "diagnostic_lifecycle_plan_payload",
    "lifecycle_plan_path",
    "run_state_path",
    "stage_attempt_path",
    "stage_receipt_path",
]
