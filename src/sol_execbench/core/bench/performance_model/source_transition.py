# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Stage-scoped source transitions and development-evidence rebinding."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.data.base_model import (
    CurrentFrozenSchemaModel,
    FrozenArtifactModel,
    NonEmptyString,
)
from sol_execbench.core.integrity import SHA256Digest
from sol_execbench.core.integrity.schema_versions import SchemaVersion

_REUSABLE_ARTIFACT_PATHS = (
    "calibration/cal.audit.json",
    "calibration/cal.json",
    "corpus/design.json",
    "vram-policy.json",
)
_QUALIFICATION_GATE_PATHS = (
    "development/canary/gate.json",
    "development/full/gate.json",
    "static/gate.json",
)


class SourceTransitionStage(StrEnum):
    """Experimental and governance stages considered by a source review."""

    VRAM_POLICY = "vram_policy"
    CALIBRATION = "calibration"
    DESIGN = "design"
    QUALIFICATION_STATIC = "qualification_static"
    QUALIFICATION_GPU = "qualification_gpu"
    RAW_COLLECTION = "raw_collection"
    SOLAR = "solar"
    INFERENCE = "inference"
    ACCEPTANCE = "acceptance"
    PUBLICATION = "publication"
    GOVERNANCE_CONTROL_PLANE = "governance_control_plane"


class SourceTransitionDisposition(StrEnum):
    """Whether a stage's experiment-relevant semantics changed."""

    UNCHANGED = "unchanged"
    CHANGED = "changed"


class SourcePathStageImpact(FrozenArtifactModel):
    """Reviewed stage effects for one exact Git name-status entry."""

    path: NonEmptyString
    previous_path: NonEmptyString | None = None
    change: Literal["added", "modified", "deleted", "renamed"]
    affected_stages: tuple[SourceTransitionStage, ...] = ()
    rationale: NonEmptyString

    @model_validator(mode="after")
    def _fields_are_canonical(self) -> SourcePathStageImpact:
        if (self.change == "renamed") != (self.previous_path is not None):
            raise ValueError("only renamed paths carry previous_path")
        values = tuple(stage.value for stage in self.affected_stages)
        if values != tuple(sorted(values)) or len(values) != len(set(values)):
            raise ValueError("affected stages must be sorted and unique")
        return self


class SourceStageDecision(FrozenArtifactModel):
    """Final disposition of one stage after reviewing every changed path."""

    stage: SourceTransitionStage
    disposition: SourceTransitionDisposition
    rationale: NonEmptyString


class SemanticProjectionPair(FrozenArtifactModel):
    """Before/after digest of a deliberately scoped semantic projection."""

    base_sha256: SHA256Digest
    comparison_sha256: SHA256Digest

    @property
    def unchanged(self) -> bool:
        """Return whether the two independently computed projections match."""
        return self.base_sha256 == self.comparison_sha256


class QualificationTimeoutTransition(FrozenArtifactModel):
    """Monotonic whole-batch qualification watchdog change."""

    old_seconds: int = Field(gt=0)
    new_seconds: int = Field(gt=0)
    completed_before_old_timeout_semantics_preserved: Literal[True] = True

    @model_validator(mode="after")
    def _timeout_only_expands(self) -> QualificationTimeoutTransition:
        if self.new_seconds <= self.old_seconds:
            raise ValueError("qualification timeout transition must increase")
        return self


class DiagnosticSourceTransitionAttestation(CurrentFrozenSchemaModel):
    """Exact diff plus semantic proofs for reusing unaffected evidence."""

    current_schema_version = SchemaVersion.DIAGNOSTIC_SOURCE_TRANSITION

    schema_version: Literal[SchemaVersion.DIAGNOSTIC_SOURCE_TRANSITION] = (
        SchemaVersion.DIAGNOSTIC_SOURCE_TRANSITION
    )
    purpose: DiagnosticEvidencePurpose = DiagnosticEvidencePurpose.PRODUCTION
    base_source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    comparison_artifact_source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    target_source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    base_policy_path: NonEmptyString
    comparison_policy_path: NonEmptyString
    base_calibration_profile_path: NonEmptyString
    base_calibration_audit_path: NonEmptyString
    base_design_path: NonEmptyString
    comparison_design_path: NonEmptyString
    base_problems_root: NonEmptyString
    comparison_problems_root: NonEmptyString
    git_patch_sha256: SHA256Digest
    source_changes: tuple[SourcePathStageImpact, ...] = Field(min_length=1)
    stage_decisions: tuple[SourceStageDecision, ...]
    policy_behavior_projection: SemanticProjectionPair
    design_behavior_projection: SemanticProjectionPair
    problems_tree_projection: SemanticProjectionPair
    raw_collection_projection: SemanticProjectionPair
    qualification_timeout: QualificationTimeoutTransition
    reusable_artifacts: tuple[DiagnosticLifecycleArtifact, ...] = Field(
        min_length=4
    )
    base_problems_inventory: tuple[DiagnosticLifecycleArtifact, ...] = Field(
        min_length=1
    )
    comparison_problems_inventory: tuple[DiagnosticLifecycleArtifact, ...] = (
        Field(min_length=1)
    )
    calibration_validated_by_current_loader: Literal[True] = True
    created_at: NonEmptyString

    @model_validator(mode="after")
    def _review_is_complete(self) -> DiagnosticSourceTransitionAttestation:
        _require_sorted_paths(self.source_changes)
        decisions = {item.stage: item for item in self.stage_decisions}
        decision_order = tuple(item.stage for item in self.stage_decisions)
        if len(decisions) != len(self.stage_decisions) or set(decisions) != set(
            SourceTransitionStage
        ):
            raise ValueError("stage decisions must cover every stage exactly")
        if decision_order != tuple(SourceTransitionStage):
            raise ValueError("stage decisions must use canonical stage order")
        impacted = {
            stage
            for change in self.source_changes
            for stage in change.affected_stages
        }
        for stage, decision in decisions.items():
            expected = (
                SourceTransitionDisposition.CHANGED
                if stage in impacted
                else SourceTransitionDisposition.UNCHANGED
            )
            if decision.disposition is not expected:
                raise ValueError("stage disposition disagrees with source diff")
        _require_unchanged_stage_proofs(self, decisions)
        _require_sorted_inventory(self.reusable_artifacts, "reusable")
        if (
            tuple(item.relative_path for item in self.reusable_artifacts)
            != _REUSABLE_ARTIFACT_PATHS
        ):
            raise ValueError("reusable inventory must cite the exact r1 bundle")
        _require_sorted_inventory(self.base_problems_inventory, "base problems")
        _require_sorted_inventory(
            self.comparison_problems_inventory, "comparison problems"
        )
        return self


class DevelopmentCaseRebind(FrozenArtifactModel):
    """One raw development case copied under an exact transition proof."""

    case_id: NonEmptyString
    workload_kind: WorkloadKind
    phase: Literal["point_fit", "conformal"]
    workload_uuid: NonEmptyString
    evidence_manifest_sha256: SHA256Digest
    inventory: tuple[DiagnosticLifecycleArtifact, ...] = Field(min_length=1)


class DiagnosticDevelopmentCaseRebindReceipt(CurrentFrozenSchemaModel):
    """Immutable receipt for verified, non-overwriting raw-case rebinding."""

    current_schema_version = SchemaVersion.DIAGNOSTIC_DEVELOPMENT_CASE_REBIND

    schema_version: Literal[
        SchemaVersion.DIAGNOSTIC_DEVELOPMENT_CASE_REBIND
    ] = SchemaVersion.DIAGNOSTIC_DEVELOPMENT_CASE_REBIND
    purpose: DiagnosticEvidencePurpose = DiagnosticEvidencePurpose.PRODUCTION
    transition_attestation_sha256: SHA256Digest
    base_source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    target_source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_design_sha256: SHA256Digest
    target_design_sha256: SHA256Digest
    source_root: NonEmptyString
    target_root: NonEmptyString
    qualification_root: NonEmptyString
    qualification_gates: tuple[DiagnosticLifecycleArtifact, ...] = Field(
        min_length=3, max_length=3
    )
    cases: tuple[DevelopmentCaseRebind, ...] = Field(min_length=1)
    created_at: NonEmptyString

    @model_validator(mode="after")
    def _cases_are_canonical(self) -> DiagnosticDevelopmentCaseRebindReceipt:
        case_ids = tuple(item.case_id for item in self.cases)
        if case_ids != tuple(sorted(case_ids)) or len(case_ids) != len(
            set(case_ids)
        ):
            raise ValueError("rebound case IDs must be sorted and unique")
        for case in self.cases:
            _require_sorted_inventory(case.inventory, case.case_id)
        _require_sorted_inventory(self.qualification_gates, "qualification")
        if (
            tuple(item.relative_path for item in self.qualification_gates)
            != _QUALIFICATION_GATE_PATHS
        ):
            raise ValueError("rebind receipt requires the exact three gates")
        return self


def _require_sorted_paths(changes: tuple[SourcePathStageImpact, ...]) -> None:
    paths = tuple(item.path for item in changes)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise ValueError("source change paths must be sorted and unique")


def _require_sorted_inventory(
    inventory: tuple[DiagnosticLifecycleArtifact, ...], label: str
) -> None:
    paths = tuple(item.relative_path for item in inventory)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise ValueError(f"{label} inventory must be sorted and unique")


def _require_unchanged_stage_proofs(
    attestation: DiagnosticSourceTransitionAttestation,
    decisions: dict[SourceTransitionStage, SourceStageDecision],
) -> None:
    required = {
        SourceTransitionStage.VRAM_POLICY: attestation.policy_behavior_projection,
        SourceTransitionStage.DESIGN: attestation.design_behavior_projection,
        SourceTransitionStage.RAW_COLLECTION: (
            attestation.raw_collection_projection
        ),
    }
    for stage, projection in required.items():
        if (
            decisions[stage].disposition
            is SourceTransitionDisposition.UNCHANGED
            and not projection.unchanged
        ):
            raise ValueError(f"unchanged {stage.value} lacks matching proof")
    if decisions[
        SourceTransitionStage.DESIGN
    ].disposition is SourceTransitionDisposition.UNCHANGED and (
        not attestation.problems_tree_projection.unchanged
        or attestation.base_problems_inventory
        != attestation.comparison_problems_inventory
    ):
        raise ValueError("unchanged design lacks identical prepared problems")


__all__ = [
    "DevelopmentCaseRebind",
    "DiagnosticDevelopmentCaseRebindReceipt",
    "DiagnosticSourceTransitionAttestation",
    "QualificationTimeoutTransition",
    "SemanticProjectionPair",
    "SourcePathStageImpact",
    "SourceStageDecision",
    "SourceTransitionDisposition",
    "SourceTransitionStage",
]
