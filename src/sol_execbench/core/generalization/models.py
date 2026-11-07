# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-owned contracts for cross-hardware Agent evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from sol_execbench.core.data.base_model import (
    CurrentFrozenSchemaModel,
    FrozenArtifactModel,
    NonEmptyString,
    NonNegativeInt,
)
from sol_execbench.core.dataset.corpus_models import (
    CorpusOperationFamily,
    CorpusProfile,
    GeneratedWorkloadRecord,
    WorkloadRegime,
    WorkloadRole,
)
from sol_execbench.core.dataset.schema_versions import DatasetArtifactSchema

SHA256_PATTERN = r"^[0-9a-f]{64}$"
FAST_THRESHOLDS = (1.0, 1.1, 1.2, 1.5, 2.0)
BOOTSTRAP_REPLICATES = 10_000


class AgentTrack(StrEnum):
    """Candidate protocol evaluated by a cell."""

    TARGET_CONDITIONED = "target_conditioned"
    SOLUTION_PORTABILITY = "solution_portability"


class HardwareContextView(StrEnum):
    """Identity visibility in the Agent-facing target facts."""

    FULL_FACTS = "full_facts"
    ANONYMIZED_FACTS = "anonymized_facts"


class HardwareShift(StrEnum):
    """Target class derived from the minimal training exposure declaration."""

    SEEN_CONFIGURATION = "seen_configuration"
    SAME_ISA_NEW_CAPACITY = "same_isa_new_capacity"
    SAME_ISA_NEW_CONFIGURATION = "same_isa_new_configuration"
    UNSEEN_ARCHITECTURE = "unseen_architecture"


class GeneralizationArtifactKind(StrEnum):
    """Variants of the hardware-generalization evidence family."""

    PLAN = "plan"
    CELL = "cell"
    REPORT = "report"


class GeneralizationReportStatus(StrEnum):
    """Whether an aggregate can support a generalization conclusion."""

    COMPLETE = "complete"
    DESCRIPTIVE = "descriptive"
    INCOMPLETE = "incomplete"


class CellResultStatus(StrEnum):
    """Normalized candidate and evaluator outcomes."""

    PASSED = "passed"
    INCORRECT = "incorrect"
    COMPILE_ERROR = "compile_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    CANDIDATE_OOM = "candidate_oom"
    REWARD_HACK = "reward_hack"
    MISSING_SOLUTION = "missing_solution"
    EVALUATOR_FAILURE = "evaluator_failure"


class TrainingHardwareExposure(FrozenArtifactModel):
    """One target tuple declared as visible during training."""

    gfx_target: NonEmptyString
    hardware_configuration_id: str = Field(pattern=SHA256_PATTERN)
    capacity_class_bytes: NonNegativeInt
    distribution_id: str = Field(pattern=SHA256_PATTERN)


class TrainingExposureDeclaration(FrozenArtifactModel):
    """Minimum external metadata needed to derive hardware-shift labels."""

    self_declared: bool = True
    hardware: tuple[TrainingHardwareExposure, ...] = ()

    @model_validator(mode="after")
    def _unique_hardware(self) -> TrainingExposureDeclaration:
        identities = {
            (
                item.gfx_target.strip().lower(),
                item.hardware_configuration_id,
                item.capacity_class_bytes,
                item.distribution_id,
            )
            for item in self.hardware
        }
        if len(identities) != len(self.hardware):
            raise ValueError("training exposure contains duplicate targets")
        return self


class NormalizedHardwareFacts(FrozenArtifactModel):
    """Only target facts already owned by workload generation."""

    study_target_id: NonEmptyString
    context_view: HardwareContextView
    context_digest: str = Field(pattern=SHA256_PATTERN)
    gfx_target: NonEmptyString | None
    target_id: NonEmptyString | None
    hardware_configuration_id: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    device_model: NonEmptyString | None = None
    product_sku: NonEmptyString | None = None
    configuration_kind: NonEmptyString | None = None
    visible_compute_units: int | None = Field(default=None, gt=0)
    visible_memory_bytes: int | None = Field(default=None, gt=0)
    partition: NonEmptyString | None = None
    virtualization: NonEmptyString | None = None
    isolation: NonEmptyString | None = None
    capacity_class_bytes: NonNegativeInt
    supported_dtypes: tuple[NonEmptyString, ...]
    supported_quantization: tuple[NonEmptyString, ...]
    capabilities: tuple[NonEmptyString, ...]
    max_tensor_bytes: int = Field(gt=0)
    reference_ipc_limit_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def _identity_boundary(self) -> NormalizedHardwareFacts:
        required_identities = (
            self.gfx_target,
            self.target_id,
            self.hardware_configuration_id,
            self.configuration_kind,
        )
        all_identities = (
            *required_identities,
            self.device_model,
            self.product_sku,
            self.visible_compute_units,
            self.visible_memory_bytes,
            self.partition,
            self.virtualization,
            self.isolation,
        )
        if self.context_view is HardwareContextView.FULL_FACTS and any(
            item is None for item in required_identities
        ):
            raise ValueError("full facts require target identity")
        if self.context_view is HardwareContextView.ANONYMIZED_FACTS and any(
            item is not None for item in all_identities
        ):
            raise ValueError("anonymous facts leak target identity")
        return self


class AgentDefinitionView(FrozenArtifactModel):
    """Definition, public rule identity, and development workloads."""

    semantic_id: NonEmptyString
    problem_name: NonEmptyString
    definition_path: NonEmptyString
    generation_rule_path: NonEmptyString
    generation_rule_sha256: str = Field(pattern=SHA256_PATTERN)
    operation_family: CorpusOperationFamily
    distribution_id: str = Field(pattern=SHA256_PATTERN)
    development_workloads: tuple[GeneratedWorkloadRecord, ...] = Field(
        min_length=4,
        max_length=4,
    )
    withheld_slot_ids: tuple[NonEmptyString, ...] = Field(
        min_length=4,
        max_length=4,
    )

    @model_validator(mode="after")
    def _development_only(self) -> AgentDefinitionView:
        if any(
            item.role is not WorkloadRole.DEVELOPMENT
            for item in self.development_workloads
        ):
            raise ValueError("Agent view may expose only development workloads")
        return self


class CorpusAgentView(CurrentFrozenSchemaModel):
    """Deterministic Agent projection of a complete evaluator target view."""

    current_schema_version = DatasetArtifactSchema.CORPUS_AGENT_VIEW

    schema_version: Literal[DatasetArtifactSchema.CORPUS_AGENT_VIEW]
    corpus_id: NonEmptyString
    release_id: NonEmptyString
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    hardware_facts: NormalizedHardwareFacts
    requested_profiles: tuple[CorpusProfile, ...]
    public_generator_version: NonEmptyString
    definitions: tuple[AgentDefinitionView, ...]
    agent_view_digest: str = Field(pattern=SHA256_PATTERN)


class CandidateDeclaration(FrozenArtifactModel):
    """Benchmark-relevant identity of one supplied solution bundle."""

    semantic_id: NonEmptyString
    solution_digest: str = Field(pattern=SHA256_PATTERN)
    portability_digest: str = Field(pattern=SHA256_PATTERN)
    agent_view_digest: str = Field(pattern=SHA256_PATTERN)
    hardware_context_digest: str = Field(pattern=SHA256_PATTERN)
    used_holdout_feedback: bool = False


class PlannedCell(FrozenArtifactModel):
    """One immutable target, track, and context evaluation cell."""

    cell_id: NonEmptyString
    study_target_id: NonEmptyString
    track: AgentTrack
    context_view: HardwareContextView
    shift: HardwareShift
    target_view_digest: str = Field(pattern=SHA256_PATTERN)
    generation_cohort_id: str = Field(pattern=SHA256_PATTERN)
    hardware_configuration_id: str = Field(pattern=SHA256_PATTERN)
    hardware_context_digest: str = Field(pattern=SHA256_PATTERN)
    agent_view_digest: str = Field(pattern=SHA256_PATTERN)
    zero_shot: bool = True


class HardwareGeneralizationPlan(CurrentFrozenSchemaModel):
    """Immutable benchmark execution matrix."""

    current_schema_version = DatasetArtifactSchema.HARDWARE_GENERALIZATION
    current_artifact_kind = GeneralizationArtifactKind.PLAN

    schema_version: Literal[DatasetArtifactSchema.HARDWARE_GENERALIZATION]
    artifact_kind: Literal[GeneralizationArtifactKind.PLAN]
    study_id: NonEmptyString
    exposure: TrainingExposureDeclaration
    corpus_manifest_digest: str = Field(pattern=SHA256_PATTERN)
    profile_set: tuple[CorpusProfile, ...] = Field(min_length=1)
    cells: tuple[PlannedCell, ...] = Field(min_length=1)
    bootstrap_replicates: Literal[10_000] = BOOTSTRAP_REPLICATES
    plan_digest: str = Field(pattern=SHA256_PATTERN)


class CellWorkloadResult(FrozenArtifactModel):
    """Normalized outcome of one scored workload."""

    semantic_id: NonEmptyString
    workload_uuid: NonEmptyString
    slot_id: NonEmptyString
    role: WorkloadRole
    regime: WorkloadRegime
    operation_family: CorpusOperationFamily
    profiles: tuple[CorpusProfile, ...] = Field(min_length=1)
    status: CellResultStatus
    compiled: bool
    correct: bool
    speedup: float | None = Field(default=None, gt=0.0)


class HardwareGeneralizationCell(CurrentFrozenSchemaModel):
    """Sealed existing-evaluator evidence for one planned cell."""

    current_schema_version = DatasetArtifactSchema.HARDWARE_GENERALIZATION
    current_artifact_kind = GeneralizationArtifactKind.CELL

    schema_version: Literal[DatasetArtifactSchema.HARDWARE_GENERALIZATION]
    artifact_kind: Literal[GeneralizationArtifactKind.CELL]
    plan_digest: str = Field(pattern=SHA256_PATTERN)
    cell_id: NonEmptyString
    observed_gfx_target: NonEmptyString
    observed_hardware_configuration_id: str = Field(pattern=SHA256_PATTERN)
    observed_capacity_class_bytes: NonNegativeInt
    candidates: tuple[CandidateDeclaration, ...]
    results: tuple[CellWorkloadResult, ...]
    evaluator_failures: tuple[NonEmptyString, ...] = ()
    cell_digest: str = Field(pattern=SHA256_PATTERN)


class MetricEstimate(FrozenArtifactModel):
    """One estimate with a Definition-cluster bootstrap interval."""

    value: float | None
    ci_low: float | None
    ci_high: float | None


class StratumMetrics(FrozenArtifactModel):
    """Definition-equal metrics for one report stratum."""

    definition_count: NonNegativeInt
    workload_count: NonNegativeInt
    compilation_rate: MetricEstimate
    correctness_rate: MetricEstimate
    fast_p: dict[str, MetricEstimate]
    conditional_geomean_speedup: MetricEstimate


class WorkloadDrift(FrozenArtifactModel):
    """Measured representativeness drift between two target views."""

    source_target_id: NonEmptyString
    target_target_id: NonEmptyString
    source_definition_count: NonNegativeInt
    target_definition_count: NonNegativeInt
    common_definition_count: NonNegativeInt
    support_jaccard: float = Field(ge=0.0, le=1.0)
    skip_reason_counts: dict[NonEmptyString, NonNegativeInt]
    latent_slot_signature_equal: bool
    categorical_jsd_bits: dict[NonEmptyString, float]
    axis_log2_shifts: dict[NonEmptyString, tuple[float, ...]]
    common_scale_ratios: tuple[float, ...]
    resource_fraction_shifts: dict[NonEmptyString, tuple[float, ...]]


class ComparisonMetrics(FrozenArtifactModel):
    """Paired target-versus-control metric deltas."""

    control_cell_id: NonEmptyString
    target_cell_id: NonEmptyString
    common_support_definitions: NonNegativeInt
    correctness_delta: MetricEstimate
    fast_p_deltas: dict[str, MetricEstimate]
    conditional_speedup_delta: MetricEstimate
    portability_performance_delta: MetricEstimate | None = None


class HardwareGeneralizationReport(CurrentFrozenSchemaModel):
    """Layered benchmark report without a composite score."""

    current_schema_version = DatasetArtifactSchema.HARDWARE_GENERALIZATION
    current_artifact_kind = GeneralizationArtifactKind.REPORT

    schema_version: Literal[DatasetArtifactSchema.HARDWARE_GENERALIZATION]
    artifact_kind: Literal[GeneralizationArtifactKind.REPORT]
    plan_digest: str = Field(pattern=SHA256_PATTERN)
    status: GeneralizationReportStatus
    generalization_conclusion_allowed: bool
    missing_cell_ids: tuple[NonEmptyString, ...]
    target_full: dict[NonEmptyString, StratumMetrics]
    common_support: dict[NonEmptyString, StratumMetrics]
    stratified: dict[NonEmptyString, StratumMetrics]
    comparisons: tuple[ComparisonMetrics, ...]
    workload_drift: tuple[WorkloadDrift, ...]
    report_digest: str = Field(pattern=SHA256_PATTERN)


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "FAST_THRESHOLDS",
    "AgentTrack",
    "CandidateDeclaration",
    "CellResultStatus",
    "CellWorkloadResult",
    "ComparisonMetrics",
    "CorpusAgentView",
    "GeneralizationArtifactKind",
    "GeneralizationReportStatus",
    "HardwareContextView",
    "HardwareGeneralizationCell",
    "HardwareGeneralizationPlan",
    "HardwareGeneralizationReport",
    "HardwareShift",
    "MetricEstimate",
    "NormalizedHardwareFacts",
    "PlannedCell",
    "StratumMetrics",
    "TrainingExposureDeclaration",
    "TrainingHardwareExposure",
    "WorkloadDrift",
]
