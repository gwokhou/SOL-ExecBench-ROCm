# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for rule-defined, hardware-generated problem corpora."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from math import gcd
from typing import Literal

from pydantic import Field, model_validator

from sol_execbench.core.data.base_model import (
    CurrentFrozenSchemaModel,
    FrozenArtifactModel,
    NonEmptyString,
    NonNegativeInt,
)
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.dataset.schema_versions import DatasetArtifactSchema
from sol_execbench.core.platform.memory_quota import GPUMemoryQuotaEvidence
from sol_execbench.core.platform.schema_versions import PlatformArtifactSchema

SHA256_PATTERN = r"^[0-9a-f]{64}$"
REVISION_PATTERN = r"^[0-9a-f]{40}$"
WORKLOAD_GENERATOR_VERSION = "llm_core_common_scale.v1"
WORKLOAD_GENERATION_PROTOCOL_MAJOR = 1
MINIMUM_CAPACITY_CLASS_GIB = 1
SATURATED_CAPACITY_CLASS_GIB = 384


class CorpusProfile(StrEnum):
    """Selectable corpus profiles."""

    CORE = "core"
    MOE = "moe"
    KV_CACHE = "kv_cache"
    LONG_CONTEXT = "long_context"
    QUANTIZED = "quantized"
    ARCHITECTURE_SPECIFIC = "architecture_specific"


class CorpusOperationFamily(StrEnum):
    """Closed operator-family vocabulary."""

    LINEAR = "linear"
    NORM_ACTIVATION = "norm_activation"
    POSITION = "position"
    ATTENTION = "attention"
    ADVANCED_ATTENTION = "advanced_attention"
    KV_CACHE = "kv_cache"
    MOE = "moe"
    QUANTIZATION = "quantization"
    INDEXING_REDUCTION = "indexing_reduction"


class CorpusPassKind(StrEnum):
    """Supported execution directions."""

    FORWARD = "forward"


class WorkloadRole(StrEnum):
    """Lifecycle roles for generated workloads."""

    SMOKE = "smoke"
    DEVELOPMENT = "development"
    HOLDOUT = "holdout"
    ROBUSTNESS = "robustness"


class WorkloadRegime(StrEnum):
    """Balanced performance regimes."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    IRREGULAR = "irregular"
    CAPACITY = "capacity"


class ServingPhase(StrEnum):
    """Serving phases represented by workload slots."""

    DECODE = "decode"
    PREFILL = "prefill"
    MIXED = "mixed"
    NOT_APPLICABLE = "not_applicable"


class ShapeBinding(StrEnum):
    """Sources for fixed, non-scalable axes."""

    MODEL = "model"
    BOUNDARY = "boundary"


class QuantizationScheme(StrEnum):
    """Supported quantization semantics."""

    FP8_PER_TENSOR = "fp8_per_tensor"
    FP8_PER_TOKEN = "fp8_per_token"
    MXFP8_BLOCK = "mxfp8_block"
    FP4_GROUP = "fp4_group"
    MXFP4_BLOCK = "mxfp4_block"
    INT8_PER_TENSOR = "int8_per_tensor"
    INT8_PER_TOKEN = "int8_per_token"
    INT8_WEIGHT_ONLY = "int8_weight_only"


class StaticCapability(StrEnum):
    """Target capabilities required by generation rules."""

    DENSE_TENSOR = "dense_tensor"
    GROUPED_GEMM = "grouped_gemm"
    PAGED_MEMORY = "paged_memory"
    SPARSE_ATTENTION = "sparse_attention"
    INDEXED_ATTENTION = "indexed_attention"
    PACKED_LOW_PRECISION = "packed_low_precision"


class CorpusReleaseState(StrEnum):
    """Corpus release lifecycle states."""

    CANDIDATE = "candidate"
    FROZEN = "frozen"


class TargetQualificationStatus(StrEnum):
    """Target declaration qualification states."""

    DECLARED = "declared"
    HARDWARE_QUALIFIED = "hardware_qualified"


class GenerationDecisionStatus(StrEnum):
    """Definition-level generation outcomes."""

    GENERATED = "generated"
    PROFILE_NOT_SELECTED = "profile_not_selected"
    TARGET_NOT_ELIGIBLE = "target_not_eligible"
    UNSUPPORTED_DTYPE = "unsupported_dtype"
    UNSUPPORTED_QUANTIZATION = "unsupported_quantization"
    MISSING_CAPABILITY = "missing_capability"
    INSUFFICIENT_CAPACITY = "insufficient_capacity"


class TargetCoverageStatus(StrEnum):
    """Generated-view coverage outcomes."""

    COMPLETE = "complete"
    INSUFFICIENT_CAPACITY_COVERAGE = "insufficient_capacity_coverage"


class ModelSourceBase(FrozenArtifactModel):
    source_id: NonEmptyString
    model_id: NonEmptyString
    url: NonEmptyString
    revision: str = Field(pattern=REVISION_PATTERN)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    license: NonEmptyString
    license_reviewed: Literal[True]
    clean_room: Literal[True]
    captured_at: date
    architecture_role: NonEmptyString


class ModelArchitectureFacts(FrozenArtifactModel):
    """Normalized model facts bound by generation rules."""

    hidden_size: int | None = Field(default=None, gt=0)
    intermediate_size: int | None = Field(default=None, gt=0)
    query_heads: int | None = Field(default=None, gt=0)
    kv_heads: int | None = Field(default=None, gt=0)
    head_dimension: int | None = Field(default=None, gt=0)
    expert_count: int | None = Field(default=None, gt=0)
    experts_per_token: int | None = Field(default=None, gt=0)
    page_size: int | None = Field(default=None, gt=0)
    maximum_context: int | None = Field(default=None, gt=0)
    tensor_parallel: int | None = Field(default=None, gt=0)
    expert_parallel: int | None = Field(default=None, gt=0)
    quantization_block_size: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_facts(self) -> ModelArchitectureFacts:
        if not any(value is not None for value in self.model_dump().values()):
            raise ValueError("model architecture facts cannot be empty")
        if (
            self.query_heads
            and self.kv_heads
            and self.kv_heads > self.query_heads
        ):
            raise ValueError("KV head count exceeds query head count")
        if (
            self.expert_count
            and self.experts_per_token
            and self.experts_per_token > self.expert_count
        ):
            raise ValueError("experts per token exceeds expert count")
        return self


class ModelSource(ModelSourceBase):
    """Pinned public model source and normalized facts."""

    architecture_facts: ModelArchitectureFacts


class ResourceEnvelope(FrozenArtifactModel):
    """Static resource bounds for one generated workload."""

    input_bytes: NonNegativeInt
    output_bytes: NonNegativeInt
    max_tensor_bytes: NonNegativeInt
    reference_ipc_bytes: NonNegativeInt
    temporary_bytes: NonNegativeInt
    reference_peak_bytes: NonNegativeInt

    @model_validator(mode="after")
    def _validate_peak(self) -> ResourceEnvelope:
        floor = self.input_bytes + self.output_bytes + self.temporary_bytes
        if self.reference_peak_bytes < floor:
            raise ValueError("reference peak is smaller than its components")
        if self.max_tensor_bytes > max(self.input_bytes, self.output_bytes):
            raise ValueError("max tensor exceeds total tensor storage")
        return self


class StaticRequirements(FrozenArtifactModel):
    """Target requirements for one generated workload."""

    dtypes: tuple[DType, ...] = Field(min_length=1)
    quantization: tuple[QuantizationScheme, ...] = ()
    capabilities: tuple[StaticCapability, ...] = ()
    resources: ResourceEnvelope


class GenerationSlotRule(FrozenArtifactModel):
    """One fixed point in a cross-hardware distribution."""

    slot_id: NonEmptyString
    role: WorkloadRole
    regime: WorkloadRegime
    serving_phase: ServingPhase
    binding: ShapeBinding
    scale_numerator: int = Field(gt=0)
    scale_denominator: int = Field(gt=0)
    irregular: bool = False


class WorkloadGenerationRule(CurrentFrozenSchemaModel):
    """Typed rule interpreted by the common-scale generator."""

    current_schema_version = DatasetArtifactSchema.WORKLOAD_GENERATION_RULE

    schema_version: Literal[DatasetArtifactSchema.WORKLOAD_GENERATION_RULE]
    semantic_id: NonEmptyString
    algorithm_version: Literal["llm_core_common_scale.v1"]
    operation_family: CorpusOperationFamily
    variant: NonEmptyString
    source_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    eligible_gfx_targets: tuple[NonEmptyString, ...] = Field(min_length=1)
    quantization: QuantizationScheme | None = None
    capabilities: tuple[StaticCapability, ...] = ()
    slots: tuple[GenerationSlotRule, ...] = Field(min_length=9, max_length=9)

    @model_validator(mode="after")
    def _validate_distribution(self) -> WorkloadGenerationRule:
        ids = [slot.slot_id for slot in self.slots]
        if len(ids) != len(set(ids)):
            raise ValueError("generation slot IDs must be unique")
        ratios = []
        for slot in self.slots:
            divisor = gcd(slot.scale_numerator, slot.scale_denominator)
            ratios.append(
                (
                    slot.scale_numerator // divisor,
                    slot.scale_denominator // divisor,
                )
            )
        if len(ratios) != len(set(ratios)):
            raise ValueError("generation slot coefficients must be unique")
        if sum(slot.role is WorkloadRole.SMOKE for slot in self.slots) != 1:
            raise ValueError("generation rule must contain one smoke slot")
        development = tuple(
            slot for slot in self.slots if slot.role is WorkloadRole.DEVELOPMENT
        )
        if len(development) != 8:
            raise ValueError(
                "generation rule must contain eight development slots"
            )
        for regime in WorkloadRegime:
            if sum(slot.regime is regime for slot in development) != 2:
                raise ValueError(f"generation rule must balance {regime}")
        return self


class CorpusEntry(FrozenArtifactModel):
    """One Definition and its frozen generation rule."""

    semantic_id: NonEmptyString
    semantic_fingerprint: str = Field(pattern=SHA256_PATTERN)
    problem_name: NonEmptyString
    operation_family: CorpusOperationFamily
    profiles: tuple[CorpusProfile, ...] = Field(min_length=1)
    pass_kind: Literal[CorpusPassKind.FORWARD] = CorpusPassKind.FORWARD
    source_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    definition_path: NonEmptyString
    generation_rule_path: NonEmptyString
    definition_sha256: str = Field(pattern=SHA256_PATTERN)
    generation_rule_sha256: str = Field(pattern=SHA256_PATTERN)


class CorpusCoveragePolicy(FrozenArtifactModel):
    """Definition inventory and generated coverage floors."""

    definition_count: int = Field(ge=1)
    operation_definition_counts: dict[CorpusOperationFamily, int]
    workloads_per_generated_definition: Literal[9] = 9
    profile_minimum_generated_definitions: dict[CorpusProfile, int]


class CorpusGenerationPolicy(FrozenArtifactModel):
    """Frozen capacity normalization and executor policy."""

    algorithm_version: Literal["llm_core_common_scale.v1"]
    capacity_classes_gib: tuple[int, ...] = Field(min_length=1)
    maximum_capacity_numerator: Literal[1] = 1
    maximum_capacity_denominator: Literal[2] = 2

    @model_validator(mode="after")
    def _validate_classes(self) -> CorpusGenerationPolicy:
        expected = tuple(sorted(set(self.capacity_classes_gib)))
        if expected != self.capacity_classes_gib:
            raise ValueError("capacity classes must be sorted and unique")
        if expected[0] != MINIMUM_CAPACITY_CLASS_GIB:
            raise ValueError("capacity classes must start at 1 GiB")
        if expected[-1] != SATURATED_CAPACITY_CLASS_GIB:
            raise ValueError("capacity classes must saturate at 384 GiB")
        return self


class CorpusManifest(CurrentFrozenSchemaModel):
    """Frozen Definition and rule corpus without concrete workloads."""

    current_schema_version = DatasetArtifactSchema.CORPUS_MANIFEST

    schema_version: Literal[DatasetArtifactSchema.CORPUS_MANIFEST]
    corpus_id: NonEmptyString
    release_id: NonEmptyString
    release_state: CorpusReleaseState
    license: NonEmptyString
    source_freeze_date: date
    profiles: tuple[CorpusProfile, ...] = Field(min_length=1)
    sources: tuple[ModelSource, ...] = Field(min_length=1)
    generation_policy: CorpusGenerationPolicy
    coverage_policy: CorpusCoveragePolicy
    entries: tuple[CorpusEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_inventory(self) -> CorpusManifest:
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("model source IDs must be unique")
        if len(self.entries) != self.coverage_policy.definition_count:
            raise ValueError("definition inventory differs from policy")
        semantic_ids = [entry.semantic_id for entry in self.entries]
        fingerprints = [entry.semantic_fingerprint for entry in self.entries]
        if len(semantic_ids) != len(set(semantic_ids)):
            raise ValueError("semantic IDs must be unique")
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("semantic fingerprints must be unique")
        known = set(source_ids)
        if any(not set(entry.source_ids) <= known for entry in self.entries):
            raise ValueError("corpus entry references an unknown source")
        observed = {
            family: sum(
                entry.operation_family is family for entry in self.entries
            )
            for family in CorpusOperationFamily
        }
        if observed != self.coverage_policy.operation_definition_counts:
            raise ValueError("operation inventory differs from policy")
        return self


class StaticTargetDescriptor(CurrentFrozenSchemaModel):
    """Declared target capabilities and execution limits."""

    current_schema_version = PlatformArtifactSchema.STATIC_TARGET_DESCRIPTOR

    schema_version: Literal[PlatformArtifactSchema.STATIC_TARGET_DESCRIPTOR]
    target_id: NonEmptyString
    gfx_target: NonEmptyString
    qualification_status: TargetQualificationStatus
    declaration_source: NonEmptyString
    max_tensor_bytes: int = Field(gt=0)
    reference_ipc_limit_bytes: int = Field(gt=0)
    supported_dtypes: tuple[DType, ...] = Field(min_length=1)
    supported_quantization: tuple[QuantizationScheme, ...] = ()
    capabilities: tuple[StaticCapability, ...] = ()


class GeneratedWorkloadRecord(FrozenArtifactModel):
    """Auditable metadata for one generated workload."""

    semantic_id: NonEmptyString
    slot_id: NonEmptyString
    uuid: NonEmptyString
    axes: dict[NonEmptyString, NonNegativeInt]
    role: WorkloadRole
    regime: WorkloadRegime
    serving_phase: ServingPhase
    binding: ShapeBinding
    scale_numerator: int = Field(gt=0)
    scale_denominator: int = Field(gt=0)
    common_scale: int = Field(gt=0)
    input_profile_id: NonEmptyString
    input_profile_digest: str = Field(pattern=SHA256_PATTERN)
    correctness_profile_id: NonEmptyString
    correctness_profile_digest: str = Field(pattern=SHA256_PATTERN)
    requirements: StaticRequirements


class GenerationDecision(FrozenArtifactModel):
    """Result of applying one rule to one generation cohort."""

    semantic_id: NonEmptyString
    status: GenerationDecisionStatus
    detail: NonEmptyString
    distribution_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    common_scale: int | None = Field(default=None, gt=0)
    workload_uuids: tuple[NonEmptyString, ...] = ()


class GeneratedProblem(FrozenArtifactModel):
    """One concrete generated problem artifact pair."""

    semantic_id: NonEmptyString
    problem_name: NonEmptyString
    definition_path: NonEmptyString
    workload_path: NonEmptyString
    workload_uuids: tuple[NonEmptyString, ...] = Field(
        min_length=9, max_length=9
    )
    definition_sha256: str = Field(pattern=SHA256_PATTERN)
    workload_sha256: str = Field(pattern=SHA256_PATTERN)


class CorpusTargetViewManifest(CurrentFrozenSchemaModel):
    """Auditable hardware-specific view generated from frozen rules."""

    current_schema_version = DatasetArtifactSchema.CORPUS_TARGET_VIEW

    schema_version: Literal[DatasetArtifactSchema.CORPUS_TARGET_VIEW]
    corpus_id: NonEmptyString
    release_id: NonEmptyString
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    target_descriptor_sha256: str = Field(pattern=SHA256_PATTERN)
    target: StaticTargetDescriptor
    capacity_evidence: GPUMemoryQuotaEvidence
    capacity_class_id: NonEmptyString | None
    capacity_class_bytes: NonNegativeInt
    distribution_id: str = Field(pattern=SHA256_PATTERN)
    generation_cohort_id: str = Field(pattern=SHA256_PATTERN)
    generator_version: Literal["llm_core_common_scale.v1"]
    workload_view_digest: str = Field(pattern=SHA256_PATTERN)
    requested_profiles: tuple[CorpusProfile, ...] = Field(min_length=1)
    require_complete_profile: bool
    coverage_status: TargetCoverageStatus
    coverage: dict[str, int]
    problems: tuple[GeneratedProblem, ...]
    workloads: tuple[GeneratedWorkloadRecord, ...]
    decisions: tuple[GenerationDecision, ...]

    @model_validator(mode="after")
    def _validate_target_identity(self) -> CorpusTargetViewManifest:
        if self.target.gfx_target != self.capacity_evidence.gfx_target:
            raise ValueError("capacity evidence does not match target gfx")
        generated_ids = {
            uuid for problem in self.problems for uuid in problem.workload_uuids
        }
        if generated_ids != {workload.uuid for workload in self.workloads}:
            raise ValueError(
                "workload inventory does not match generated problems"
            )
        return self


__all__ = [
    "MINIMUM_CAPACITY_CLASS_GIB",
    "SATURATED_CAPACITY_CLASS_GIB",
    "WORKLOAD_GENERATION_PROTOCOL_MAJOR",
    "WORKLOAD_GENERATOR_VERSION",
    "CorpusCoveragePolicy",
    "CorpusEntry",
    "CorpusGenerationPolicy",
    "CorpusManifest",
    "CorpusOperationFamily",
    "CorpusPassKind",
    "CorpusProfile",
    "CorpusReleaseState",
    "CorpusTargetViewManifest",
    "GeneratedProblem",
    "GeneratedWorkloadRecord",
    "GenerationDecision",
    "GenerationDecisionStatus",
    "GenerationSlotRule",
    "ModelArchitectureFacts",
    "ModelSource",
    "QuantizationScheme",
    "ResourceEnvelope",
    "ServingPhase",
    "ShapeBinding",
    "StaticCapability",
    "StaticRequirements",
    "StaticTargetDescriptor",
    "TargetCoverageStatus",
    "TargetQualificationStatus",
    "WorkloadGenerationRule",
    "WorkloadRegime",
    "WorkloadRole",
]
