# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Wire contracts for hardware-independent problem corpora."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
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
from sol_execbench.core.platform.schema_versions import PlatformArtifactSchema

SHA256_PATTERN = r"^[0-9a-f]{64}$"
REVISION_PATTERN = r"^[0-9a-f]{40}$"


class CorpusProfile(StrEnum):
    """Selectable LLM corpus profiles."""

    CORE = "core"
    MOE = "moe"
    KV_CACHE = "kv_cache"
    LONG_CONTEXT = "long_context"
    QUANTIZED = "quantized"
    ARCHITECTURE_SPECIFIC = "architecture_specific"


class CorpusOperationFamily(StrEnum):
    """Closed operator-family coverage vocabulary."""

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
    """Execution direction supported by LLM Core V1."""

    FORWARD = "forward"


class ShapeTier(StrEnum):
    """Shape provenance tier for a workload."""

    MICRO = "micro"
    EDGE = "edge"
    DECODE = "decode"
    PREFILL = "prefill"
    LONG_CONTEXT = "long_context"


class QuantizationScheme(StrEnum):
    """Quantization semantics independent of tensor storage dtype."""

    FP8_PER_TENSOR = "fp8_per_tensor"
    FP8_PER_TOKEN = "fp8_per_token"
    MXFP8_BLOCK = "mxfp8_block"
    FP4_GROUP = "fp4_group"
    MXFP4_BLOCK = "mxfp4_block"
    INT8_PER_TENSOR = "int8_per_tensor"
    INT8_PER_TOKEN = "int8_per_token"
    INT8_WEIGHT_ONLY = "int8_weight_only"


class StaticCapability(StrEnum):
    """Hardware features used by deterministic corpus selection."""

    DENSE_TENSOR = "dense_tensor"
    GROUPED_GEMM = "grouped_gemm"
    PAGED_MEMORY = "paged_memory"
    SPARSE_ATTENTION = "sparse_attention"
    INDEXED_ATTENTION = "indexed_attention"
    PACKED_LOW_PRECISION = "packed_low_precision"


class CorpusReleaseState(StrEnum):
    """Lifecycle state of a corpus snapshot."""

    CANDIDATE = "candidate"
    FROZEN = "frozen"


class TargetQualificationStatus(StrEnum):
    """Whether a target declaration has real-hardware evidence."""

    DECLARED = "declared"
    HARDWARE_QUALIFIED = "hardware_qualified"


class SelectionReason(StrEnum):
    """Stable workload selection reason codes in evaluation order."""

    INCLUDED = "included"
    PROFILE_NOT_SELECTED = "profile_not_selected"
    UNSUPPORTED_DTYPE = "unsupported_dtype"
    UNSUPPORTED_QUANTIZATION = "unsupported_quantization"
    MISSING_CAPABILITY = "missing_capability"
    TENSOR_LIMIT_EXCEEDED = "tensor_limit_exceeded"
    REFERENCE_IPC_LIMIT_EXCEEDED = "reference_ipc_limit_exceeded"
    MEMORY_BUDGET_EXCEEDED = "memory_budget_exceeded"


class ModelSource(FrozenArtifactModel):
    """One clean-room model-architecture source pinned to a revision."""

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


class ResourceEnvelope(FrozenArtifactModel):
    """Deterministic upper bounds used without executing a workload."""

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
            raise ValueError(
                "reference_peak_bytes is smaller than its components"
            )
        if self.max_tensor_bytes > max(self.input_bytes, self.output_bytes):
            raise ValueError("max_tensor_bytes exceeds total tensor storage")
        return self


class StaticRequirements(FrozenArtifactModel):
    """Static target requirements for one concrete workload."""

    dtypes: tuple[DType, ...] = Field(min_length=1)
    quantization: tuple[QuantizationScheme, ...] = ()
    capabilities: tuple[StaticCapability, ...] = ()
    resources: ResourceEnvelope


class CorpusWorkloadRecord(FrozenArtifactModel):
    """Manifest metadata for one Workload row."""

    uuid: NonEmptyString
    shape_tier: ShapeTier
    source_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    requirements: StaticRequirements


class CorpusEntry(FrozenArtifactModel):
    """One deduplicated Definition and its concrete workloads."""

    semantic_id: NonEmptyString
    semantic_fingerprint: str = Field(pattern=SHA256_PATTERN)
    problem_name: NonEmptyString
    operation_family: CorpusOperationFamily
    profiles: tuple[CorpusProfile, ...] = Field(min_length=1)
    pass_kind: Literal[CorpusPassKind.FORWARD] = CorpusPassKind.FORWARD
    source_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    definition_path: NonEmptyString
    workload_path: NonEmptyString
    definition_sha256: str = Field(pattern=SHA256_PATTERN)
    workload_sha256: str = Field(pattern=SHA256_PATTERN)
    workloads: tuple[CorpusWorkloadRecord, ...] = Field(min_length=1)


class CorpusCoveragePolicy(FrozenArtifactModel):
    """Hard release and optional strict-selection coverage floors."""

    minimum_definitions: int = Field(ge=1)
    minimum_workloads: int = Field(ge=1)
    operation_minimum_definitions: dict[CorpusOperationFamily, int]
    profile_minimum_workloads: dict[CorpusProfile, int]


class CorpusManifest(CurrentFrozenSchemaModel):
    """Generic, immutable problem-corpus manifest."""

    current_schema_version = DatasetArtifactSchema.CORPUS_MANIFEST

    schema_version: Literal[DatasetArtifactSchema.CORPUS_MANIFEST]
    corpus_id: NonEmptyString
    release_id: NonEmptyString
    release_state: CorpusReleaseState
    license: NonEmptyString
    source_freeze_date: date
    profiles: tuple[CorpusProfile, ...] = Field(min_length=1)
    sources: tuple[ModelSource, ...] = Field(min_length=1)
    coverage_policy: CorpusCoveragePolicy
    entries: tuple[CorpusEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_registry(self) -> CorpusManifest:
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("model source IDs must be unique")
        known_sources = set(source_ids)
        semantic_ids = [entry.semantic_id for entry in self.entries]
        fingerprints = [entry.semantic_fingerprint for entry in self.entries]
        if len(semantic_ids) != len(set(semantic_ids)):
            raise ValueError("semantic IDs must be unique")
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("semantic fingerprints must be unique")
        workload_ids = [
            workload.uuid
            for entry in self.entries
            for workload in entry.workloads
        ]
        if len(workload_ids) != len(set(workload_ids)):
            raise ValueError("workload UUIDs must be globally unique")
        for entry in self.entries:
            if not set(entry.source_ids) <= known_sources:
                raise ValueError(f"unknown source in {entry.semantic_id}")
            for workload in entry.workloads:
                if not set(workload.source_ids) <= set(entry.source_ids):
                    raise ValueError(
                        f"workload source escapes {entry.semantic_id}",
                    )
        return self


class StaticTargetDescriptor(CurrentFrozenSchemaModel):
    """User-supplied, non-probed hardware capability declaration."""

    current_schema_version = PlatformArtifactSchema.STATIC_TARGET_DESCRIPTOR

    schema_version: Literal[PlatformArtifactSchema.STATIC_TARGET_DESCRIPTOR]
    target_id: NonEmptyString
    gfx_target: NonEmptyString
    qualification_status: TargetQualificationStatus
    declaration_source: NonEmptyString
    memory_budget_bytes: int | None = Field(default=None, gt=0)
    max_tensor_bytes: int = Field(gt=0)
    reference_ipc_limit_bytes: int = Field(gt=0)
    supported_dtypes: tuple[DType, ...] = Field(min_length=1)
    supported_quantization: tuple[QuantizationScheme, ...] = ()
    capabilities: tuple[StaticCapability, ...] = ()


class SelectionDecision(FrozenArtifactModel):
    """One deterministic workload inclusion decision."""

    semantic_id: NonEmptyString
    workload_uuid: NonEmptyString
    included: bool
    reason: SelectionReason
    detail: NonEmptyString


class SelectedProblem(FrozenArtifactModel):
    """One problem emitted into a static selection."""

    semantic_id: NonEmptyString
    problem_name: NonEmptyString
    definition_path: NonEmptyString
    workload_path: NonEmptyString
    workload_uuids: tuple[NonEmptyString, ...] = Field(min_length=1)
    definition_sha256: str = Field(pattern=SHA256_PATTERN)
    workload_sha256: str = Field(pattern=SHA256_PATTERN)


class CorpusSelectionManifest(CurrentFrozenSchemaModel):
    """Auditable result of static, hardware-free corpus selection."""

    current_schema_version = DatasetArtifactSchema.CORPUS_SELECTION_MANIFEST

    schema_version: Literal[DatasetArtifactSchema.CORPUS_SELECTION_MANIFEST]
    corpus_id: NonEmptyString
    release_id: NonEmptyString
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    target_descriptor_sha256: str = Field(pattern=SHA256_PATTERN)
    target: StaticTargetDescriptor
    requested_profiles: tuple[CorpusProfile, ...] = Field(min_length=1)
    require_complete_profile: bool
    coverage_complete: bool
    coverage: dict[str, int]
    problems: tuple[SelectedProblem, ...]
    decisions: tuple[SelectionDecision, ...]


__all__ = [
    "CorpusCoveragePolicy",
    "CorpusEntry",
    "CorpusManifest",
    "CorpusOperationFamily",
    "CorpusPassKind",
    "CorpusProfile",
    "CorpusReleaseState",
    "CorpusSelectionManifest",
    "CorpusWorkloadRecord",
    "ModelSource",
    "QuantizationScheme",
    "ResourceEnvelope",
    "SelectedProblem",
    "SelectionDecision",
    "SelectionReason",
    "ShapeTier",
    "StaticCapability",
    "StaticRequirements",
    "StaticTargetDescriptor",
    "TargetQualificationStatus",
]
