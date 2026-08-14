# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict wire schemas for AKA corpus and materialization manifests."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    StrictArtifactModel,
)
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.dataset.aka_compatibility import (
    AKACompatibilityStage,
    AKATargetGeneration,
)
from sol_execbench.core.dataset.aka_contract import (
    AKAArtifactRole,
    AKACapability,
    AKACorpusRole,
    AKAFusionDepth,
    AKAOfficialScoringStatus,
    AKAOperation,
    AKAPassKind,
    AKAReleasePolicy,
    AKASourceFamily,
    AKASuite,
)
from sol_execbench.core.dataset.schema_versions import (
    AKA_CORPUS_MANIFEST_SCHEMA_VERSION,
    AKA_MATERIALIZATION_MANIFEST_SCHEMA_VERSION,
)

_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class _AKAModel(StrictArtifactModel):
    model_config = _CONFIG


class AKASourceSchema(_AKAModel):
    repository: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: str
    provenance_class: str
    aka_commit_sha256: str = Field(pattern=r"^[0-9a-f]{40}$")


class AKAMaterializationSourceSchema(_AKAModel):
    repository: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license: str
    provenance_class: str


class AKAExecutionTargetSchema(_AKAModel):
    generation: AKATargetGeneration
    supported_tensor_dtypes: list[DType] = Field(min_length=1)


class AKAFormalAnalysisSchema(_AKAModel):
    architecture_profile: str
    formal_gfx_target: str
    architecture_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AKAArtifactBindingSchema(_AKAModel):
    role: AKAArtifactRole
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AKACorpusEntrySchema(_AKAModel):
    slot: str
    task_path: str
    problem_name: str
    operation: AKAOperation
    input_dtypes: list[DType] = Field(min_length=1)
    output_dtypes: list[DType] = Field(min_length=1)
    capabilities: list[AKACapability]
    pass_kind: AKAPassKind
    fusion_depth: AKAFusionDepth
    source_family: AKASourceFamily
    suite: AKASuite
    role: AKACorpusRole = AKACorpusRole.SCORED
    exclusion_reason_code: str = ""
    workload_uuids: list[str] = Field(min_length=1)
    aka_artifacts: list[AKAArtifactBindingSchema]
    golden: dict[str, Any]


class AKAMaterializedProblemSchema(_AKAModel):
    path: str
    task_path: str
    definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    workload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AKAToleranceBindingSchema(_AKAModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AKAOfficialScoringSchema(_AKAModel):
    status: AKAOfficialScoringStatus
    baseline_id: str
    reason_code: str | None = None
    release_policy: AKAReleasePolicy | None = None
    required_evidence: list[str] | None = None


class AKACoverageSchema(_AKAModel):
    axes: dict[str, dict[str, int]]
    combinations: list[dict[str, str | int]]


class AKACorpusManifestSchema(CurrentSchemaModel):
    """Current repository AKA corpus manifest."""

    model_config = _CONFIG
    current_schema_version = AKA_CORPUS_MANIFEST_SCHEMA_VERSION

    schema_version: int = AKA_CORPUS_MANIFEST_SCHEMA_VERSION
    source: AKASourceSchema
    execution_targets: dict[str, AKAExecutionTargetSchema]
    formal_analysis: AKAFormalAnalysisSchema
    tolerance_calibration: AKAToleranceBindingSchema
    official_scoring: AKAOfficialScoringSchema
    formal_coverage_requirements: AKACoverageSchema
    materialized_problems: list[AKAMaterializedProblemSchema]
    entries: list[AKACorpusEntrySchema]


class AKACacheClearSchema(_AKAModel):
    detected_l2_bytes: int | None
    clear_buffer_bytes: int = Field(gt=0)
    source: str
    fallback_reason: str | None


class AKAMaterializationTargetSchema(_AKAModel):
    device: str
    device_index: int
    device_name: str
    gfx_target: str
    total_memory_bytes: int = Field(gt=0)
    l2_cache_bytes: int | None
    torch_version: str
    hip_version: str
    cache_clear: AKACacheClearSchema


class AKASelectionPolicySchema(_AKAModel):
    static_filter: str
    live_probe: str
    probe_timeout_seconds: float = Field(gt=0)
    unknown_targets: str


class AKAMaterializedSelectionSchema(_AKAModel):
    path: str
    task_path: str
    definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_workload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    workload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    workload_uuids: list[str] = Field(min_length=1)


class AKAWorkloadDecisionSchema(_AKAModel):
    path: str
    workload_uuid: str
    included: bool
    stage: AKACompatibilityStage
    reason_code: str
    detail: str
    metrics: dict[str, int]


class AKAMaterializationManifestSchema(CurrentSchemaModel):
    """Current target-specific AKA materialization manifest."""

    model_config = _CONFIG
    current_schema_version = AKA_MATERIALIZATION_MANIFEST_SCHEMA_VERSION

    schema_version: int = AKA_MATERIALIZATION_MANIFEST_SCHEMA_VERSION
    source: AKAMaterializationSourceSchema
    aka_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target: AKAMaterializationTargetSchema
    selection_policy: AKASelectionPolicySchema
    problems: list[AKAMaterializedSelectionSchema]
    workload_decisions: list[AKAWorkloadDecisionSchema]
    coverage: dict[str, Any]


__all__ = ["AKACorpusManifestSchema", "AKAMaterializationManifestSchema"]
