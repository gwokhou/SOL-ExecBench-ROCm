# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed qualification receipts preceding diagnostic GPU collection."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.batch_gpu_qualification import (
    BatchGPUQualificationStage,
    validate_qualification_coverage,
    validate_qualification_parent,
    validate_unique_qualification_ids,
)
from sol_execbench.core.bench.performance_model.diagnostic_schema_versions import (
    DiagnosticArtifactSchema,
    DiagnosticQualificationArtifactKind,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    NonEmptyString,
)
from sol_execbench.core.integrity import SHA256Digest

_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DiagnosticQualificationReceipt(CurrentSchemaModel):
    """Verified evaluator result for one family within one GPU gate."""

    model_config = _CONFIG
    current_schema_version = (
        DiagnosticArtifactSchema.DIAGNOSTIC_CORPUS_QUALIFICATION
    )
    current_artifact_kind = DiagnosticQualificationArtifactKind.RECEIPT

    schema_version: Literal[
        DiagnosticArtifactSchema.DIAGNOSTIC_CORPUS_QUALIFICATION
    ] = DiagnosticArtifactSchema.DIAGNOSTIC_CORPUS_QUALIFICATION
    artifact_kind: Literal[DiagnosticQualificationArtifactKind.RECEIPT] = (
        DiagnosticQualificationArtifactKind.RECEIPT
    )
    stage: Literal[
        BatchGPUQualificationStage.CANARY,
        BatchGPUQualificationStage.FULL,
    ]
    role: Literal["development", "held_out"]
    family: WorkloadKind
    case_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    workload_uuids: tuple[NonEmptyString, ...] = Field(min_length=1)
    definition_sha256: SHA256Digest
    solution_sha256: SHA256Digest
    workload_sha256: SHA256Digest
    config_sha256: SHA256Digest
    trace_sha256: SHA256Digest
    log_sha256: SHA256Digest
    trace_count: int = Field(gt=0)
    all_passed: Literal[True] = True
    performance_authority: Literal[False] = False
    created_at: NonEmptyString

    @model_validator(mode="after")
    def identities_are_one_to_one(self) -> DiagnosticQualificationReceipt:
        """Reject missing, repeated, or ambiguous workload identities."""
        if len(self.case_ids) != len(self.workload_uuids):
            raise ValueError("qualification case/workload counts differ")
        if self.trace_count != len(self.case_ids):
            raise ValueError("qualification trace count differs from cases")
        validate_unique_qualification_ids(
            self.case_ids,
            owner="qualification receipt",
            item_name="case",
        )
        validate_unique_qualification_ids(
            self.workload_uuids,
            owner="qualification receipt",
            item_name="workload",
        )
        return self


class DiagnosticCorpusQualification(CurrentSchemaModel):
    """Content-bound completion gate for one qualification stage."""

    model_config = _CONFIG
    current_schema_version = (
        DiagnosticArtifactSchema.DIAGNOSTIC_CORPUS_QUALIFICATION
    )
    current_artifact_kind = DiagnosticQualificationArtifactKind.GATE

    schema_version: Literal[
        DiagnosticArtifactSchema.DIAGNOSTIC_CORPUS_QUALIFICATION
    ] = DiagnosticArtifactSchema.DIAGNOSTIC_CORPUS_QUALIFICATION
    artifact_kind: Literal[DiagnosticQualificationArtifactKind.GATE] = (
        DiagnosticQualificationArtifactKind.GATE
    )
    stage: BatchGPUQualificationStage
    role: Literal["all", "development", "held_out"]
    purpose: Literal["correctness_qualification"] = "correctness_qualification"
    performance_authority: Literal[False] = False
    design_sha256: SHA256Digest
    contract_sha256: SHA256Digest
    collector_sha256: SHA256Digest
    config_sha256: SHA256Digest
    preflight_sha256: SHA256Digest
    source_revision: NonEmptyString
    parent_gate_sha256: SHA256Digest | None = None
    case_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    receipts: tuple[DiagnosticQualificationReceipt, ...] = ()
    created_at: NonEmptyString

    @model_validator(mode="after")
    def stage_contract_is_consistent(self) -> DiagnosticCorpusQualification:
        """Require the exact parent and receipt shape for each gate."""
        validate_unique_qualification_ids(
            self.case_ids,
            owner="qualification gate",
            item_name="case",
        )
        if self.stage is BatchGPUQualificationStage.STATIC:
            if self.role != "all":
                raise ValueError(
                    "static qualification must be unparented/all-role"
                )
            validate_qualification_parent(self.stage, self.parent_gate_sha256)
            if self.receipts:
                raise ValueError(
                    "static qualification cannot contain GPU receipts"
                )
            return self
        if self.role == "all":
            raise ValueError("GPU qualification requires a corpus role")
        validate_qualification_parent(self.stage, self.parent_gate_sha256)
        validate_qualification_coverage(
            self.case_ids,
            tuple(receipt.case_ids for receipt in self.receipts),
            item_name="case",
        )
        if any(
            receipt.stage is not self.stage or receipt.role != self.role
            for receipt in self.receipts
        ):
            raise ValueError("qualification receipt stage/role mismatch")
        return self


__all__ = [
    "BatchGPUQualificationStage",
    "DiagnosticCorpusQualification",
    "DiagnosticQualificationReceipt",
]
