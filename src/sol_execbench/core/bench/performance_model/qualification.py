# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed qualification receipts preceding diagnostic GPU collection."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.batch_gpu_qualification import (
    BatchGPUQualificationStage,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    NonEmptyString,
)
from sol_execbench.core.integrity import SHA256Digest
from sol_execbench.core.integrity.schema_versions import SchemaVersion

_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DiagnosticQualificationReceipt(CurrentSchemaModel):
    """Verified evaluator result for one family within one GPU gate."""

    model_config = _CONFIG
    current_schema_version = (
        SchemaVersion.DIAGNOSTIC_CORPUS_QUALIFICATION_RECEIPT
    )

    schema_version: Literal[
        SchemaVersion.DIAGNOSTIC_CORPUS_QUALIFICATION_RECEIPT
    ] = SchemaVersion.DIAGNOSTIC_CORPUS_QUALIFICATION_RECEIPT
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
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("qualification receipt repeats case IDs")
        if len(set(self.workload_uuids)) != len(self.workload_uuids):
            raise ValueError("qualification receipt repeats workload UUIDs")
        return self


class DiagnosticCorpusQualification(CurrentSchemaModel):
    """Content-bound completion gate for one qualification stage."""

    model_config = _CONFIG
    current_schema_version = SchemaVersion.DIAGNOSTIC_CORPUS_QUALIFICATION

    schema_version: Literal[SchemaVersion.DIAGNOSTIC_CORPUS_QUALIFICATION] = (
        SchemaVersion.DIAGNOSTIC_CORPUS_QUALIFICATION
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
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("qualification gate repeats case IDs")
        if self.stage is BatchGPUQualificationStage.STATIC:
            if self.role != "all" or self.parent_gate_sha256 is not None:
                raise ValueError(
                    "static qualification must be unparented/all-role"
                )
            if self.receipts:
                raise ValueError(
                    "static qualification cannot contain GPU receipts"
                )
            return self
        if self.role == "all" or self.parent_gate_sha256 is None:
            raise ValueError("GPU qualification requires role and parent gate")
        receipt_cases = tuple(
            case_id for receipt in self.receipts for case_id in receipt.case_ids
        )
        if not self.receipts or set(receipt_cases) != set(self.case_ids):
            raise ValueError("qualification receipts do not cover gate cases")
        if len(receipt_cases) != len(set(receipt_cases)):
            raise ValueError("qualification receipts overlap case IDs")
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
