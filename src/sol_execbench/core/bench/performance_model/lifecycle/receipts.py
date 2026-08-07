# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed stage and case receipts for lifecycle completion.

A stage is complete only when a verifier re-checks the typed receipt, every
input identity, and the exact output inventory. The presence of an output
filename is never proof of completion.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticLifecycleStage,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
    DiagnosticLifecycleParent,
)
from sol_execbench.core.data.base_model import (
    CurrentFrozenSchemaModel,
    FrozenArtifactModel,
)
from sol_execbench.core.integrity import SHA256Digest
from sol_execbench.core.integrity.schema_versions import SchemaVersion


class DiagnosticStageReceipt(CurrentFrozenSchemaModel):
    """One typed receipt for a completed lifecycle stage.

    ``verification`` is ``receipt_verified`` only when a verifier has
    re-checked all input identities and the exact output inventory against
    the cited values; no writer may stamp it without that re-check.
    """

    current_schema_version = SchemaVersion.DIAGNOSTIC_STAGE_RECEIPT

    schema_version: Literal[SchemaVersion.DIAGNOSTIC_STAGE_RECEIPT] = (
        SchemaVersion.DIAGNOSTIC_STAGE_RECEIPT
    )
    stage: DiagnosticLifecycleStage
    stage_id: str = Field(min_length=1)
    producer_version: str = "4.0.0"
    command: str = Field(min_length=1)
    started_at: str = Field(min_length=1)
    finished_at: str = Field(min_length=1)
    attempts: int = Field(ge=1)
    input_identities: tuple[DiagnosticLifecycleParent, ...] = ()
    output_inventory: tuple[DiagnosticLifecycleArtifact, ...] = ()
    verification: Literal["receipt_verified"] = "receipt_verified"


class DiagnosticCaseReceipt(FrozenArtifactModel):
    """One per-case collection receipt within a stage run.

    A case is complete only when its evidence and SOLAR manifests match the
    expected digests and the output inventory is exact.
    """

    case_id: str = Field(min_length=1)
    expected_evidence_manifest_sha256: SHA256Digest
    expected_solar_manifest_sha256: SHA256Digest
    outputs: tuple[DiagnosticLifecycleArtifact, ...] = ()


__all__ = [
    "DiagnosticCaseReceipt",
    "DiagnosticStageReceipt",
]
