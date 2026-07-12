# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Models for cross-report consistency diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from sol_execbench.core.data.json_utils import stable_model_checksum, stable_model_json
from sol_execbench.core.dataset.manifest import DatasetManifestChecksum

CONSISTENCY_REPORT_SCHEMA_VERSION = "sol_execbench.consistency_report.v1"

CLAIM_BOUNDARY_TEXT = (
    "This report is diagnostic-only cross-report consistency lint: not score "
    "authority, not paper parity, not leaderboard authority, not native-host "
    "validation, and not new-hardware validation."
)


@dataclass(frozen=True, slots=True)
class ConsistencyInputs:
    """Read-only artifact boundary for one consistency-report build."""

    execution_closure: Mapping[str, object] | None = None
    paper_denominator: Mapping[str, object] | None = None
    matrix_report: Mapping[str, object] | None = None
    runtime_evidence: Mapping[str, object] | None = None
    static_evidence: Mapping[str, object] | None = None
    amd_score_report: Mapping[str, object] | None = None
    amd_sol_report: Mapping[str, object] | None = None
    solar_derivation: Mapping[str, object] | None = None
    amd_bound_sanity: Mapping[str, object] | None = None
    source_paths: Mapping[str, Path | None] = field(default_factory=dict)
    created_at: str | None = None


class ConsistencySourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    path: str | None = None
    ref: str | None = None
    schema_version: str | None = None
    checksum: str | None = None


class ConsistencyFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: str
    reason_code: str
    sources: list[str]
    refs: list[str] = Field(default_factory=list)
    message: str
    next_step: str


class ConsistencyFindingTotals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blocker: int = 0
    warning: int = 0
    info: int = 0

    def add(self, severity: str) -> None:
        if severity not in {"blocker", "warning", "info"}:
            raise ValueError(f"Unknown consistency severity: {severity}")
        setattr(self, severity, getattr(self, severity) + 1)


class ConsistencySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources_checked: int
    findings_total: int
    finding_totals: ConsistencyFindingTotals


class ConsistencyClaimBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnostic_only: bool = True
    score_authority: bool = False
    paper_parity: bool = False
    leaderboard_authority: bool = False
    native_host_validation: bool = False
    new_hardware_validation: bool = False


class ConsistencyReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = CONSISTENCY_REPORT_SCHEMA_VERSION
    created_at: str
    sources: list[ConsistencySourceRef]
    summary: ConsistencySummary
    findings: list[ConsistencyFinding]
    claim_boundary: ConsistencyClaimBoundary = Field(
        default_factory=ConsistencyClaimBoundary
    )
    report_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "ConsistencyReport":
        return self.model_copy(
            update={
                "report_checksum": DatasetManifestChecksum(
                    value=stable_model_checksum(self, "report_checksum")
                )
            }
        )

    def to_json(self) -> str:
        return stable_model_json(self)
