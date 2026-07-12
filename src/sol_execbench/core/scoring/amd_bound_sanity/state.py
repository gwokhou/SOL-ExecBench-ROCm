# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Typed mutable state used while auditing AMD bound evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import AmdBoundSanitySourceStatuses


@dataclass
class WorkloadAuditState:
    """Normalized evidence accumulated for one workload before serialization."""

    problem_id: str
    workload_uuid: str
    category: str = "unknown"
    problem_path: str | None = None
    definition: str | None = None
    row_index: int | None = None
    diagnostic_flags: set[str] = field(default_factory=set)
    source_statuses: AmdBoundSanitySourceStatuses = field(
        default_factory=AmdBoundSanitySourceStatuses
    )
    amd_score_supported: bool | None = None
    coverage_summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    evidence_refs: dict[str, str] = field(default_factory=dict)
    evidence_gaps: list[str] = field(default_factory=list)
    blocker_codes: set[str] = field(default_factory=set)


@dataclass
class EvidenceGapState:
    """Mutable aggregate for one missing-evidence reason."""

    reason_code: str
    count: int = 0
    example_refs: list[str] = field(default_factory=list)


@dataclass
class SanityAuditState:
    """All normalized workload and aggregate state produced during ingestion."""

    workloads: dict[str, WorkloadAuditState] = field(default_factory=dict)
    evidence_gap_groups: dict[str, EvidenceGapState] = field(default_factory=dict)
    amd_sol_statuses: dict[str, int] = field(default_factory=dict)
    solar_statuses: dict[str, int] = field(default_factory=dict)
    operator_counts: dict[str, int] = field(default_factory=dict)
    op_family_counts: dict[str, int] = field(default_factory=dict)
    blocker_counts_by_operator: dict[str, dict[str, int]] = field(default_factory=dict)
    blocker_counts_by_op_family: dict[str, dict[str, int]] = field(default_factory=dict)


__all__ = ["EvidenceGapState", "SanityAuditState", "WorkloadAuditState"]
