# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Data models for fusion-aware AMD SOL v3 sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sol_execbench.core.platform.arch_capabilities import ArchIsaBudget
from sol_execbench.core.scoring.amd_hardware_models import AmdHardwareModel
from sol_execbench.core.scoring.amd_sol.fusion import FusionGroup
from sol_execbench.core.scoring.confidence import EstimateConfidence


AMD_SOL_V3_SCHEMA_VERSION = "sol_execbench.amd_sol_bound.v3"
AGGREGATE_STATUSES = frozenset({"scored", "degraded", "unscored"})


@dataclass(frozen=True)
class AmdSolAggregateBound:
    """Aggregate bound and score-eligibility state for a v3 sidecar."""

    status: str
    scored: bool
    sol_bound_ms: float
    reason: str
    node_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "scored": self.scored,
            "sol_bound_ms": self.sol_bound_ms,
            "reason": self.reason,
            "node_ids": list(self.node_ids),
        }


@dataclass(frozen=True)
class AmdSolCoverageSummary:
    """Family-aware coverage summary for a v3 sidecar."""

    total_ops: int
    supported_ops: int
    inexact_ops: int
    unsupported_ops: int
    op_family_counts: dict[str, int]
    confidence_counts_by_family: dict[str, dict[str, int]]
    worst_confidence: EstimateConfidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_ops": self.total_ops,
            "supported_ops": self.supported_ops,
            "inexact_ops": self.inexact_ops,
            "unsupported_ops": self.unsupported_ops,
            "op_family_counts": dict(sorted(self.op_family_counts.items())),
            "confidence_counts_by_family": {
                family: dict(sorted(counts.items()))
                for family, counts in sorted(self.confidence_counts_by_family.items())
            },
            "worst_confidence": self.worst_confidence.value,
        }


@dataclass(frozen=True)
class AmdSolV3GroupBound:
    """Roofline bound for a fusion group or a singleton node."""

    group_id: str
    pattern_id: str
    node_ids: tuple[str, ...]
    compute_bound_ms: float
    memory_bound_ms: float
    sol_bound_ms: float
    limiting_resource: str
    confidence: EstimateConfidence
    rationale: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "pattern_id": self.pattern_id,
            "node_ids": list(self.node_ids),
            "compute_bound_ms": self.compute_bound_ms,
            "memory_bound_ms": self.memory_bound_ms,
            "sol_bound_ms": self.sol_bound_ms,
            "limiting_resource": self.limiting_resource,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class AmdSolBoundV3Artifact:
    """Stable fusion-aware AMD SOL artifact.

    v3 intentionally retains the rich v2 node estimates so coverage reports can
    explain why a group is blocked without reverse engineering group totals.
    """

    definition: str
    workload_uuid: str
    hardware_model_ref: str | None
    hardware_model: AmdHardwareModel
    capability_budget_ref: str | None
    capability_budget: ArchIsaBudget | None
    bound_graph: dict[str, object]
    operator_work_estimates: tuple[dict[str, object], ...]
    fusion_groups: tuple[FusionGroup, ...]
    group_bounds: tuple[AmdSolV3GroupBound, ...]
    aggregate_bound: AmdSolAggregateBound
    warnings: tuple[str, ...]
    coverage_summary: AmdSolCoverageSummary
    schema_version: str = AMD_SOL_V3_SCHEMA_VERSION
    derived: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "derived": self.derived,
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "hardware_model_ref": self.hardware_model_ref,
            "hardware_model": self.hardware_model.to_dict(),
            "capability_budget_ref": self.capability_budget_ref,
            "capability_budget": (
                self.capability_budget.model_dump(mode="json")
                if self.capability_budget is not None
                else None
            ),
            "bound_graph": dict(self.bound_graph),
            "operator_work_estimates": [
                dict(estimate) for estimate in self.operator_work_estimates
            ],
            "fusion_groups": [group.to_dict() for group in self.fusion_groups],
            "group_bounds": [bound.to_dict() for bound in self.group_bounds],
            "aggregate_bound": self.aggregate_bound.to_dict(),
            "warnings": list(self.warnings),
            "coverage_summary": self.coverage_summary.to_dict(),
        }
