# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Data models for AMD SOL v2 bound artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sol_execbench.core.scoring.amd_hardware_models import AmdHardwareModel, EstimateConfidence

AMD_SOL_V2_SCHEMA_VERSION = "sol_execbench.amd_sol_bound.v2"
AGGREGATE_STATUSES = frozenset({"scored", "degraded", "unscored"})


@dataclass(frozen=True)
class AmdSolV2OpBound:
    """Per-operation AMD SOL bound derived from rich operator evidence."""

    node_id: str
    op_family: str
    op_name: str
    compute_bound_ms: float
    memory_bound_ms: float
    sol_bound_ms: float
    limiting_resource: str
    confidence: EstimateConfidence
    rationale: str
    estimate_warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "op_family": self.op_family,
            "op_name": self.op_name,
            "compute_bound_ms": self.compute_bound_ms,
            "memory_bound_ms": self.memory_bound_ms,
            "sol_bound_ms": self.sol_bound_ms,
            "limiting_resource": self.limiting_resource,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
            "estimate_warnings": list(self.estimate_warnings),
        }


@dataclass(frozen=True)
class AmdSolV2AggregateBound:
    """Aggregate bound and score eligibility state for a v2 artifact."""

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
class AmdSolV2CoverageSummary:
    """Family-aware coverage summary for v2 AMD SOL bounds."""

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
class AmdSolBoundV2Artifact:
    """Stable AMD SOL bound artifact v2 sidecar."""

    definition: str
    workload_uuid: str
    hardware_model_ref: str | None
    hardware_model: AmdHardwareModel
    bound_graph: dict[str, object]
    operator_work_estimates: tuple[dict[str, object], ...]
    op_bounds: tuple[AmdSolV2OpBound, ...]
    aggregate_bound: AmdSolV2AggregateBound
    warnings: tuple[str, ...]
    coverage_summary: AmdSolV2CoverageSummary
    schema_version: str = AMD_SOL_V2_SCHEMA_VERSION
    derived: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "derived": self.derived,
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "hardware_model_ref": self.hardware_model_ref,
            "hardware_model": self.hardware_model.to_dict(),
            "bound_graph": dict(self.bound_graph),
            "operator_work_estimates": [
                dict(estimate) for estimate in self.operator_work_estimates
            ],
            "op_bounds": [bound.to_dict() for bound in self.op_bounds],
            "aggregate_bound": self.aggregate_bound.to_dict(),
            "warnings": list(self.warnings),
            "coverage_summary": self.coverage_summary.to_dict(),
        }
