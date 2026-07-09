# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Data models for AMD SOL v1 bound artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sol_execbench.core.scoring.amd_hardware_models import AmdHardwareModel
from sol_execbench.core.scoring.confidence import EstimateConfidence

AMD_SOL_SCHEMA_VERSION = "sol_execbench.amd_sol_bound.v1"


@dataclass(frozen=True)
class GraphNode:
    """Normalized operation graph node."""

    node_id: str
    op_type: str
    expression: str
    confidence: EstimateConfidence
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "op_type": self.op_type,
            "expression": self.expression,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class WorkEstimate:
    """Estimated FLOPs and bytes for one graph node."""

    node_id: str
    flops: float
    bytes_accessed: float
    confidence: EstimateConfidence
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "flops": self.flops,
            "bytes_accessed": self.bytes_accessed,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class OpSolBound:
    """Per-operation AMD speed-of-light bound."""

    node_id: str
    compute_bound_ms: float
    memory_bound_ms: float
    sol_bound_ms: float
    limiting_resource: str
    confidence: EstimateConfidence
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "compute_bound_ms": self.compute_bound_ms,
            "memory_bound_ms": self.memory_bound_ms,
            "sol_bound_ms": self.sol_bound_ms,
            "limiting_resource": self.limiting_resource,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class AmdSolBoundArtifact:
    """Auditable AMD SOL bound artifact for one workload."""

    definition: str
    workload_uuid: str
    hardware_model: AmdHardwareModel
    graph_nodes: tuple[GraphNode, ...]
    work_estimates: tuple[WorkEstimate, ...]
    op_bounds: tuple[OpSolBound, ...]
    coverage_summary: AmdSolCoverageSummary
    schema_version: str = AMD_SOL_SCHEMA_VERSION
    derived: bool = True

    @property
    def aggregate_sol_bound_ms(self) -> float:
        """Aggregate SOL bound as the sum of per-op bounds."""
        return sum(bound.sol_bound_ms for bound in self.op_bounds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "derived": self.derived,
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "hardware_model": self.hardware_model.to_dict(),
            "graph_nodes": [node.to_dict() for node in self.graph_nodes],
            "work_estimates": [estimate.to_dict() for estimate in self.work_estimates],
            "op_bounds": [bound.to_dict() for bound in self.op_bounds],
            "aggregate_sol_bound_ms": self.aggregate_sol_bound_ms,
            "coverage_summary": self.coverage_summary.to_dict(),
        }


@dataclass(frozen=True)
class AmdSolCoverageSummary:
    """Derived AMD SOL coverage summary for one graph or artifact."""

    total_ops: int
    supported_ops: int
    inexact_ops: int
    unsupported_ops: int
    op_type_counts: dict[str, int]
    derived: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "derived": self.derived,
            "total_ops": self.total_ops,
            "supported_ops": self.supported_ops,
            "inexact_ops": self.inexact_ops,
            "unsupported_ops": self.unsupported_ops,
            "op_type_counts": dict(self.op_type_counts),
        }
