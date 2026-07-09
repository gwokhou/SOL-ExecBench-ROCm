# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Data models for AMD bound work estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sol_execbench.core.scoring.amd_bound_graph.enums import OpFamily
from sol_execbench.core.scoring.confidence import EstimateConfidence


@dataclass(frozen=True)
class OperatorWorkEstimate:
    """Auditable work estimate for one BoundGraph operation node."""

    node_id: str
    op_family: OpFamily
    op_name: str
    formula_kind: str
    formula: str
    formula_inputs: dict[str, Any]
    flops: float
    read_bytes: float
    write_bytes: float
    intermediate_bytes: float
    movement_bytes: float
    total_bytes: float
    confidence: EstimateConfidence
    rationale: str
    axis_source: str | None = None
    movement_kind: str | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize as JSON-safe derived estimate evidence."""
        return {
            "node_id": self.node_id,
            "op_family": self.op_family.value,
            "op_name": self.op_name,
            "formula_kind": self.formula_kind,
            "formula": self.formula,
            "formula_inputs": dict(self.formula_inputs),
            "flops": self.flops,
            "read_bytes": self.read_bytes,
            "write_bytes": self.write_bytes,
            "intermediate_bytes": self.intermediate_bytes,
            "movement_bytes": self.movement_bytes,
            "total_bytes": self.total_bytes,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
            "axis_source": self.axis_source,
            "movement_kind": self.movement_kind,
            "warnings": list(self.warnings),
        }
