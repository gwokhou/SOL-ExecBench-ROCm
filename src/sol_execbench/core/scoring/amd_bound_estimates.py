# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator-level AMD bound work estimates derived from BoundGraph IR."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from sol_execbench.core.data.definition import DType
from sol_execbench.core.scoring.amd_bound_graph import BoundGraph, BoundTensor, OpFamily
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence


@dataclass(frozen=True)
class OperatorWorkEstimate:
    """Auditable work estimate for one BoundGraph operation node."""

    node_id: str
    op_family: OpFamily
    op_name: str
    formula_kind: str
    formula: str
    formula_inputs: dict[str, object]
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

    def to_dict(self) -> dict[str, object]:
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


def estimate_bound_work(graph: BoundGraph) -> tuple[OperatorWorkEstimate, ...]:
    """Estimate per-node operator work from a structured bound graph."""
    return tuple(_unsupported_estimate(node) for node in graph.nodes)


def _unsupported_estimate(node) -> OperatorWorkEstimate:
    warning_kind = (
        "unsupported_operator"
        if node.op_family == OpFamily.UNSUPPORTED
        else "unsupported_family"
    )
    warning = f"{warning_kind}:{node.op_name or node.op_family.value}"
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=node.op_family,
        op_name=node.op_name,
        formula_kind="unsupported",
        formula="0",
        formula_inputs={},
        flops=0.0,
        read_bytes=0.0,
        write_bytes=0.0,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=0.0,
        confidence=EstimateConfidence.UNSUPPORTED,
        rationale=(
            f"unsupported operation estimate for {node.op_family.value}: "
            f"{node.op_name or node.source_expression}"
        ),
        warnings=(warning,),
    )


def _dtype_bytes(dtype: DType | str) -> float | None:
    raw = dtype.value if isinstance(dtype, DType) else str(dtype)
    return {
        DType.FLOAT64.value: 8.0,
        DType.FLOAT32.value: 4.0,
        DType.FLOAT16.value: 2.0,
        DType.BFLOAT16.value: 2.0,
        DType.FLOAT8_E4M3FN.value: 1.0,
        DType.FLOAT8_E5M2.value: 1.0,
        DType.FLOAT4_E2M1.value: 0.5,
        DType.FLOAT4_E2M1FN_X2.value: 0.5,
        DType.INT64.value: 8.0,
        DType.INT32.value: 4.0,
        DType.INT16.value: 2.0,
        DType.INT8.value: 1.0,
        DType.BOOL.value: 1.0,
    }.get(raw)


def _tensor_numel(tensor: BoundTensor) -> int | None:
    if tensor.shape is None:
        return None
    return prod(tensor.shape)


def _tensor_bytes(tensor: BoundTensor) -> float | None:
    numel = _tensor_numel(tensor)
    dtype_bytes = _dtype_bytes(tensor.dtype)
    if numel is None or dtype_bytes is None:
        return None
    return float(numel * dtype_bytes)
