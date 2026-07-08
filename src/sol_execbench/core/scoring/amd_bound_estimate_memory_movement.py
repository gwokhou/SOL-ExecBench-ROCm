# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Memory, pointwise, reduction, and conversion AMD bound work estimators."""

from __future__ import annotations


from sol_execbench.core.scoring.amd_bound_estimate_models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph_models import BoundGraph, BoundGraphNode
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence
from sol_execbench.core.scoring.amd_bound_estimate_common import (
    _dtype_bytes,
    _estimate_tensors,
    _first_tensor_dtype,
    _join_rationale,
    _movement_kind_from_op_name,
    _sum_tensor_bytes,
)


def _data_movement_estimate(
    graph: BoundGraph, node: BoundGraphNode
) -> OperatorWorkEstimate:
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(
        graph, node
    )
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    movement_kind = str(
        node.attributes.get("movement_kind") or _movement_kind_from_op_name(node)
    )
    if movement_kind == "materialized":
        movement_bytes = read_bytes + write_bytes
        rationale = (
            "materialized data movement estimate for contiguous or copy-like operation"
        )
    elif movement_kind == "broadcast_view":
        movement_bytes = 0.0
        rationale = "broadcast view evidence with zero movement bytes"
    else:
        movement_kind = "logical_view"
        movement_bytes = 0.0
        rationale = "logical view evidence with zero movement bytes"
    total_bytes = read_bytes + write_bytes
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=node.op_family,
        op_name=node.op_name,
        formula_kind="data_movement_bytes",
        formula="movement_bytes",
        formula_inputs={"movement_bytes": movement_bytes},
        flops=0.0,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=movement_bytes,
        total_bytes=total_bytes,
        confidence=EstimateConfidence.INEXACT,
        rationale=_join_rationale(rationale, rationale_parts),
        movement_kind=movement_kind,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _dtype_conversion_estimate(
    graph: BoundGraph, node: BoundGraphNode
) -> OperatorWorkEstimate:
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(
        graph, node
    )
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    target_dtype = node.attributes.get("target_dtype") or _first_tensor_dtype(
        output_tensors
    )
    if target_dtype is None or _dtype_bytes(str(target_dtype)) is None:
        warnings.append("inexact_dtype_conversion:missing_target_dtype")
        rationale_parts.append("missing target dtype for dtype conversion")
    movement_bytes = read_bytes + write_bytes
    total_bytes = read_bytes + write_bytes
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=node.op_family,
        op_name=node.op_name,
        formula_kind="dtype_conversion_bytes",
        formula="read_bytes+write_bytes",
        formula_inputs={
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "target_dtype": str(target_dtype) if target_dtype is not None else None,
        },
        flops=0.0,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=movement_bytes,
        total_bytes=total_bytes,
        confidence=EstimateConfidence.INEXACT,
        rationale=_join_rationale(
            "dtype conversion movement estimate", rationale_parts
        ),
        movement_kind="dtype_conversion",
        warnings=tuple(dict.fromkeys(warnings)),
    )
