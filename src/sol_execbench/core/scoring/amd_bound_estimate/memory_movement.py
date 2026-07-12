# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Memory, pointwise, reduction, and conversion AMD bound work estimators."""

from __future__ import annotations


from sol_execbench.core.scoring.amd_bound_estimate.models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph.models import BoundGraph, BoundGraphNode
from sol_execbench.core.scoring.confidence import EstimateConfidence
from sol_execbench.core.scoring.amd_bound_estimate.common import (
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
    if node.attributes.get("materialization_kind") == "fill":
        # zeros_like consumes shape/dtype metadata but does not read tensor data.
        read_bytes = 0.0
    movement_kind = str(
        node.attributes.get("movement_kind") or _movement_kind_from_op_name(node)
    )
    if movement_kind == "materialized":
        movement_bytes = read_bytes + write_bytes
        total_bytes = movement_bytes
        rationale = (
            "materialized data movement estimate for contiguous or copy-like operation"
        )
    elif movement_kind == "broadcast_view":
        movement_bytes = 0.0
        total_bytes = 0.0
        rationale = "broadcast view evidence with zero movement bytes"
    else:
        movement_kind = "logical_view"
        movement_bytes = 0.0
        total_bytes = 0.0
        rationale = "logical view evidence with zero movement bytes"
    confidence = (
        EstimateConfidence.SUPPORTED
        if node.confidence == EstimateConfidence.SUPPORTED and not warnings
        else EstimateConfidence.INEXACT
    )
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
        confidence=confidence,
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
    source_dtype = _first_tensor_dtype(input_tensors)
    output_dtype = _first_tensor_dtype(output_tensors)
    target_dtype = node.attributes.get("target_dtype") or _first_tensor_dtype(
        output_tensors
    )
    if target_dtype is None or _dtype_bytes(str(target_dtype)) is None:
        warnings.append("inexact_dtype_conversion:missing_target_dtype")
        rationale_parts.append("missing target dtype for dtype conversion")
    exact_contract = (
        len(input_tensors) == 1
        and len(output_tensors) == 1
        and input_tensors[0].shape is not None
        and input_tensors[0].shape == output_tensors[0].shape
        and source_dtype is not None
        and target_dtype is not None
        and output_dtype == target_dtype
        and _dtype_bytes(str(source_dtype)) is not None
        and _dtype_bytes(str(target_dtype)) is not None
        and not warnings
    )
    no_op = exact_contract and source_dtype == target_dtype
    if no_op:
        read_bytes = 0.0
        write_bytes = 0.0
    movement_bytes = read_bytes + write_bytes
    total_bytes = movement_bytes
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=node.op_family,
        op_name=node.op_name,
        formula_kind="dtype_conversion_bytes",
        formula="0" if no_op else "read_bytes+write_bytes",
        formula_inputs={
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "target_dtype": str(target_dtype) if target_dtype is not None else None,
            "source_dtype": source_dtype,
            "no_op": no_op,
        },
        flops=0.0,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=movement_bytes,
        total_bytes=total_bytes,
        confidence=(
            EstimateConfidence.SUPPORTED
            if exact_contract
            else EstimateConfidence.INEXACT
        ),
        rationale=_join_rationale(
            (
                "shape-proved no-op dtype conversion"
                if no_op
                else "shape-proved dtype conversion movement estimate"
                if exact_contract
                else "dtype conversion movement estimate"
            ),
            rationale_parts,
        ),
        movement_kind="dtype_conversion",
        warnings=tuple(dict.fromkeys(warnings)),
    )
