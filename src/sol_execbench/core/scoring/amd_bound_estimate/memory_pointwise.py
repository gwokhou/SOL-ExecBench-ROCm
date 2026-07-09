# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Memory, pointwise, reduction, and conversion AMD bound work estimators."""

from __future__ import annotations

from typing import Any

from sol_execbench.core.scoring.amd_bound_estimate.models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph.models import BoundGraph, BoundGraphNode
from sol_execbench.core.scoring.confidence import EstimateConfidence
from sol_execbench.core.scoring.amd_bound_estimate.common import (
    _axis_evidence,
    _estimate_tensors,
    _join_rationale,
    _pointwise_estimate,
    _sum_tensor_bytes,
    _sum_tensor_numel,
)


def _elementwise_estimate(
    graph: BoundGraph, node: BoundGraphNode
) -> OperatorWorkEstimate:
    return _pointwise_estimate(
        graph,
        node,
        formula_kind="elementwise_flops",
        formula="output_elements",
        formula_inputs_extra={},
        rationale="elementwise work estimated as one operation per output element",
    )


def _activation_estimate(
    graph: BoundGraph, node: BoundGraphNode
) -> OperatorWorkEstimate:
    return _pointwise_estimate(
        graph,
        node,
        formula_kind="activation_flops",
        formula="activation_ops_per_element*output_elements",
        formula_inputs_extra={"activation_ops_per_element": 1},
        rationale="activation work conservatively estimated as one operation per output element",
    )


def _reduction_estimate(
    graph: BoundGraph, node: BoundGraphNode
) -> OperatorWorkEstimate:
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(
        graph, node
    )
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    input_elements = (
        _sum_tensor_numel(input_tensors, "input", warnings, rationale_parts) or 0
    )
    axis_source, axis = _axis_evidence(node)
    formula_inputs: dict[str, Any] = {"input_elements": input_elements, "axis": axis}
    total_bytes = read_bytes + write_bytes
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=node.op_family,
        op_name=node.op_name,
        formula_kind="reduction_flops",
        formula="input_elements",
        formula_inputs=formula_inputs,
        flops=float(input_elements),
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=EstimateConfidence.INEXACT,
        rationale=_join_rationale(
            "conservative reduction pass-count estimate over input elements",
            rationale_parts,
        ),
        axis_source=axis_source,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _normalization_estimate(
    graph: BoundGraph, node: BoundGraphNode
) -> OperatorWorkEstimate:
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(
        graph, node
    )
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    input_elements = (
        _sum_tensor_numel(input_tensors, "input", warnings, rationale_parts) or 0
    )
    axis_source, axis = _axis_evidence(node)
    normalization_passes = 4
    formula_inputs: dict[str, Any] = {
        "input_elements": input_elements,
        "normalization_passes": normalization_passes,
        "axis": axis,
    }
    total_bytes = read_bytes + write_bytes
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=node.op_family,
        op_name=node.op_name,
        formula_kind="normalization_flops",
        formula="normalization_passes*input_elements",
        formula_inputs=formula_inputs,
        flops=float(normalization_passes * input_elements),
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=EstimateConfidence.INEXACT,
        rationale=_join_rationale(
            "conservative normalization pass-count estimate over input elements",
            rationale_parts,
        ),
        axis_source=axis_source,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _softmax_estimate(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(
        graph, node
    )
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    input_elements = (
        _sum_tensor_numel(input_tensors, "input", warnings, rationale_parts) or 0
    )
    axis_source, axis = _axis_evidence(node)
    softmax_passes = 5
    formula_inputs: dict[str, Any] = {
        "input_elements": input_elements,
        "softmax_passes": softmax_passes,
        "axis": axis,
    }
    total_bytes = read_bytes + write_bytes
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=node.op_family,
        op_name=node.op_name,
        formula_kind="softmax_flops",
        formula="softmax_passes*input_elements",
        formula_inputs=formula_inputs,
        flops=float(softmax_passes * input_elements),
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=EstimateConfidence.INEXACT,
        rationale=_join_rationale(
            "conservative softmax-like pass-count estimate covering max, exp, sum, and normalize",
            rationale_parts,
        ),
        axis_source=axis_source,
        warnings=tuple(dict.fromkeys(warnings)),
    )
