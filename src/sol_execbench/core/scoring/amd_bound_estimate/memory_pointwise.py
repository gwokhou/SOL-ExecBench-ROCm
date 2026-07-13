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
    output_elements = (
        _sum_tensor_numel(output_tensors, "output", warnings, rationale_parts) or 0
    )
    formula_inputs: dict[str, Any] = {
        "input_elements": input_elements,
        "output_elements": output_elements,
        "axis": axis,
    }
    total_bytes = read_bytes + write_bytes
    exact_sum = _has_exact_sum_reduction_contract(
        node, axis, input_elements, output_elements, warnings
    )
    leaf_name = node.op_name.rsplit(".", maxsplit=1)[-1]
    exact_logsumexp = _has_exact_logsumexp_contract(
        node, axis, input_elements, output_elements, warnings
    )
    flops = (
        float(4 * input_elements)
        if exact_logsumexp
        else float(
            input_elements
            - output_elements
            + (output_elements if leaf_name == "mean" else 0)
        )
        if exact_sum
        else float(input_elements)
    )
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=node.op_family,
        op_name=node.op_name,
        formula_kind="reduction_flops",
        formula=(
            "4*input_elements"
            if exact_logsumexp
            else "input_elements"
            if leaf_name == "mean" and exact_sum
            else "input_elements-output_elements"
            if exact_sum
            else "input_elements"
        ),
        formula_inputs=formula_inputs,
        flops=flops,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=(
            EstimateConfidence.SUPPORTED
            if exact_sum or exact_logsumexp
            else EstimateConfidence.INEXACT
        ),
        rationale=_join_rationale(
            (
                "exact logsumexp operation count from static input and output shapes"
                if exact_logsumexp
                else "exact sum-reduction operation count from static input and output shapes"
                if exact_sum
                else "conservative reduction pass-count estimate over input elements"
            ),
            rationale_parts,
        ),
        axis_source=axis_source,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _has_exact_sum_reduction_contract(
    node: BoundGraphNode,
    axis: object,
    input_elements: int,
    output_elements: int,
    warnings: list[str],
) -> bool:
    """Return whether the visible operation is a statically exact sum reduction."""
    if node.op_name.rsplit(".", maxsplit=1)[-1] not in {"sum", "mean"}:
        return False
    if input_elements <= 0 or output_elements <= 0:
        return False
    if input_elements < output_elements or input_elements % output_elements:
        return False
    return not warnings


def _has_exact_logsumexp_contract(
    node: BoundGraphNode,
    axis: object,
    input_elements: int,
    output_elements: int,
    warnings: list[str],
) -> bool:
    """Return whether stable logsumexp work is fixed by static tensor shapes."""
    if node.op_name.rsplit(".", maxsplit=1)[-1] != "logsumexp" or axis is None:
        return False
    return (
        input_elements > 0
        and output_elements > 0
        and input_elements >= output_elements
        and input_elements % output_elements == 0
        and not warnings
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
    leaf_name = node.op_name.rsplit(".", maxsplit=1)[-1]
    groups = _softmax_group_count(input_tensors, output_tensors, axis, warnings)
    softmax_passes = 5
    formula_inputs: dict[str, Any] = {
        "input_elements": input_elements,
        "softmax_passes": softmax_passes,
        "axis": axis,
        "groups": groups,
    }
    total_bytes = read_bytes + write_bytes
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=node.op_family,
        op_name=node.op_name,
        formula_kind="softmax_flops",
        formula="softmax_passes*input_elements",
        formula_inputs=formula_inputs,
        flops=float(
            softmax_passes * input_elements
            - (groups if leaf_name == "log_softmax" else 2 * groups)
            if groups is not None
            else softmax_passes * input_elements
        ),
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=(
            EstimateConfidence.SUPPORTED
            if groups is not None
            else EstimateConfidence.INEXACT
        ),
        rationale=_join_rationale(
            (
                "exact softmax operation count from static shape and axis"
                if groups is not None
                else "conservative softmax-like pass-count estimate covering max, exp, sum, and normalize"
            ),
            rationale_parts,
        ),
        axis_source=axis_source,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _softmax_group_count(
    input_tensors: tuple[Any, ...],
    output_tensors: tuple[Any, ...],
    axis: object,
    warnings: list[str],
) -> int | None:
    """Return the statically known number of independently reduced rows."""
    if len(input_tensors) != 1 or len(output_tensors) != 1 or warnings:
        return None
    input_shape = input_tensors[0].shape
    output_shape = output_tensors[0].shape
    if input_shape is None or output_shape != input_shape or not isinstance(axis, int):
        return None
    normalized_axis = axis if axis >= 0 else len(input_shape) + axis
    if normalized_axis < 0 or normalized_axis >= len(input_shape):
        return None
    extent = input_shape[normalized_axis]
    if extent <= 0:
        return None
    input_elements = 1
    for dimension in input_shape:
        input_elements *= dimension
    return input_elements // extent
