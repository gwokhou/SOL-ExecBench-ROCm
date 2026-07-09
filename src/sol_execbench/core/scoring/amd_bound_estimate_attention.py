# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Attention-family AMD bound work estimators."""

from __future__ import annotations

from typing import Any

from sol_execbench.core.scoring.amd_bound_estimate_models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph_models import (
    BoundGraph,
    BoundGraphNode,
    OpFamily,
)
from sol_execbench.core.scoring.confidence import EstimateConfidence

from sol_execbench.core.scoring.amd_bound_estimate_common import (
    _attention_dims_from_attributes,
    _attention_dims_from_node_or_shapes,
    _attention_output_projection_dims,
    _axis_evidence,
    _estimate_tensors,
    _join_rationale,
    _pointwise_estimate,
    _sum_tensor_bytes,
    _sum_tensor_numel,
    _unsupported_estimate,
)

def _attention_estimate(
    graph: BoundGraph, node: BoundGraphNode
) -> OperatorWorkEstimate:
    subrole = str(node.attributes.get("subrole") or "")
    if node.confidence == EstimateConfidence.UNSUPPORTED:
        return _unsupported_estimate(
            node,
            rationale=node.rationale,
            warnings=("unsupported_operator:dynamic_attention_axes",),
        )
    if subrole == "qk_scores":
        return _attention_matmul_estimate(
            graph,
            node,
            formula_kind="attention_scores_flops",
            formula="2*B*H*S_q*S_k*D",
            rationale="attention QK score FLOPs estimated from tensor shapes",
        )
    if subrole == "pv_aggregation":
        return _attention_matmul_estimate(
            graph,
            node,
            formula_kind="attention_pv_flops",
            formula="2*B*H*S_q*S_k*D",
            rationale="attention PV aggregation FLOPs estimated from tensor shapes",
        )
    if subrole == "softmax":
        return _attention_softmax_estimate(graph, node)
    if subrole == "scale_or_mask":
        return _attention_scale_or_mask_estimate(graph, node)
    if subrole == "output_projection":
        return _attention_output_projection_estimate(graph, node)
    return _unsupported_estimate(
        node,
        rationale=f"unsupported attention subrole: {subrole or '<missing>'}",
        warnings=(f"unsupported_operator:attention_{subrole or 'missing_subrole'}",),
    )


def _attention_matmul_estimate(
    graph: BoundGraph,
    node: BoundGraphNode,
    *,
    formula_kind: str,
    formula: str,
    rationale: str,
) -> OperatorWorkEstimate:
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(
        graph, node
    )
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    dims = _attention_dims_from_node_or_shapes(input_tensors, output_tensors, node)
    formula_inputs: dict[str, Any] = {}
    flops = 0.0
    axis_source: str | None = None
    if dims is None:
        warnings.append("inexact_operator:attention_missing_dimensions")
        rationale_parts.append("missing attention dimensions")
        confidence = EstimateConfidence.INEXACT
    else:
        formula_inputs = dims
        flops = float(2 * dims["B"] * dims["H"] * dims["S_q"] * dims["S_k"] * dims["D"])
        confidence = (
            EstimateConfidence.SUPPORTED if not warnings else EstimateConfidence.INEXACT
        )
        axis_source = "tensor_shapes"
    total_bytes = read_bytes + write_bytes
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=OpFamily.ATTENTION,
        op_name=node.op_name,
        formula_kind=formula_kind,
        formula=formula,
        formula_inputs=formula_inputs,
        flops=flops,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=confidence,
        rationale=_join_rationale(rationale, rationale_parts),
        axis_source=axis_source,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _attention_softmax_estimate(
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
    if axis is None:
        warnings.append("inexact_operator:attention_softmax_missing_axis")
    dims = _attention_dims_from_attributes(node)
    formula_inputs: dict[str, Any] = {"input_elements": input_elements, "axis": axis}
    if dims is not None:
        formula_inputs.update(dims)
    softmax_passes = 5
    formula_inputs["softmax_passes"] = softmax_passes
    total_bytes = read_bytes + write_bytes
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=OpFamily.ATTENTION,
        op_name=node.op_name,
        formula_kind="attention_softmax_flops",
        formula="softmax_passes*B*H*S_q*S_k",
        formula_inputs=formula_inputs,
        flops=float(softmax_passes * input_elements),
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=EstimateConfidence.SUPPORTED
        if axis is not None and not warnings
        else EstimateConfidence.INEXACT,
        rationale=_join_rationale(
            "attention softmax pass-count estimate over score matrix",
            rationale_parts,
        ),
        axis_source=axis_source,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _attention_scale_or_mask_estimate(
    graph: BoundGraph, node: BoundGraphNode
) -> OperatorWorkEstimate:
    estimate = _pointwise_estimate(
        graph,
        node,
        formula_kind="attention_mask_bytes",
        formula="output_elements",
        formula_inputs_extra={
            "mask_semantics": node.attributes.get("mask_semantics"),
        },
        rationale="attention scale or mask handling estimated from visible tensor movement",
    )
    warnings = list(estimate.warnings)
    confidence = estimate.confidence
    if node.attributes.get("mask_semantics") == "partial":
        warnings.extend(
            ("inexact_operator:attention_mask", "inexact_mask:missing_sparsity")
        )
        confidence = EstimateConfidence.INEXACT
    return OperatorWorkEstimate(
        node_id=estimate.node_id,
        op_family=OpFamily.ATTENTION,
        op_name=estimate.op_name,
        formula_kind=estimate.formula_kind,
        formula=estimate.formula,
        formula_inputs=estimate.formula_inputs,
        flops=estimate.flops,
        read_bytes=estimate.read_bytes,
        write_bytes=estimate.write_bytes,
        intermediate_bytes=estimate.intermediate_bytes,
        movement_bytes=estimate.movement_bytes,
        total_bytes=estimate.total_bytes,
        confidence=confidence,
        rationale=estimate.rationale,
        axis_source="tensor_shapes",
        movement_kind=estimate.movement_kind,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _attention_output_projection_estimate(
    graph: BoundGraph,
    node: BoundGraphNode,
) -> OperatorWorkEstimate:
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(
        graph, node
    )
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    dims = _attention_output_projection_dims(input_tensors, output_tensors, node)
    if dims is None:
        warnings.append("inexact_operator:attention_output_projection_missing_shape")
        formula_inputs: dict[str, Any] = {}
        flops = 0.0
        axis_source = None
        confidence = EstimateConfidence.INEXACT
    else:
        formula_inputs = dims
        flops = float(2 * dims["M"] * dims["N"] * dims["K"])
        axis_source = "tensor_shapes"
        confidence = (
            EstimateConfidence.SUPPORTED if not warnings else EstimateConfidence.INEXACT
        )
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=OpFamily.ATTENTION,
        op_name=node.op_name,
        formula_kind="gemm_flops",
        formula="2*M*N*K",
        formula_inputs=formula_inputs,
        flops=flops,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=read_bytes + write_bytes,
        confidence=confidence,
        rationale=_join_rationale(
            "attention output projection uses GEMM-compatible estimate",
            rationale_parts,
        ),
        axis_source=axis_source,
        warnings=tuple(dict.fromkeys(warnings)),
    )
