# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Work estimation helpers for AMD SOL v1 artifacts."""

from __future__ import annotations

from math import prod

from sol_execbench.core.data.definition import DType, Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_bound_estimates import (
    OperatorWorkEstimate,
    estimate_bound_work,
)
from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence
from sol_execbench.core.scoring.amd_sol_models import GraphNode, WorkEstimate

def estimate_work(
    definition: Definition,
    workload: Workload,
    graph_nodes: tuple[GraphNode, ...],
) -> tuple[WorkEstimate, ...]:
    """Estimate FLOPs and bytes for graph nodes."""
    try:
        bound_graph = build_bound_graph(definition, workload)
        rich_estimates = estimate_bound_work(bound_graph)
        if graph_nodes and len(graph_nodes) != len(rich_estimates):
            raise ValueError(
                "legacy graph node count does not match rich bound estimate count"
            )
        return tuple(
            _work_estimate_from_rich_estimate(
                estimate,
                node_id=graph_nodes[index].node_id if graph_nodes else estimate.node_id,
            )
            for index, estimate in enumerate(rich_estimates)
        )
    except Exception as exc:
        return _legacy_estimate_work(
            definition,
            workload,
            graph_nodes,
            fallback_reason=f"rich bound estimate failed: {exc}",
        )


def _legacy_estimate_work(
    definition: Definition,
    workload: Workload,
    graph_nodes: tuple[GraphNode, ...],
    *,
    fallback_reason: str,
) -> tuple[WorkEstimate, ...]:
    """Legacy whole-definition estimator retained as explicit exceptional fallback."""
    axes = definition.get_resolved_axes_values(workload.axes)
    input_shapes = definition.get_input_shapes(workload.axes)
    output_shapes = definition.get_output_shapes(workload.axes)
    tensor_bytes = _tensor_bytes(definition, input_shapes, output_shapes)
    output_numel = sum(prod(shape) for shape in output_shapes.values() if shape)
    input_numel = sum(prod(shape) for shape in input_shapes.values() if shape)
    reduction_dim = _largest_reduction_dim(definition, axes)
    return tuple(
        _legacy_estimate_for_node(
            node,
            tensor_bytes=tensor_bytes,
            output_numel=output_numel,
            input_numel=input_numel,
            reduction_dim=reduction_dim,
            fallback_reason=fallback_reason,
        )
        for node in graph_nodes
    )


def _legacy_estimate_for_node(
    node: GraphNode,
    *,
    tensor_bytes: float,
    output_numel: int,
    input_numel: int,
    reduction_dim: int,
    fallback_reason: str,
) -> WorkEstimate:
    if node.op_type == "matmul" and output_numel and reduction_dim:
        return WorkEstimate(
            node_id=node.node_id,
            flops=float(2 * output_numel * reduction_dim),
            bytes_accessed=float(tensor_bytes),
            confidence=EstimateConfidence.SUPPORTED,
            rationale=(
                "legacy fallback: matmul FLOPs estimated as 2 * output "
                f"elements * reduction dimension ({fallback_reason})"
            )
        )
    if node.op_type in {"elementwise", "activation"} and output_numel:
        return WorkEstimate(
            node_id=node.node_id,
            flops=float(output_numel),
            bytes_accessed=float(tensor_bytes),
            confidence=EstimateConfidence.INEXACT,
            rationale=(
                f"legacy fallback: {node.op_type} work estimated as one "
                f"operation per output element ({fallback_reason})"
            ),
        )
    if node.op_type == "reduction" and (input_numel or output_numel):
        return WorkEstimate(
            node_id=node.node_id,
            flops=float(max(input_numel, output_numel)),
            bytes_accessed=float(tensor_bytes),
            confidence=EstimateConfidence.INEXACT,
            rationale=(
                "legacy fallback: reduction work conservatively estimated "
                f"from input elements ({fallback_reason})"
            ),
        )
    if node.op_type == "normalization" and (input_numel or output_numel):
        return WorkEstimate(
            node_id=node.node_id,
            flops=float(4 * max(input_numel, output_numel)),
            bytes_accessed=float(tensor_bytes),
            confidence=EstimateConfidence.INEXACT,
            rationale=(
                "legacy fallback: normalization-like work conservatively "
                "estimates reductions, scaling, and elementwise application "
                f"({fallback_reason})"
            ),
        )
    if node.op_type == "softmax" and (input_numel or output_numel):
        return WorkEstimate(
            node_id=node.node_id,
            flops=float(5 * max(input_numel, output_numel)),
            bytes_accessed=float(tensor_bytes),
            confidence=EstimateConfidence.INEXACT,
            rationale=(
                "legacy fallback: softmax-like work conservatively estimates "
                f"max, exp, sum, and normalization passes ({fallback_reason})"
            ),
        )
    if node.op_type == "data_movement":
        return WorkEstimate(
            node_id=node.node_id,
            flops=0.0,
            bytes_accessed=float(tensor_bytes),
            confidence=EstimateConfidence.INEXACT,
            rationale=(
                "legacy fallback: data movement or view-like operation "
                f"modeled as zero-FLOP tensor traffic ({fallback_reason})"
            ),
        )
    return WorkEstimate(
        node_id=node.node_id,
        flops=0.0,
        bytes_accessed=float(tensor_bytes),
        confidence=EstimateConfidence.UNSUPPORTED,
        rationale=(
            f"legacy fallback: unsupported operation estimate for "
            f"{node.op_type} ({fallback_reason})"
        ),
    )


def _work_estimate_from_rich_estimate(
    estimate: OperatorWorkEstimate,
    *,
    node_id: str,
) -> WorkEstimate:
    return WorkEstimate(
        node_id=node_id,
        flops=estimate.flops,
        bytes_accessed=estimate.total_bytes,
        confidence=estimate.confidence,
        rationale=estimate.rationale,
    )

def _dtype_bytes(dtype: DType) -> float:
    return {
        DType.FLOAT64: 8.0,
        DType.FLOAT32: 4.0,
        DType.FLOAT16: 2.0,
        DType.BFLOAT16: 2.0,
        DType.FLOAT8_E4M3FN: 1.0,
        DType.FLOAT8_E5M2: 1.0,
        DType.FLOAT4_E2M1: 0.5,
        DType.FLOAT4_E2M1FN_X2: 0.5,
        DType.INT64: 8.0,
        DType.INT32: 4.0,
        DType.INT16: 2.0,
        DType.INT8: 1.0,
        DType.BOOL: 1.0,
    }[dtype]


def _tensor_bytes(
    definition: Definition,
    input_shapes: dict[str, tuple[int, ...] | None],
    output_shapes: dict[str, tuple[int, ...] | None],
) -> float:
    total = 0.0
    for name, shape in input_shapes.items():
        if shape:
            total += prod(shape) * _dtype_bytes(definition.inputs[name].dtype)
    for name, shape in output_shapes.items():
        if shape:
            total += prod(shape) * _dtype_bytes(definition.outputs[name].dtype)
    return total


def _largest_reduction_dim(definition: Definition, axes: dict[str, int]) -> int:
    """Infer the common matmul K dimension from the first shaped input."""
    for spec in definition.inputs.values():
        if spec.shape:
            axis_name = spec.shape[-1]
            if axis_name.isdigit():
                return int(axis_name)
            return axes.get(axis_name, 0)
    return 0
