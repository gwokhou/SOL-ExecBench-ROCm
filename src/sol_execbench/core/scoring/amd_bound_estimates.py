# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator-level AMD bound work estimates derived from BoundGraph IR."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from sol_execbench.core.data.definition import DType
from sol_execbench.core.scoring.amd_bound_graph import (
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    OpFamily,
)
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
    return tuple(_estimate_node(graph, node) for node in graph.nodes)


def _estimate_node(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    if node.op_family == OpFamily.ATTENTION:
        return _attention_estimate(graph, node)
    if node.op_family == OpFamily.CONVOLUTION:
        return _convolution_estimate(graph, node)
    if node.op_family == OpFamily.EMBEDDING_POSITIONAL:
        return _embedding_positional_estimate(graph, node)
    if node.op_family in {OpFamily.GEMM, OpFamily.LINEAR_PROJECTION}:
        return _gemm_estimate(graph, node)
    if node.op_family == OpFamily.ELEMENTWISE:
        return _elementwise_estimate(graph, node)
    if node.op_family == OpFamily.MLP_ACTIVATION:
        return _activation_estimate(graph, node)
    if node.op_family == OpFamily.REDUCTION:
        return _reduction_estimate(graph, node)
    if node.op_family == OpFamily.NORMALIZATION:
        return _normalization_estimate(graph, node)
    if node.op_family == OpFamily.SOFTMAX:
        return _softmax_estimate(graph, node)
    if node.op_family == OpFamily.DATA_MOVEMENT:
        return _data_movement_estimate(graph, node)
    if node.op_family == OpFamily.DTYPE_CONVERSION:
        return _dtype_conversion_estimate(graph, node)
    return _unsupported_estimate(node)


def _attention_estimate(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
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
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(graph, node)
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    dims = _attention_dims_from_node_or_shapes(input_tensors, output_tensors, node)
    formula_inputs: dict[str, object] = {}
    flops = 0.0
    axis_source: str | None = None
    if dims is None:
        warnings.append("inexact_operator:attention_missing_dimensions")
        rationale_parts.append("missing attention dimensions")
        confidence = EstimateConfidence.INEXACT
    else:
        formula_inputs = dims
        flops = float(2 * dims["B"] * dims["H"] * dims["S_q"] * dims["S_k"] * dims["D"])
        confidence = EstimateConfidence.SUPPORTED if not warnings else EstimateConfidence.INEXACT
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


def _attention_softmax_estimate(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(graph, node)
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    input_elements = _sum_tensor_numel(input_tensors, "input", warnings, rationale_parts) or 0
    axis_source, axis = _axis_evidence(node)
    if axis is None:
        warnings.append("inexact_operator:attention_softmax_missing_axis")
    dims = _attention_dims_from_attributes(node)
    formula_inputs: dict[str, object] = {"input_elements": input_elements, "axis": axis}
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
        confidence=EstimateConfidence.SUPPORTED if axis is not None and not warnings else EstimateConfidence.INEXACT,
        rationale=_join_rationale(
            "attention softmax pass-count estimate over score matrix",
            rationale_parts,
        ),
        axis_source=axis_source,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _attention_scale_or_mask_estimate(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
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
        warnings.extend(("inexact_operator:attention_mask", "inexact_mask:missing_sparsity"))
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
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(graph, node)
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    dims = _attention_output_projection_dims(input_tensors, output_tensors, node)
    if dims is None:
        warnings.append("inexact_operator:attention_output_projection_missing_shape")
        formula_inputs: dict[str, object] = {}
        flops = 0.0
        axis_source = None
        confidence = EstimateConfidence.INEXACT
    else:
        formula_inputs = dims
        flops = float(2 * dims["M"] * dims["N"] * dims["K"])
        axis_source = "tensor_shapes"
        confidence = EstimateConfidence.SUPPORTED if not warnings else EstimateConfidence.INEXACT
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


def _gemm_estimate(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    input_tensors = _node_tensors(graph, node.input_tensor_ids)
    output_tensors = _node_tensors(graph, node.output_tensor_ids)
    warnings: list[str] = []
    rationale_parts: list[str] = []
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    total_bytes = read_bytes + write_bytes

    dims = _infer_gemm_dims(input_tensors, output_tensors)
    if dims is None:
        if not input_tensors and not output_tensors:
            return _unsupported_estimate(
                node,
                rationale="unsupported GEMM estimate: all key tensors are unresolved",
                warnings=("unsupported_operator:gemm_missing_tensors",),
            )
        return OperatorWorkEstimate(
            node_id=node.node_id,
            op_family=node.op_family,
            op_name=node.op_name,
            formula_kind="gemm_flops",
            formula="2*M*N*K",
            formula_inputs={},
            flops=0.0,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            intermediate_bytes=0.0,
            movement_bytes=0.0,
            total_bytes=total_bytes,
            confidence=EstimateConfidence.INEXACT,
            rationale=_join_rationale(
                "GEMM semantics recognized but missing shape evidence for M/N/K",
                rationale_parts,
            ),
            warnings=tuple(warnings or ("inexact_operator:gemm_missing_shape",)),
        )

    if "B" in dims:
        formula_kind = "batched_gemm_flops"
        formula = "2*B*M*N*K"
        flops = float(2 * dims["B"] * dims["M"] * dims["N"] * dims["K"])
    else:
        formula_kind = "gemm_flops"
        formula = "2*M*N*K"
        flops = float(2 * dims["M"] * dims["N"] * dims["K"])

    confidence = EstimateConfidence.SUPPORTED if not warnings else EstimateConfidence.INEXACT
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=node.op_family,
        op_name=node.op_name,
        formula_kind=formula_kind,
        formula=formula,
        formula_inputs=dict(dims),
        flops=flops,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=confidence,
        rationale=_join_rationale(
            "GEMM FLOPs estimated from input/output tensor shapes",
            rationale_parts,
        ),
        axis_source="tensor_shapes",
        warnings=tuple(warnings),
    )


def _convolution_estimate(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(graph, node)
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    total_bytes = read_bytes + write_bytes
    dims, missing = _convolution_dims(input_tensors, output_tensors, node)
    warnings.extend(f"inexact_operator:convolution_missing_{item}" for item in missing)
    rationale_parts.extend(f"missing convolution {item}" for item in missing)
    if dims is None:
        return OperatorWorkEstimate(
            node_id=node.node_id,
            op_family=OpFamily.CONVOLUTION,
            op_name=node.op_name,
            formula_kind="convolution_flops",
            formula="2*N*C_out*output_spatial_elements*(C_in/groups)*kernel_elements",
            formula_inputs={},
            flops=0.0,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            intermediate_bytes=0.0,
            movement_bytes=0.0,
            total_bytes=total_bytes,
            confidence=EstimateConfidence.INEXACT,
            rationale=_join_rationale(
                "convolution semantics recognized but static metadata is incomplete",
                rationale_parts,
            ),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    flops = float(
        2
        * dims["N"]
        * dims["C_out"]
        * dims["output_spatial_elements"]
        * (dims["C_in"] // dims["groups"])
        * dims["kernel_elements"]
    )
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=OpFamily.CONVOLUTION,
        op_name=node.op_name,
        formula_kind="convolution_flops",
        formula="2*N*C_out*output_spatial_elements*(C_in/groups)*kernel_elements",
        formula_inputs=dims,
        flops=flops,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=EstimateConfidence.SUPPORTED if not warnings else EstimateConfidence.INEXACT,
        rationale=_join_rationale(
            "convolution FLOPs estimated from input, weight, output, and grouping metadata",
            rationale_parts,
        ),
        axis_source="tensor_shapes",
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _embedding_positional_estimate(
    graph: BoundGraph,
    node: BoundGraphNode,
) -> OperatorWorkEstimate:
    subrole = str(node.attributes.get("memory_subrole") or "")
    if subrole in {"embedding_lookup", "gather_lookup"}:
        return _lookup_memory_estimate(graph, node, subrole=subrole)
    return _visible_memory_estimate(graph, node, subrole=subrole or "memory_bound")


def _lookup_memory_estimate(
    graph: BoundGraph,
    node: BoundGraphNode,
    *,
    subrole: str,
) -> OperatorWorkEstimate:
    warnings: list[str] = []
    rationale_parts: list[str] = []
    index_tensor = graph.tensors.get(str(node.attributes.get("index_tensor_id") or ""))
    table_tensor = graph.tensors.get(str(node.attributes.get("table_tensor_id") or ""))
    output_tensors = _node_tensors(graph, node.output_tensor_ids)
    output_tensor = output_tensors[0] if output_tensors else None
    missing: list[str] = []
    if index_tensor is None:
        missing.append("index_tensor")
    if table_tensor is None:
        missing.append("table_tensor")
    output_shape = node.attributes.get("output_shape")
    if output_tensor is None or output_tensor.shape is None or output_shape is None:
        missing.append("output_shape")
    if index_tensor is None or _dtype_bytes(index_tensor.dtype) is None:
        missing.append("index_dtype")
    if table_tensor is None or _dtype_bytes(table_tensor.dtype) is None:
        missing.append("table_dtype")

    index_elements = _tensor_numel(index_tensor) if index_tensor is not None else None
    selected_elements = (
        int(node.attributes["selected_elements"])
        if isinstance(node.attributes.get("selected_elements"), int)
        else None
    )
    if index_elements is None:
        missing.append("index_shape")
    if selected_elements is None:
        missing.append("selected_elements")
    warnings.extend(f"inexact_operator:embedding_positional_missing_{item}" for item in missing)
    rationale_parts.extend(f"missing embedding/gather {item}" for item in missing)

    index_bytes = (
        float(index_elements * _dtype_bytes(index_tensor.dtype))
        if index_tensor is not None
        and index_elements is not None
        and _dtype_bytes(index_tensor.dtype) is not None
        else 0.0
    )
    selected_read_bytes = (
        float(selected_elements * _dtype_bytes(table_tensor.dtype))
        if table_tensor is not None
        and selected_elements is not None
        and _dtype_bytes(table_tensor.dtype) is not None
        else 0.0
    )
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    read_bytes = index_bytes + selected_read_bytes
    total_bytes = read_bytes + write_bytes
    formula_inputs: dict[str, object] = {}
    confidence = EstimateConfidence.INEXACT
    axis_source: str | None = None
    if not missing:
        formula_inputs = {
            "index_elements": int(index_elements or 0),
            "selected_elements": int(selected_elements or 0),
            "index_dtype": index_tensor.dtype if index_tensor is not None else None,
            "table_dtype": table_tensor.dtype if table_tensor is not None else None,
            "memory_subrole": subrole,
        }
        confidence = EstimateConfidence.SUPPORTED if not warnings else EstimateConfidence.INEXACT
        axis_source = "tensor_shapes"
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=OpFamily.EMBEDDING_POSITIONAL,
        op_name=node.op_name,
        formula_kind="embedding_positional_bytes",
        formula="index_bytes+selected_element_bytes+output_bytes",
        formula_inputs=formula_inputs,
        flops=0.0,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=confidence,
        rationale=_join_rationale(
            "memory-bound lookup counts index bytes and selected table elements",
            rationale_parts,
        ),
        axis_source=axis_source,
        movement_kind=subrole,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _visible_memory_estimate(
    graph: BoundGraph,
    node: BoundGraphNode,
    *,
    subrole: str,
) -> OperatorWorkEstimate:
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(graph, node)
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    output_elements = _sum_tensor_numel(output_tensors, "output", warnings, rationale_parts)
    missing: list[str] = []
    if output_elements is None:
        missing.append("output_shape")
    if subrole == "rotary_like" and len(input_tensors) < 2:
        missing.append("rotary_axes")
    warnings.extend(f"inexact_operator:embedding_positional_missing_{item}" for item in missing)
    total_bytes = read_bytes + write_bytes
    formula_inputs: dict[str, object] = {}
    axis_source: str | None = None
    confidence = EstimateConfidence.INEXACT
    if not missing:
        formula_inputs = {
            "output_elements": int(output_elements or 0),
            "memory_subrole": subrole,
        }
        axis_source = "tensor_shapes"
        confidence = EstimateConfidence.SUPPORTED if not warnings else EstimateConfidence.INEXACT
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=OpFamily.EMBEDDING_POSITIONAL,
        op_name=node.op_name,
        formula_kind="embedding_positional_bytes",
        formula="read_bytes+write_bytes",
        formula_inputs=formula_inputs,
        flops=0.0,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=total_bytes,
        confidence=confidence,
        rationale=_join_rationale(
            f"{subrole} memory-bound evidence estimated from visible tensor movement",
            rationale_parts,
        ),
        axis_source=axis_source,
        movement_kind=subrole,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _elementwise_estimate(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    return _pointwise_estimate(
        graph,
        node,
        formula_kind="elementwise_flops",
        formula="output_elements",
        formula_inputs_extra={},
        rationale="elementwise work estimated as one operation per output element",
    )


def _activation_estimate(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    return _pointwise_estimate(
        graph,
        node,
        formula_kind="activation_flops",
        formula="activation_ops_per_element*output_elements",
        formula_inputs_extra={"activation_ops_per_element": 1},
        rationale="activation work conservatively estimated as one operation per output element",
    )


def _reduction_estimate(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(graph, node)
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    input_elements = _sum_tensor_numel(input_tensors, "input", warnings, rationale_parts) or 0
    axis_source, axis = _axis_evidence(node)
    formula_inputs: dict[str, object] = {"input_elements": input_elements, "axis": axis}
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


def _normalization_estimate(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(graph, node)
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    input_elements = _sum_tensor_numel(input_tensors, "input", warnings, rationale_parts) or 0
    axis_source, axis = _axis_evidence(node)
    normalization_passes = 4
    formula_inputs: dict[str, object] = {
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
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(graph, node)
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    input_elements = _sum_tensor_numel(input_tensors, "input", warnings, rationale_parts) or 0
    axis_source, axis = _axis_evidence(node)
    softmax_passes = 5
    formula_inputs: dict[str, object] = {
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


def _data_movement_estimate(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(graph, node)
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    movement_kind = str(node.attributes.get("movement_kind") or _movement_kind_from_op_name(node))
    if movement_kind == "materialized":
        movement_bytes = read_bytes + write_bytes
        rationale = "materialized data movement estimate for contiguous or copy-like operation"
    elif movement_kind == "broadcast_view":
        movement_bytes = 0.0
        rationale = "broadcast view evidence with zero movement bytes"
    else:
        movement_kind = "logical_view"
        movement_bytes = 0.0
        rationale = "logical view evidence with zero movement bytes"
    total_bytes = read_bytes + write_bytes + movement_bytes
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


def _dtype_conversion_estimate(graph: BoundGraph, node: BoundGraphNode) -> OperatorWorkEstimate:
    input_tensors, output_tensors, warnings, rationale_parts = _estimate_tensors(graph, node)
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    target_dtype = node.attributes.get("target_dtype") or _first_tensor_dtype(output_tensors)
    if target_dtype is None or _dtype_bytes(str(target_dtype)) is None:
        warnings.append("inexact_dtype_conversion:missing_target_dtype")
        rationale_parts.append("missing target dtype for dtype conversion")
    movement_bytes = read_bytes + write_bytes
    total_bytes = read_bytes + write_bytes + movement_bytes
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
        rationale=_join_rationale("dtype conversion movement estimate", rationale_parts),
        movement_kind="dtype_conversion",
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _pointwise_estimate(
    graph: BoundGraph,
    node: BoundGraphNode,
    *,
    formula_kind: str,
    formula: str,
    formula_inputs_extra: dict[str, object],
    rationale: str,
) -> OperatorWorkEstimate:
    input_tensors = _node_tensors(graph, node.input_tensor_ids)
    output_tensors = _node_tensors(graph, node.output_tensor_ids)
    warnings: list[str] = []
    rationale_parts: list[str] = []
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    output_elements = _sum_tensor_numel(output_tensors, "output", warnings, rationale_parts)
    total_bytes = read_bytes + write_bytes
    if output_elements is None:
        if not input_tensors and not output_tensors:
            return _unsupported_estimate(
                node,
                rationale=f"unsupported {node.op_family.value} estimate: all key tensors are unresolved",
                warnings=(f"unsupported_operator:{node.op_family.value}_missing_tensors",),
            )
        output_elements = 0
        warnings.append(f"inexact_operator:{node.op_family.value}_missing_shape")

    formula_inputs = {"output_elements": output_elements}
    formula_inputs.update(formula_inputs_extra)
    flops = float(output_elements)
    if "activation_ops_per_element" in formula_inputs:
        flops *= float(formula_inputs["activation_ops_per_element"])

    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=node.op_family,
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
        confidence=EstimateConfidence.INEXACT,
        rationale=_join_rationale(rationale, rationale_parts),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _unsupported_estimate(
    node: BoundGraphNode,
    *,
    rationale: str | None = None,
    warnings: tuple[str, ...] | None = None,
) -> OperatorWorkEstimate:
    warning_kind = (
        "unsupported_operator"
        if node.op_family == OpFamily.UNSUPPORTED
        else "unsupported_family"
    )
    warning = f"{warning_kind}:{node.op_name or node.op_family.value}"
    estimate_warnings = warnings or (warning,)
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
        rationale=rationale or (
            f"unsupported operation estimate for {node.op_family.value}: "
            f"{node.op_name or node.source_expression}"
        ),
        warnings=estimate_warnings,
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


def _node_tensors(graph: BoundGraph, tensor_ids: tuple[str, ...]) -> tuple[BoundTensor, ...]:
    return tuple(graph.tensors[tensor_id] for tensor_id in tensor_ids if tensor_id in graph.tensors)


def _sum_tensor_bytes(
    tensors: tuple[BoundTensor, ...],
    bucket: str,
    warnings: list[str],
    rationale_parts: list[str],
) -> float:
    total = 0.0
    for tensor in tensors:
        tensor_bytes = _tensor_bytes(tensor)
        if tensor_bytes is None:
            if tensor.shape is None:
                rationale_parts.append(f"missing shape for {bucket} tensor {tensor.tensor_id}")
                warnings.append(f"inexact_bytes:missing_shape:{tensor.tensor_id}")
            if _dtype_bytes(tensor.dtype) is None:
                rationale_parts.append(f"missing dtype for {bucket} tensor {tensor.tensor_id}")
                warnings.append(f"inexact_bytes:missing_dtype:{tensor.tensor_id}")
            continue
        total += tensor_bytes
    return total


def _sum_tensor_numel(
    tensors: tuple[BoundTensor, ...],
    bucket: str,
    warnings: list[str],
    rationale_parts: list[str],
) -> int | None:
    if not tensors:
        rationale_parts.append(f"missing {bucket} tensor metadata")
        warnings.append(f"inexact_elements:missing_{bucket}_tensor")
        return None
    total = 0
    for tensor in tensors:
        numel = _tensor_numel(tensor)
        if numel is None:
            rationale_parts.append(f"missing shape for {bucket} tensor {tensor.tensor_id}")
            warnings.append(f"inexact_elements:missing_shape:{tensor.tensor_id}")
            return None
        total += numel
    return total


def _estimate_tensors(
    graph: BoundGraph,
    node: BoundGraphNode,
) -> tuple[tuple[BoundTensor, ...], tuple[BoundTensor, ...], list[str], list[str]]:
    return (
        _node_tensors(graph, node.input_tensor_ids),
        _node_tensors(graph, node.output_tensor_ids),
        [],
        [],
    )


def _axis_evidence(node: BoundGraphNode) -> tuple[str, object]:
    if "dim" in node.attributes:
        return str(node.attributes.get("axis_source") or "attribute"), node.attributes["dim"]
    if "axis" in node.attributes:
        return str(node.attributes.get("axis_source") or "attribute"), node.attributes["axis"]
    return "missing", None


def _movement_kind_from_op_name(node: BoundGraphNode) -> str:
    leaf_name = node.op_name.rsplit(".", maxsplit=1)[-1]
    if leaf_name in {"expand", "broadcast_to"}:
        return "broadcast_view"
    if leaf_name == "contiguous":
        return "materialized"
    return "logical_view"


def _first_tensor_dtype(tensors: tuple[BoundTensor, ...]) -> str | None:
    for tensor in tensors:
        if tensor.dtype and tensor.dtype != "unknown":
            return tensor.dtype
    return None


def _infer_gemm_dims(
    input_tensors: tuple[BoundTensor, ...],
    output_tensors: tuple[BoundTensor, ...],
) -> dict[str, int] | None:
    if len(input_tensors) < 2 or not output_tensors:
        return None
    lhs_shape = input_tensors[0].shape
    rhs_shape = input_tensors[1].shape
    out_shape = output_tensors[0].shape
    if lhs_shape is None or rhs_shape is None or out_shape is None:
        return None
    if len(lhs_shape) == 2 and len(rhs_shape) == 2 and len(out_shape) >= 2:
        return {"M": int(lhs_shape[-2]), "N": int(out_shape[-1]), "K": int(lhs_shape[-1])}
    if len(lhs_shape) >= 3 and len(rhs_shape) == 2 and len(out_shape) >= 3:
        batch = prod(out_shape[:-2])
        return {
            "B": int(batch),
            "M": int(out_shape[-2]),
            "N": int(out_shape[-1]),
            "K": int(lhs_shape[-1]),
        }
    if len(lhs_shape) >= 3 and len(rhs_shape) >= 3 and len(out_shape) >= 3:
        batch_dims = out_shape[:-2]
        batch = prod(batch_dims)
        return {
            "B": int(batch),
            "M": int(out_shape[-2]),
            "N": int(out_shape[-1]),
            "K": int(lhs_shape[-1]),
        }
    return None


def _convolution_dims(
    input_tensors: tuple[BoundTensor, ...],
    output_tensors: tuple[BoundTensor, ...],
    node: BoundGraphNode,
) -> tuple[dict[str, int] | None, list[str]]:
    missing: list[str] = []
    dimensionality = node.attributes.get("dimensionality")
    if not isinstance(dimensionality, int) or dimensionality not in {1, 2, 3}:
        missing.append("dimensionality")
    for key in ("stride", "padding", "dilation", "groups", "output_spatial"):
        if key not in node.attributes:
            missing.append(key)
    if len(input_tensors) < 2:
        missing.append("input_or_weight")
        return None, missing
    if not output_tensors:
        missing.append("output")
        return None, missing
    input_shape = input_tensors[0].shape
    weight_shape = input_tensors[1].shape
    output_shape = output_tensors[0].shape
    if input_shape is None:
        missing.append("input_shape")
    if weight_shape is None:
        missing.append("kernel_shape")
    if output_shape is None:
        missing.append("output_shape")
    groups = node.attributes.get("groups")
    if not isinstance(groups, int) or groups <= 0:
        missing.append("groups")
    output_spatial = node.attributes.get("output_spatial")
    if not isinstance(output_spatial, tuple):
        missing.append("output_spatial")
    if missing:
        return None, list(dict.fromkeys(missing))
    assert isinstance(dimensionality, int)
    assert isinstance(groups, int)
    assert isinstance(output_spatial, tuple)
    assert input_shape is not None and weight_shape is not None and output_shape is not None
    if (
        len(input_shape) != dimensionality + 2
        or len(weight_shape) != dimensionality + 2
        or len(output_shape) != dimensionality + 2
        or len(output_spatial) != dimensionality
    ):
        return None, list(dict.fromkeys([*missing, "dimensionality_shape_match"]))
    batch = int(input_shape[0])
    input_channels = int(input_shape[1])
    output_channels = int(output_shape[1])
    kernel_elements = int(prod(weight_shape[2:]))
    if input_channels % groups != 0:
        return None, list(dict.fromkeys([*missing, "groups"]))
    if int(weight_shape[0]) != output_channels:
        return None, list(dict.fromkeys([*missing, "kernel_shape"]))
    expected_channels_per_group = input_channels // groups
    if int(weight_shape[1]) != expected_channels_per_group:
        return None, list(dict.fromkeys([*missing, "groups"]))
    return {
        "N": batch,
        "C_in": input_channels,
        "C_out": output_channels,
        "groups": groups,
        "output_spatial_elements": int(prod(output_spatial)),
        "kernel_elements": kernel_elements,
        "dimensionality": dimensionality,
    }, []


def _attention_dims_from_node_or_shapes(
    input_tensors: tuple[BoundTensor, ...],
    output_tensors: tuple[BoundTensor, ...],
    node: BoundGraphNode,
) -> dict[str, int] | None:
    dims = _attention_dims_from_attributes(node)
    if dims is not None:
        return dims
    if len(input_tensors) < 2 or not output_tensors:
        return None
    lhs_shape = input_tensors[0].shape
    rhs_shape = input_tensors[1].shape
    out_shape = output_tensors[0].shape
    if lhs_shape is None or rhs_shape is None or out_shape is None:
        return None
    if len(lhs_shape) != 4 or len(rhs_shape) != 4 or len(out_shape) != 4:
        return None
    subrole = str(node.attributes.get("subrole") or "")
    if subrole == "qk_scores":
        return {
            "B": int(lhs_shape[0]),
            "H": int(lhs_shape[1]),
            "S_q": int(lhs_shape[2]),
            "S_k": int(rhs_shape[3]),
            "D": int(lhs_shape[3]),
        }
    if subrole == "pv_aggregation":
        return {
            "B": int(out_shape[0]),
            "H": int(out_shape[1]),
            "S_q": int(out_shape[2]),
            "S_k": int(rhs_shape[2]),
            "D": int(out_shape[3]),
        }
    return None


def _attention_dims_from_attributes(node: BoundGraphNode) -> dict[str, int] | None:
    keys = {
        "B": "batch",
        "H": "heads",
        "S_q": "sequence_q",
        "S_k": "sequence_k",
        "D": "head_dim",
    }
    result: dict[str, int] = {}
    for formula_name, attribute_name in keys.items():
        value = node.attributes.get(attribute_name)
        if not isinstance(value, int):
            return None
        result[formula_name] = value
    return result


def _attention_output_projection_dims(
    input_tensors: tuple[BoundTensor, ...],
    output_tensors: tuple[BoundTensor, ...],
    node: BoundGraphNode,
) -> dict[str, int] | None:
    if len(input_tensors) < 2 or not output_tensors:
        return None
    lhs_shape = input_tensors[0].shape
    rhs_shape = input_tensors[1].shape
    out_shape = output_tensors[0].shape
    if lhs_shape is None or rhs_shape is None or out_shape is None:
        return None
    if len(lhs_shape) == 4 and len(rhs_shape) == 2 and len(out_shape) == 4:
        return {
            "M": int(lhs_shape[0] * lhs_shape[1] * lhs_shape[2]),
            "N": int(out_shape[-1]),
            "K": int(lhs_shape[-1]),
        }
    dims = _attention_dims_from_attributes(node)
    if dims is not None:
        return {
            "M": int(dims["B"] * dims["H"] * dims["S_q"]),
            "N": int(dims["D"]),
            "K": int(dims["D"]),
        }
    return _infer_gemm_dims(input_tensors, output_tensors)


def _join_rationale(primary: str, details: list[str]) -> str:
    if not details:
        return primary
    return f"{primary}; {'; '.join(dict.fromkeys(details))}"
