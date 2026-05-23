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
    if node.op_family in {OpFamily.GEMM, OpFamily.LINEAR_PROJECTION}:
        return _gemm_estimate(graph, node)
    if node.op_family == OpFamily.ELEMENTWISE:
        return _elementwise_estimate(graph, node)
    if node.op_family == OpFamily.MLP_ACTIVATION:
        return _activation_estimate(graph, node)
    return _unsupported_estimate(node)


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
        warnings=tuple(warnings),
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


def _join_rationale(primary: str, details: list[str]) -> str:
    if not details:
        return primary
    return f"{primary}; {'; '.join(dict.fromkeys(details))}"
