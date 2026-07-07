# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for AMD bound work estimators."""

from __future__ import annotations

from math import prod
from typing import Any

from sol_execbench.core.data.definition import DType
from sol_execbench.core.scoring.amd_bound_estimate_models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph_models import (
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    OpFamily,
)
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence

def _pointwise_estimate(
    graph: BoundGraph,
    node: BoundGraphNode,
    *,
    formula_kind: str,
    formula: str,
    formula_inputs_extra: dict[str, Any],
    rationale: str,
) -> OperatorWorkEstimate:
    input_tensors = _node_tensors(graph, node.input_tensor_ids)
    output_tensors = _node_tensors(graph, node.output_tensor_ids)
    warnings: list[str] = []
    rationale_parts: list[str] = []
    read_bytes = _sum_tensor_bytes(input_tensors, "read", warnings, rationale_parts)
    write_bytes = _sum_tensor_bytes(output_tensors, "write", warnings, rationale_parts)
    output_elements = _sum_tensor_numel(
        output_tensors, "output", warnings, rationale_parts
    )
    total_bytes = read_bytes + write_bytes
    if output_elements is None:
        if not input_tensors and not output_tensors:
            return _unsupported_estimate(
                node,
                rationale=f"unsupported {node.op_family.value} estimate: all key tensors are unresolved",
                warnings=(
                    f"unsupported_operator:{node.op_family.value}_missing_tensors",
                ),
            )
        output_elements = 0
        warnings.append(f"inexact_operator:{node.op_family.value}_missing_shape")

    formula_inputs: dict[str, Any] = {"output_elements": output_elements}
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
        rationale=rationale
        or (
            f"unsupported operation estimate for {node.op_family.value}: "
            f"{node.op_name or node.source_expression}"
        ),
        warnings=estimate_warnings,
    )


_DTYPE_BYTES = {
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
}


def _dtype_bytes(dtype: DType | str) -> float | None:
    raw = dtype.value if isinstance(dtype, DType) else str(dtype)
    return _DTYPE_BYTES.get(raw)


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


def _node_tensors(
    graph: BoundGraph, tensor_ids: tuple[str, ...]
) -> tuple[BoundTensor, ...]:
    return tuple(
        graph.tensors[tensor_id]
        for tensor_id in tensor_ids
        if tensor_id in graph.tensors
    )


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
                rationale_parts.append(
                    f"missing shape for {bucket} tensor {tensor.tensor_id}"
                )
                warnings.append(f"inexact_bytes:missing_shape:{tensor.tensor_id}")
            if _dtype_bytes(tensor.dtype) is None:
                rationale_parts.append(
                    f"missing dtype for {bucket} tensor {tensor.tensor_id}"
                )
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
            rationale_parts.append(
                f"missing shape for {bucket} tensor {tensor.tensor_id}"
            )
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
        return str(node.attributes.get("axis_source") or "attribute"), node.attributes[
            "dim"
        ]
    if "axis" in node.attributes:
        return str(node.attributes.get("axis_source") or "attribute"), node.attributes[
            "axis"
        ]
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
        return {
            "M": int(lhs_shape[-2]),
            "N": int(out_shape[-1]),
            "K": int(lhs_shape[-1]),
        }
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
    assert (
        input_shape is not None
        and weight_shape is not None
        and output_shape is not None
    )
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
