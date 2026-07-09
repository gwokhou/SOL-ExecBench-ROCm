# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for converting torch.fx nodes to bound graph evidence."""

from __future__ import annotations

import operator
from typing import Any

from sol_execbench.core.data.definition import DType, Definition
from sol_execbench.core.scoring.amd_bound_estimate.classification import (
    CallClassification as _CallClassification,
    classify_call as _classify_call,
    movement_kind_for_name as _movement_kind_for_name,
)
from sol_execbench.core.scoring.amd_bound_graph.common import (
    _MISSING,
    _axis_from_values,
    _classification_family,
    _convolution_attributes,
    _memory_bound_call_attributes,
    _moe_call_attributes,
    _ssm_mamba_call_attributes,
    _target_dtype_from_values,
    _fx_tensor_meta,
)
from sol_execbench.core.scoring.amd_bound_graph.models import (
    BoundTensor,
    OpFamily,
)
from sol_execbench.core.scoring.confidence import EstimateConfidence

__all__ = [
    "_classification_family",
    "_classify_fx_node",
    "_first_input_dtype",
    "_first_input_shape",
    "_flatten_fx_output_tensor_ids",
    "_fx_input_tensor_ids",
    "_fx_node_attributes",
    "_fx_node_name",
    "_fx_source_expression",
    "_fx_tensor_meta",
    "_torch_dtype",
]


def _classify_fx_node(node: Any) -> tuple[str, _CallClassification, str | None]:
    func_name = _fx_node_name(node)
    classification = _classify_call(func_name)
    if node.op == "call_function" and node.target is operator.matmul:
        return (
            "@",
            _CallClassification(
                OpFamily.GEMM.value,
                EstimateConfidence.SUPPORTED,
                "recognized traced matrix multiply",
            ),
            None,
        )
    if node.op == "call_function" and node.target in {
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.pow,
    }:
        return (
            func_name,
            _CallClassification(
                OpFamily.ELEMENTWISE.value,
                EstimateConfidence.INEXACT,
                "recognized traced elementwise operation",
            ),
            None,
        )
    if classification is not None:
        return func_name, classification, None
    return (
        func_name,
        _CallClassification(
            OpFamily.UNSUPPORTED.value,
            EstimateConfidence.UNSUPPORTED,
            "unsupported traced operation preserved as graph evidence",
        ),
        f"unsupported_operator:{func_name or '<unknown>'}",
    )


def _torch_dtype(torch: Any, dtype: DType) -> Any:
    return {
        DType.FLOAT64: torch.float64,
        DType.FLOAT32: torch.float32,
        DType.FLOAT16: torch.float16,
        DType.BFLOAT16: torch.bfloat16,
        DType.INT64: torch.int64,
        DType.INT32: torch.int32,
        DType.INT16: torch.int16,
        DType.INT8: torch.int8,
        DType.BOOL: torch.bool,
        DType.FLOAT8_E4M3FN: torch.float16,
        DType.FLOAT8_E5M2: torch.float16,
        DType.FLOAT4_E2M1: torch.float16,
        DType.FLOAT4_E2M1FN_X2: torch.float16,
    }[dtype]


def _fx_node_name(node: Any) -> str:
    target = node.target
    if isinstance(target, str):
        return target
    if hasattr(target, "__module__") and hasattr(target, "__name__"):
        module = target.__module__
        name = target.__name__
        if module == "_operator":
            return name
        if module == "torch._C._linalg" and name.startswith("linalg_"):
            return f"torch.linalg.{name.removeprefix('linalg_')}"
        return f"{module}.{name}"
    return str(target)


def _fx_input_tensor_ids(
    node: Any,
    node_outputs: dict[Any, str],
    definition: Definition,
) -> tuple[str, ...]:
    result: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, (tuple, list)):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif value in node_outputs:
            result.append(node_outputs[value])
        elif isinstance(value, str) and value in definition.inputs:
            result.append(f"input:{value}")

    collect(node.args)
    collect(node.kwargs)
    return tuple(result)


def _flatten_fx_output_tensor_ids(
    value: Any, node_outputs: dict[Any, str]
) -> tuple[str, ...]:
    result: list[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, (tuple, list)):
            for nested in item:
                collect(nested)
        elif item in node_outputs:
            result.append(node_outputs[item])

    collect(value)
    return tuple(result)


def _fx_source_expression(node: Any) -> str:
    if (
        node.op == "call_function"
        and node.target is operator.matmul
        and len(node.args) >= 2
    ):
        return f"{node.args[0]} @ {node.args[1]}"
    func_name = _fx_node_name(node)
    args = ", ".join(str(arg) for arg in node.args)
    return f"{func_name}({args})"


def _fx_node_attributes(
    node: Any,
    func_name: str,
    classification: _CallClassification,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    leaf_name = func_name.rsplit(".", maxsplit=1)[-1]
    movement_kind = _movement_kind_for_name(leaf_name)
    if movement_kind is not None:
        attributes["movement_kind"] = movement_kind

    op_family = _classification_family(classification)
    if op_family in {
        OpFamily.REDUCTION,
        OpFamily.NORMALIZATION,
        OpFamily.SOFTMAX,
    }:
        axis = _axis_from_values(node.args[1:], node.kwargs)
        if axis is not _MISSING:
            attributes["dim"] = axis
            attributes["axis_source"] = "attribute"

    target_dtype = _target_dtype_from_values(leaf_name, node.args[1:], node.kwargs)
    if target_dtype is not None:
        attributes["target_dtype"] = target_dtype
    if op_family == OpFamily.CONVOLUTION:
        attributes.update(
            _convolution_attributes(leaf_name, node.args, node.kwargs, node)
        )
    if op_family == OpFamily.EMBEDDING_POSITIONAL:
        attributes.update(
            _memory_bound_call_attributes(leaf_name, node.args, node.kwargs, node)
        )
    if op_family == OpFamily.MOE:
        attributes.update(_moe_call_attributes(leaf_name, node.args, node.kwargs))
    if op_family == OpFamily.SSM_MAMBA:
        attributes.update(_ssm_mamba_call_attributes(leaf_name))
    return attributes


def _first_input_shape(
    input_tensor_ids: tuple[str, ...],
    tensors: dict[str, BoundTensor],
    output_shapes: dict[str, tuple[int, ...] | None],
) -> tuple[int, ...] | None:
    for tensor_id in input_tensor_ids:
        tensor = tensors.get(tensor_id)
        if tensor and tensor.shape is not None:
            return tensor.shape
    for shape in output_shapes.values():
        if shape is not None:
            return shape
    return None


def _first_input_dtype(
    input_tensor_ids: tuple[str, ...],
    tensors: dict[str, BoundTensor],
    definition: Definition,
) -> str:
    for tensor_id in input_tensor_ids:
        tensor = tensors.get(tensor_id)
        if tensor:
            return tensor.dtype
    if definition.outputs:
        return next(iter(definition.outputs.values())).dtype.value
    return "unknown"
