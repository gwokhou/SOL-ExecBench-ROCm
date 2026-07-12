# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for AMD bound graph extraction."""

from __future__ import annotations

import ast
from typing import Any

from sol_execbench.core.data.definition import DType
from sol_execbench.core.scoring.amd_bound_estimate.classification import (
    CallClassification as _CallClassification,
    dtype_method_target as _dtype_method_target,
    movement_kind_for_name as _movement_kind_for_name,
)
from sol_execbench.core.scoring.amd_bound_graph.enums import OpFamily


def _fx_tensor_meta(node: Any) -> tuple[tuple[int, ...] | None, str | None]:
    meta = node.meta.get("tensor_meta") if hasattr(node, "meta") else None
    if meta is None:
        return None, None
    shape = (
        tuple(int(dim) for dim in meta.shape)
        if getattr(meta, "shape", None) is not None
        else None
    )
    dtype = (
        str(meta.dtype).removeprefix("torch.")
        if getattr(meta, "dtype", None) is not None
        else None
    )
    return shape, dtype


_MISSING = object()


def _classification_family(classification: _CallClassification) -> OpFamily:
    return OpFamily(classification.op_family)


def _axis_from_values(args: tuple[Any, ...], kwargs: dict[str, Any]) -> object:
    for name in ("dim", "axis"):
        if name in kwargs:
            return _literal_value(kwargs[name])
    for arg in args:
        value = _literal_value(arg)
        if value is not _MISSING:
            return value
    return _MISSING


def _target_dtype_from_values(
    leaf_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str | None:
    method_target = _dtype_method_target(leaf_name)
    if method_target is not None:
        return method_target
    if leaf_name not in {"to", "type"}:
        return None
    if "dtype" in kwargs:
        dtype = _normalize_dtype_value(kwargs["dtype"])
        if dtype is not None:
            return dtype
    for arg in args:
        dtype = _normalize_dtype_value(arg)
        if dtype is not None:
            return dtype
    return None


def _ast_call_attributes(
    node: ast.Call,
    func_name: str,
    classification: _CallClassification,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    leaf_name = func_name.rsplit(".", maxsplit=1)[-1]
    movement_kind = _movement_kind_for_name(leaf_name)
    if movement_kind is not None:
        attributes["movement_kind"] = movement_kind
    if leaf_name in {"zeros", "zeros_like"}:
        attributes["materialization_kind"] = "fill"

    keyword_values = {
        keyword.arg: keyword.value for keyword in node.keywords if keyword.arg
    }
    op_family = _classification_family(classification)
    if op_family in {
        OpFamily.REDUCTION,
        OpFamily.NORMALIZATION,
        OpFamily.SOFTMAX,
    }:
        positional_args = tuple(
            node.args if isinstance(node.func, ast.Attribute) else node.args[1:]
        )
        axis = _axis_from_values(positional_args, keyword_values)
        if axis is not _MISSING:
            attributes["dim"] = axis
            attributes["axis_source"] = "attribute"
        if op_family == OpFamily.REDUCTION and "keepdim" in keyword_values:
            keepdim = _literal_value(keyword_values["keepdim"])
            if isinstance(keepdim, bool):
                attributes["keepdim"] = keepdim

    if leaf_name == "pow":
        exponent_position = 0 if isinstance(node.func, ast.Attribute) else 1
        exponent = _arg_literal(tuple(node.args), exponent_position)
        if isinstance(exponent, int | float):
            attributes["exponent"] = exponent

    target_dtype = _target_dtype_from_values(
        leaf_name,
        tuple(node.args if isinstance(node.func, ast.Attribute) else node.args[1:]),
        keyword_values,
    )
    if target_dtype is not None:
        attributes["target_dtype"] = target_dtype
    args = tuple(node.args)
    if op_family == OpFamily.CONVOLUTION:
        attributes.update(
            _convolution_attributes(leaf_name, args, keyword_values, None)
        )
    if op_family == OpFamily.EMBEDDING_POSITIONAL:
        attributes.update(
            _memory_bound_call_attributes(leaf_name, args, keyword_values, None)
        )
    if op_family == OpFamily.MOE:
        attributes.update(_moe_call_attributes(leaf_name, args, keyword_values))
    if op_family == OpFamily.SSM_MAMBA:
        attributes.update(_ssm_mamba_call_attributes(leaf_name))
    return attributes


def _ssm_mamba_call_attributes(leaf_name: str) -> dict[str, Any]:
    if leaf_name.lower() in {"selective_scan", "mamba_scan", "ssm_scan"}:
        return {"subrole": "scan", "recognized_scan": True}
    return {}


def _moe_call_attributes(
    leaf_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    attrs: dict[str, object] = {}
    normalized = leaf_name.lower()
    if normalized == "router":
        attrs["subrole"] = "router"
    elif normalized in {"topk", "top_k"}:
        attrs["subrole"] = "top_k"
        value = _literal_value(kwargs["k"]) if "k" in kwargs else _arg_literal(args, 1)
        if isinstance(value, int):
            attrs["route_top_k"] = int(value)
            attrs["route_cardinality_source"] = "topk.k"
    elif normalized in {"dispatch_and_combine", "dispatch_dynamic"}:
        attrs["subrole"] = "dispatch"
    return attrs


def _convolution_attributes(
    leaf_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    node: Any | None,
) -> dict[str, Any]:
    dimensionality = {"conv1d": 1, "conv2d": 2, "conv3d": 3}.get(leaf_name)
    if dimensionality is None:
        return {}
    output_shape = _fx_tensor_meta(node)[0] if node is not None else None
    attrs: dict[str, object] = {"dimensionality": dimensionality}
    for name, position, default in (
        ("stride", 3, 1),
        ("padding", 4, 0),
        ("dilation", 5, 1),
        ("groups", 6, 1),
    ):
        provided = name in kwargs or position < len(args)
        value = (
            _literal_value(kwargs[name])
            if name in kwargs
            else _arg_literal(args, position)
        )
        if value is _MISSING and not provided:
            value = default
        if value is not _MISSING:
            attrs[name] = (
                int(value)
                if name == "groups" and isinstance(value, int)
                else _normalize_spatial_tuple(value, dimensionality)
            )
    if output_shape is not None and len(output_shape) >= dimensionality + 2:
        attrs["output_spatial"] = tuple(
            int(dim) for dim in output_shape[-dimensionality:]
        )
    return attrs


def _memory_bound_call_attributes(
    leaf_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    node: Any | None,
) -> dict[str, Any]:
    output_shape = _fx_tensor_meta(node)[0] if node is not None else None
    subrole = "embedding_lookup" if leaf_name == "embedding" else "gather_lookup"
    attrs: dict[str, object] = {"memory_subrole": subrole}
    if output_shape is not None:
        attrs["output_shape"] = output_shape
        attrs["selected_elements"] = int(_shape_numel(output_shape))
    dim = _literal_value(kwargs["dim"]) if "dim" in kwargs else _arg_literal(args, 1)
    if dim is not _MISSING and isinstance(dim, int):
        attrs["dim"] = dim
        attrs["axis_source"] = "attribute"
    return attrs


def _arg_literal(args: tuple[Any, ...], index: int) -> object:
    if index >= len(args):
        return _MISSING
    return _literal_value(args[index])


def _normalize_spatial_tuple(
    value: object, dimensionality: int
) -> tuple[int, ...] | object:
    if isinstance(value, int):
        return tuple(value for _ in range(dimensionality))
    if isinstance(value, tuple) and len(value) == dimensionality:
        normalized: list[int] = []
        for item in value:
            if not isinstance(item, int):
                return value
            normalized.append(item)
        return tuple(normalized)
    return value


def _shape_numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _literal_value(value: Any) -> object:
    if isinstance(value, ast.Constant) and isinstance(value.value, (int, type(None))):
        return value.value
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
        operand = _literal_value(value.operand)
        if isinstance(operand, int):
            return -operand
    if isinstance(value, ast.Tuple | ast.List):
        elements = tuple(_literal_value(element) for element in value.elts)
        if all(element is None or isinstance(element, int) for element in elements):
            return elements
        return _MISSING
    if value is None or isinstance(value, int):
        return value
    if isinstance(value, tuple | list) and all(
        item is None or isinstance(item, int) for item in value
    ):
        return tuple(value)
    return _MISSING


def _normalize_dtype_value(value: Any) -> str | None:
    if isinstance(value, ast.Attribute):
        return _normalize_dtype_name(value.attr)
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return _normalize_dtype_name(value.value)
    return _normalize_dtype_name(str(value).removeprefix("torch."))


def _normalize_dtype_name(raw: str) -> str | None:
    name = raw.removeprefix("torch.").lower()
    aliases = {
        "float": DType.FLOAT32.value,
        "float32": DType.FLOAT32.value,
        "half": DType.FLOAT16.value,
        "float16": DType.FLOAT16.value,
        "bfloat16": DType.BFLOAT16.value,
        "double": DType.FLOAT64.value,
        "float64": DType.FLOAT64.value,
        "bool": DType.BOOL.value,
        "int": DType.INT32.value,
        "int32": DType.INT32.value,
        "long": DType.INT64.value,
        "int64": DType.INT64.value,
        "int16": DType.INT16.value,
        "int8": DType.INT8.value,
    }
    return aliases.get(name)
