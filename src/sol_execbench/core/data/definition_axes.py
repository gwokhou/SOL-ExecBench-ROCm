# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Axis, shape, and dtype helpers for workload definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional

from .definition_models import AxisConst, AxisExpr, AxisVar, TensorSpec
from .dtypes import dtype_str_to_torch_dtype
from .shapes import resolve_shape_expression

if TYPE_CHECKING:
    import torch


def const_axes(definition: Any) -> dict[str, int]:
    """Return constant axis names and values."""
    return {
        name: axis.value
        for name, axis in definition.axes.items()
        if isinstance(axis, AxisConst)
    }


def var_axes(definition: Any) -> list[str]:
    """Return variable axis names."""
    return [
        name for name, axis in definition.axes.items() if isinstance(axis, AxisVar)
    ]


def expr_axes(definition: Any) -> dict[str, AxisExpr]:
    """Return expression axis names and specs."""
    return {
        name: axis
        for name, axis in definition.axes.items()
        if isinstance(axis, AxisExpr)
    }


def get_axes_values(
    definition: Any, input_shapes: Iterable[Optional[tuple[int, ...]]]
) -> dict[str, int]:
    """Get concrete variable axis values from input shapes."""
    var_axes_values: dict[str, int] = {}
    for (inp_name, inp_spec), inp_shape in zip(definition.inputs.items(), input_shapes):
        if inp_spec.shape is None:
            continue
        if inp_shape is None:
            raise ValueError(f"Input '{inp_name}' expected shaped tensor, got scalar")
        if len(inp_spec.shape) != len(inp_shape):
            raise ValueError(
                f"Input '{inp_name}''s defined dimension is {len(inp_spec.shape)} but the "
                f"actual dimension is {len(inp_shape)}"
            )
        for axis_name, axis_value in zip(inp_spec.shape, inp_shape):
            if axis_name in definition.axes and definition.axes[axis_name].type == "var":
                if axis_name in var_axes_values:
                    if var_axes_values[axis_name] != axis_value:
                        raise ValueError(
                            f"Axis '{axis_name}' has different values for different input "
                            f"tensors: {var_axes_values[axis_name]} and {axis_value}"
                        )
                else:
                    var_axes_values[axis_name] = axis_value

    if len(var_axes_values) != len(definition.var_axes):
        raise ValueError(
            f"Missing values for variable axes: "
            f"{set(definition.var_axes) - set(var_axes_values.keys())}"
        )
    return var_axes_values


def get_axes_values_from_inputs(
    definition: Any, inputs: Iterable[Any]
) -> dict[str, int]:
    """Get concrete variable axis values directly from input values."""
    shapes = [tuple(val.shape) if hasattr(val, "shape") else None for val in inputs]
    return get_axes_values(definition, shapes)


def get_resolved_axes_values(
    definition: Any, var_axes_values: dict[str, int]
) -> dict[str, int]:
    """Resolve constants, variables, and expression axes into concrete values."""
    resolved_axes_values: dict[str, int] = definition.const_axes.copy()

    for name, axis_value in var_axes_values.items():
        resolved_axes_values[name] = axis_value

    for name, axis in definition.expr_axes.items():
        resolved_axes_values[name] = resolve_shape_expression(
            axis.expression, resolved_axes_values
        )
    return resolved_axes_values


def get_shapes(
    definition: Any,
    tensors: Iterable[TensorSpec],
    var_axes_values: Optional[dict[str, int]] = None,
) -> list[Optional[tuple[int, ...]]]:
    """Get concrete tensor shapes given variable axis values."""
    var_axes_values = var_axes_values or {}
    shapes = []

    resolved_axes = get_resolved_axes_values(definition, var_axes_values)

    for tensor_spec in tensors:
        if tensor_spec.shape is None:
            shapes.append(None)
            continue
        shape = []
        for axis_name in tensor_spec.shape:
            if axis_name.isdigit():
                value = int(axis_name)
            else:
                value = resolved_axes[axis_name]
            shape.append(value)
        shapes.append(tuple(shape))

    return shapes


def get_input_shapes(
    definition: Any, var_axes_values: Optional[dict[str, int]] = None
) -> dict[str, Optional[tuple[int, ...]]]:
    """Get concrete input shapes given variable axis values."""
    shapes = get_shapes(definition, definition.inputs.values(), var_axes_values)
    return dict(zip(definition.inputs.keys(), shapes))


def get_output_shapes(
    definition: Any, var_values: Optional[dict[str, int]] = None
) -> dict[str, Optional[tuple[int, ...]]]:
    """Get concrete output shapes given variable axis values."""
    shapes = get_shapes(definition, definition.outputs.values(), var_values)
    return dict(zip(definition.outputs.keys(), shapes))


def torch_input_dtypes(definition: Any) -> list[torch.dtype]:
    """Get the torch data types of the input tensors."""
    return [dtype_str_to_torch_dtype(spec.dtype) for spec in definition.inputs.values()]


def torch_output_dtypes(definition: Any) -> list[torch.dtype]:
    """Get the torch data types of the output tensors."""
    return [dtype_str_to_torch_dtype(spec.dtype) for spec in definition.outputs.values()]
