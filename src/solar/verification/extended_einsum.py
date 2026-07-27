# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Runtime for the standalone extended-einsum IR dialect."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from solar.verification.errors import IrExecutionError


def execute_extended_layer(
    layer_id: str,
    layer: Mapping[str, Any],
    operands: Sequence[Any],
    output_shapes: Sequence[tuple[int, ...]],
) -> Any:
    """Execute one extended-einsum operation without consulting ATen IR data."""
    operation = layer.get("extended_op") or {}
    name = str(operation.get("operation", ""))
    arguments = [
        _decode(item, operands, layer_id) for item in operation["arguments"]
    ]
    kwargs = {
        str(key): _decode(value, operands, layer_id)
        for key, value in operation["kwargs"].items()
    }
    return _run(name, arguments, kwargs, output_shapes, layer_id)


def _decode(value: Any, operands: Sequence[Any], layer_id: str) -> Any:
    if isinstance(value, list):
        return [_decode(item, operands, layer_id) for item in value]
    if not isinstance(value, Mapping):
        return value
    if "operand" in value:
        index = int(value["operand"])
        if index not in range(len(operands)):
            raise IrExecutionError(
                f"layer {layer_id} references missing extended operand {index}",
            )
        return operands[index]
    if "literal" in value:
        return value["literal"]
    if "dtype" in value:
        import torch

        dtype = getattr(torch, str(value["dtype"]), None)
        if not isinstance(dtype, torch.dtype):
            raise IrExecutionError(
                f"layer {layer_id} references invalid dtype {value['dtype']!r}",
            )
        return dtype
    if "device" in value:
        import torch

        return torch.device(str(value["device"]))
    if "layout" in value:
        import torch

        layout = getattr(torch, str(value["layout"]), None)
        if not isinstance(layout, torch.layout):
            raise IrExecutionError(
                f"layer {layer_id} references invalid layout {value['layout']!r}",
            )
        return layout
    if "slice" in value:
        return slice(
            *[_decode(item, operands, layer_id) for item in value["slice"]]
        )
    raise IrExecutionError(f"layer {layer_id} has an invalid extended argument")


def _run(
    name: str,
    arguments: list[Any],
    kwargs: dict[str, Any],
    output_shapes: Sequence[tuple[int, ...]],
    layer_id: str,
) -> Any:
    special = _special_operation(name, arguments, kwargs, output_shapes)
    if special is not _UNHANDLED:
        return special
    result = _torch_operation(name, arguments, kwargs)
    if result is not _UNHANDLED:
        return result
    result = _tensor_operation(name, arguments, kwargs)
    if result is not _UNHANDLED:
        return result
    raise IrExecutionError(
        f"extended-einsum operation {name!r} at {layer_id} is not executable",
    )


_UNHANDLED = object()


def _special_operation(
    name: str,
    arguments: list[Any],
    kwargs: dict[str, Any],
    output_shapes: Sequence[tuple[int, ...]],
) -> Any:
    import torch
    import torch.nn.functional as functional

    if name == "identity":
        return arguments[0]
    if name in {"conv1d", "conv2d", "conv3d"}:
        return _convolution(functional, name, arguments)
    if name.startswith("conv_transpose"):
        return _transpose_convolution(functional, name, arguments)
    if name == "embedding":
        return _embedding(functional, arguments)
    if name == "scaled_dot_product_attention":
        return functional.scaled_dot_product_attention(*arguments, **kwargs)
    if name == "layer_norm":
        if len(output_shapes) > 1:
            return torch.native_layer_norm(*arguments, **kwargs)
        return functional.layer_norm(*arguments, **kwargs)
    if name == "group_norm":
        return functional.group_norm(*arguments, **kwargs)
    if name == "masked_fill":
        return arguments[0].masked_fill(*arguments[1:], **kwargs)
    if name == "transpose" and len(arguments) == 1:
        return arguments[0].t()
    if name == "amax" and len(output_shapes) > 1:
        return torch.max(*arguments, **kwargs)
    if name == "amin" and len(output_shapes) > 1:
        return torch.min(*arguments, **kwargs)
    if name == "einsum":
        return torch.einsum(str(arguments[0]), *arguments[1:])
    return _UNHANDLED


def _convolution(functional: Any, name: str, arguments: list[Any]) -> Any:
    input_value, weight, bias, stride, padding, dilation = arguments[:6]
    groups = arguments[8] if len(arguments) > 8 else 1
    return getattr(functional, name)(
        input_value,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    )


def _transpose_convolution(
    functional: Any, name: str, arguments: list[Any]
) -> Any:
    (
        input_value,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
    ) = (
        arguments[0],
        arguments[1],
        arguments[2],
        arguments[3],
        arguments[4],
        arguments[7],
        arguments[8],
        arguments[5],
    )
    return getattr(functional, name)(
        input_value,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
    )


def _embedding(functional: Any, arguments: list[Any]) -> Any:
    weight, indices = arguments[:2]
    padding_index = arguments[2] if len(arguments) > 2 else None
    scale_grad_by_freq = arguments[3] if len(arguments) > 3 else False
    sparse = arguments[4] if len(arguments) > 4 else False
    return functional.embedding(
        indices,
        weight,
        padding_index,
        None,
        2.0,
        scale_grad_by_freq,
        sparse,
    )


def _torch_operation(
    name: str, arguments: list[Any], kwargs: dict[str, Any]
) -> Any:
    import torch

    aliases = {"concat": "cat", "t": "t"}
    function = getattr(torch, aliases.get(name, name), None)
    if not callable(function):
        return _UNHANDLED
    try:
        return function(*arguments, **kwargs)
    except (AttributeError, RuntimeError, TypeError):
        return _UNHANDLED


def _tensor_operation(
    name: str, arguments: list[Any], kwargs: dict[str, Any]
) -> Any:
    if not arguments:
        return _UNHANDLED
    method = getattr(arguments[0], name, None)
    if not callable(method):
        return _UNHANDLED
    try:
        return method(*arguments[1:], **kwargs)
    except (AttributeError, RuntimeError, TypeError):
        return _UNHANDLED


__all__ = ["execute_extended_layer"]
