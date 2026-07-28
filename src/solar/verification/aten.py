# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Runtime for the standalone ATen IR dialect."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from solar.common.types import DynamicValue
from solar.verification.errors import IRExecutionError
from solar.verification.numerics import torch_equation

_UNHANDLED = object()


def execute_aten_layer(
    layer_id: str,
    layer: Mapping[str, DynamicValue],
    operands: Sequence[DynamicValue],
    output_shapes: Sequence[tuple[int, ...]],
) -> DynamicValue:
    """Execute one ATen IR operation from its ``semantic_op`` payload."""
    import torch

    semantic = layer["semantic_op"]
    if semantic["kind"] == "einsum" and not semantic.get("exact_target"):
        return torch.einsum(
            torch_equation(str(semantic["equation"])),
            *operands,
        )
    arguments = [
        _decode_semantic_argument(item, operands, layer_id)
        for item in semantic.get("arguments") or []
    ]
    kwargs = {
        str(key): _decode_semantic_argument(value, operands, layer_id)
        for key, value in (semantic.get("kwargs") or {}).items()
    }
    target = str(semantic.get("target", ""))
    handlers = (
        _execute_exact_aten,
        _execute_mutation,
        _execute_arithmetic,
        _execute_shape,
        _execute_indexing,
        _execute_functional,
        _execute_quantized,
        _execute_aten_fallback,
    )
    for handler in handlers:
        result = handler(
            target,
            arguments,
            kwargs,
            semantic,
            layer_id,
            output_shapes,
        )
        if result is not _UNHANDLED:
            return result
    raise IRExecutionError(
        f"operation {target!r} at {layer_id} is not executable exactly",
    )


def _decode_semantic_argument(
    argument: DynamicValue,
    operands: Sequence[DynamicValue],
    layer_id: str,
) -> DynamicValue:
    import torch

    if argument == "preserve_format":
        return torch.preserve_format
    if argument == "contiguous_format":
        return torch.contiguous_format
    if isinstance(argument, list):
        return [
            _decode_semantic_argument(item, operands, layer_id)
            for item in argument
        ]
    if isinstance(argument, tuple):
        return tuple(
            _decode_semantic_argument(item, operands, layer_id)
            for item in argument
        )
    if not isinstance(argument, Mapping):
        return argument
    if "tensor" in argument:
        index = int(argument["tensor"])
        if index < 0 or index >= len(operands):
            raise IRExecutionError(
                f"layer {layer_id} references missing tensor argument {index}",
            )
        return operands[index]
    if "dtype" in argument:
        dtype = getattr(torch, str(argument["dtype"]), None)
        if not isinstance(dtype, torch.dtype):
            raise IRExecutionError(
                f"layer {layer_id} references invalid dtype "
                f"{argument['dtype']!r}",
            )
        return dtype
    if "device" in argument:
        return torch.device(str(argument["device"]))
    if "layout" in argument:
        layout = getattr(torch, str(argument["layout"]), None)
        if not isinstance(layout, torch.layout):
            raise IRExecutionError(
                f"layer {layer_id} references invalid layout "
                f"{argument['layout']!r}",
            )
        return layout
    if "value" in argument:
        value = argument["value"]
        if value == "__ellipsis__":
            return Ellipsis
        if value == "preserve_format":
            return torch.preserve_format
        if value == "contiguous_format":
            return torch.contiguous_format
        return value
    if "slice" in argument:
        values = [
            _decode_semantic_argument(item, operands, layer_id)
            for item in argument["slice"]
        ]
        return slice(*values)
    raise IRExecutionError(
        f"layer {layer_id} has an invalid semantic argument",
    )


def _execute_exact_aten(
    target: str,
    arguments: list[DynamicValue],
    kwargs: dict[str, DynamicValue],
    semantic: Mapping[str, DynamicValue],
    layer_id: str,
    output_shapes: Sequence[tuple[int, ...]],
) -> DynamicValue:
    del target, layer_id, output_shapes
    import torch

    exact_target = semantic.get("exact_target")
    if not isinstance(exact_target, str):
        return _UNHANDLED
    packet = getattr(torch.ops.aten, exact_target, None)
    overload_name = str(semantic.get("overload", "default"))
    overload = getattr(packet, overload_name, None) if packet else None
    if overload is None:
        raise IRExecutionError(
            f"ATen operation {exact_target}.{overload_name} is unavailable",
        )
    return overload(*arguments, **kwargs)


def _execute_mutation(
    target: str,
    arguments: list[DynamicValue],
    kwargs: dict[str, DynamicValue],
    semantic: Mapping[str, DynamicValue],
    layer_id: str,
    output_shapes: Sequence[tuple[int, ...]],
) -> DynamicValue:
    del output_shapes
    if not (semantic.get("effects") or {}).get("mutates"):
        return _UNHANDLED
    if not arguments:
        raise IRExecutionError(
            f"mutating operation {target!r} at {layer_id} has no receiver",
        )
    method = getattr(arguments[0], f"{target}_", None)
    if method is None:
        raise IRExecutionError(
            f"mutating operation {target!r} at {layer_id} is unavailable",
        )
    return method(*arguments[1:], **kwargs)


def _execute_arithmetic(
    target: str,
    arguments: list[DynamicValue],
    kwargs: dict[str, DynamicValue],
    semantic: Mapping[str, DynamicValue],
    layer_id: str,
    output_shapes: Sequence[tuple[int, ...]],
) -> DynamicValue:
    del semantic, layer_id, output_shapes
    import torch
    import torch.nn.functional as functional

    operations = {**_binary_operations(), **_unary_operations()}
    if target in operations:
        return operations[target](*arguments, **kwargs)
    if target in {"mm", "bmm", "matmul", "addmm", "where"}:
        return getattr(torch, target)(*arguments, **kwargs)
    if target == "masked_fill":
        return arguments[0].masked_fill(*arguments[1:], **kwargs)
    if target == "cumsum":
        return torch.cumsum(*arguments, **kwargs)
    if target in {"softmax", "log_softmax"}:
        return getattr(functional, target)(*arguments, **kwargs)
    if target in {
        "sum",
        "mean",
        "prod",
        "amax",
        "amin",
        "argmax",
        "argmin",
        "logsumexp",
    }:
        return getattr(torch, target)(*arguments, **kwargs)
    return _UNHANDLED


def _binary_operations() -> dict[str, Callable[..., DynamicValue]]:
    import torch

    return {
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "div": torch.div,
        "eq": torch.eq,
        "ge": torch.ge,
        "gt": torch.gt,
        "le": torch.le,
        "lt": torch.lt,
        "ne": torch.ne,
        "pow": torch.pow,
        "maximum": torch.maximum,
        "minimum": torch.minimum,
        "bitwise_and": torch.bitwise_and,
    }


def _unary_operations() -> dict[str, Callable[..., DynamicValue]]:
    import torch
    import torch.nn.functional as functional

    return {
        "abs": torch.abs,
        "bitwise_not": torch.bitwise_not,
        "cos": torch.cos,
        "elu": functional.elu,
        "exp": torch.exp,
        "gelu": functional.gelu,
        "hardsigmoid": functional.hardsigmoid,
        "hardswish": functional.hardswish,
        "leaky_relu": functional.leaky_relu,
        "log": torch.log,
        "mish": functional.mish,
        "neg": torch.neg,
        "relu": functional.relu,
        "rsqrt": torch.rsqrt,
        "sigmoid": torch.sigmoid,
        "silu": functional.silu,
        "sin": torch.sin,
        "sqrt": torch.sqrt,
        "square": torch.square,
        "tanh": torch.tanh,
    }


def _execute_shape(
    target: str,
    arguments: list[DynamicValue],
    kwargs: dict[str, DynamicValue],
    semantic: Mapping[str, DynamicValue],
    layer_id: str,
    output_shapes: Sequence[tuple[int, ...]],
) -> DynamicValue:
    del semantic
    import torch

    if target == "identity":
        return arguments[0]
    if target == "to":
        return arguments[0].to(*arguments[1:], **kwargs)
    if target in {"bfloat16", "float", "half", "int", "long"}:
        return getattr(arguments[0], target)()
    if target in {"type_as", "clone", "detach"}:
        return getattr(arguments[0], target)(*arguments[1:], **kwargs)
    if target in {"view", "reshape"}:
        if len(arguments) > 1:
            return getattr(arguments[0], target)(*arguments[1:], **kwargs)
        shape = kwargs.pop("shape", output_shapes[0])
        return getattr(arguments[0], target)(tuple(shape))
    if target == "flatten":
        return torch.flatten(*arguments, **kwargs)
    if target == "contiguous":
        return arguments[0].contiguous(**kwargs)
    if target in {
        "squeeze",
        "unsqueeze",
        "permute",
        "repeat",
        "repeat_interleave",
        "expand",
    }:
        return getattr(arguments[0], target)(*arguments[1:], **kwargs)
    if target == "transpose":
        if len(arguments) == 1 and not kwargs:
            if arguments[0].ndim != 2:
                raise IRExecutionError(
                    f"layer {layer_id} requires explicit transpose dimensions",
                )
            return arguments[0].t()
        return torch.transpose(*arguments, **kwargs)
    if target in {"cat", "stack"}:
        if arguments and isinstance(arguments[0], (list, tuple)):
            return getattr(torch, target)(*arguments, **kwargs)
        return getattr(torch, target)(arguments, **kwargs)
    if target == "vstack":
        return torch.vstack(*arguments, **kwargs)
    if target in {"chunk", "split"}:
        return getattr(torch, target)(*arguments, **kwargs)
    return _UNHANDLED


def _execute_indexing(
    target: str,
    arguments: list[DynamicValue],
    kwargs: dict[str, DynamicValue],
    semantic: Mapping[str, DynamicValue],
    layer_id: str,
    output_shapes: Sequence[tuple[int, ...]],
) -> DynamicValue:
    del semantic, layer_id, output_shapes
    import torch

    if target in {"gather", "scatter", "index_select", "select", "narrow"}:
        return getattr(torch, target)(*arguments, **kwargs)
    if target == "getitem":
        index = arguments[1]
        if isinstance(index, list) and any(
            isinstance(item, slice) or item is None or item is Ellipsis
            for item in index
        ):
            index = tuple(index)
        return arguments[0][index]
    if target == "slice":
        dimension = int(kwargs.get("dim", 0))
        slices = [slice(None)] * arguments[0].ndim
        slices[dimension] = slice(
            kwargs.get("start"),
            kwargs.get("end"),
            kwargs.get("step"),
        )
        return arguments[0][tuple(slices)]
    return _UNHANDLED


def _execute_functional(
    target: str,
    arguments: list[DynamicValue],
    kwargs: dict[str, DynamicValue],
    semantic: Mapping[str, DynamicValue],
    layer_id: str,
    output_shapes: Sequence[tuple[int, ...]],
) -> DynamicValue:
    del semantic, layer_id, output_shapes
    import torch.nn.functional as functional

    if target == "linear" or target.startswith("conv_transpose"):
        return getattr(functional, target)(*arguments, **kwargs)
    if target in {
        "conv1d",
        "conv2d",
        "conv3d",
        "batch_norm",
        "group_norm",
        "instance_norm",
        "layer_norm",
        "embedding",
        "embedding_bag",
        "dropout",
        "max_pool2d",
        "scaled_dot_product_attention",
    }:
        return getattr(functional, target)(*arguments, **kwargs)
    return _UNHANDLED


def _execute_quantized(
    target: str,
    arguments: list[DynamicValue],
    kwargs: dict[str, DynamicValue],
    semantic: Mapping[str, DynamicValue],
    layer_id: str,
    output_shapes: Sequence[tuple[int, ...]],
) -> DynamicValue:
    del semantic, layer_id, output_shapes
    import torch

    if target in {
        "quantize_per_tensor",
        "quantize_per_channel",
        "fake_quantize_per_tensor_affine",
        "fake_quantize_per_channel_affine",
    }:
        return getattr(torch, target)(*arguments, **kwargs)
    if target == "dequantize":
        return arguments[0].dequantize()
    if target in {"ones_like", "zeros_like"}:
        return getattr(torch, target)(*arguments, **kwargs)
    if target == "clamp":
        return torch.clamp(*arguments, **kwargs)
    return _UNHANDLED


def _execute_aten_fallback(
    target: str,
    arguments: list[DynamicValue],
    kwargs: dict[str, DynamicValue],
    semantic: Mapping[str, DynamicValue],
    layer_id: str,
    output_shapes: Sequence[tuple[int, ...]],
) -> DynamicValue:
    del layer_id, output_shapes
    import torch

    if not target.isidentifier() or not hasattr(torch.ops.aten, target):
        return _UNHANDLED
    packet = getattr(torch.ops.aten, target)
    overload_name = str(semantic.get("overload", "default"))
    overload = getattr(packet, overload_name, None)
    if overload is None:
        raise IRExecutionError(
            f"ATen operation {target}.{overload_name} is unavailable",
        )
    return overload(*arguments, **kwargs)


__all__ = ["execute_aten_layer", "torch_equation"]
