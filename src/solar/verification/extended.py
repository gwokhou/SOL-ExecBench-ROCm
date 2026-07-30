# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Public-PyTorch executor for the Extended native IR dialect."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from solar.errors import (
    NativeAttributeUnsupportedError,
    NativeOperationUnsupportedError,
)
from solar.ir.extended_einsum.native_registry import native_op_spec
from solar.types import DynamicValue
from solar.verification.errors import IRExecutionError
from solar.verification.numerics import torch_equation


def execute_extended_einsum_layer(
    layer_id: str,
    layer: Mapping[str, DynamicValue],
    operands: Sequence[DynamicValue],
    output_shapes: Sequence[tuple[int, ...]],
) -> DynamicValue:
    """Execute one Extended layer without crossing the ATen IR boundary."""
    del output_shapes
    import torch

    semantic = layer.get("semantic_op") or {}
    kind = str(semantic.get("kind", ""))
    if kind == "einsum":
        arguments = _decode_operands(
            semantic.get("operands") or [], operands, layer_id
        )
        return torch.einsum(
            torch_equation(str(semantic.get("equation", ""))),
            *arguments,
        )
    if kind != "operation":
        raise NativeOperationUnsupportedError(
            f"layer {layer_id} has kind {kind!r}"
        )
    target = str(semantic.get("target", ""))
    spec = native_op_spec(target)
    arguments = _decode_operands(
        semantic.get("operands") or [], operands, layer_id
    )
    attributes = {
        str(key): _decode_value(value, operands, layer_id)
        for key, value in (semantic.get("attributes") or {}).items()
    }
    effects = semantic.get("effects") or {}
    if effects.get("mutates") and "out" not in attributes:
        return _execute_mutation(target, arguments, attributes, layer_id)
    return _dispatch_public(
        spec.executor, target, arguments, attributes, layer_id
    )


def _decode_operands(
    encoded: Sequence[DynamicValue],
    tensors: Sequence[DynamicValue],
    layer_id: str,
) -> list[DynamicValue]:
    return [_decode_value(value, tensors, layer_id) for value in encoded]


def _decode_value(
    value: DynamicValue,
    tensors: Sequence[DynamicValue],
    layer_id: str,
) -> DynamicValue:
    import torch

    if isinstance(value, list):
        return [_decode_value(item, tensors, layer_id) for item in value]
    if isinstance(value, tuple):
        return tuple(_decode_value(item, tensors, layer_id) for item in value)
    if not isinstance(value, Mapping):
        return value
    if "tensor" in value:
        index = int(value["tensor"])
        if index not in range(len(tensors)):
            raise IRExecutionError(
                f"layer {layer_id} references missing tensor operand {index}"
            )
        return tensors[index]
    if "literal" in value or "value" in value:
        literal = value.get("literal", value.get("value"))
        return Ellipsis if literal == "__ellipsis__" else literal
    if "dtype" in value:
        result = getattr(torch, str(value["dtype"]), None)
        if not isinstance(result, torch.dtype):
            raise IRExecutionError(
                f"native_attribute_unsupported: invalid dtype {value['dtype']!r}"
            )
        return result
    if "device" in value:
        return torch.device(str(value["device"]))
    if "layout" in value:
        result = getattr(torch, str(value["layout"]), None)
        if not isinstance(result, torch.layout):
            raise IRExecutionError(
                f"native_attribute_unsupported: invalid layout {value['layout']!r}"
            )
        return result
    if "slice" in value:
        parts = [
            _decode_value(item, tensors, layer_id) for item in value["slice"]
        ]
        return slice(*parts)
    raise IRExecutionError(
        f"native_attribute_unsupported: layer {layer_id} has invalid value"
    )


def _execute_mutation(
    target: str,
    arguments: list[DynamicValue],
    attributes: dict[str, DynamicValue],
    layer_id: str,
) -> DynamicValue:
    if not arguments:
        raise NativeAttributeUnsupportedError(
            f"{target} at {layer_id} has no receiver"
        )
    method = getattr(arguments[0], f"{target}_", None)
    if method is None:
        raise NativeOperationUnsupportedError(
            f"mutating {target!r} is unavailable"
        )
    return method(*arguments[1:], **attributes)


def _dispatch_public(
    executor: str,
    target: str,
    arguments: list[DynamicValue],
    attributes: dict[str, DynamicValue],
    layer_id: str,
) -> DynamicValue:
    handlers = {
        "torch": _execute_torch,
        "functional": _execute_functional,
        "method": _execute_method,
        "fft": _execute_fft,
        "linalg": _execute_linalg,
        "special": _execute_special,
    }
    try:
        handler = handlers[executor]
    except KeyError as exc:
        raise NativeOperationUnsupportedError(
            f"no executor for {target!r}"
        ) from exc
    try:
        return handler(target, arguments, attributes)
    except (AttributeError, TypeError, ValueError) as exc:
        raise NativeAttributeUnsupportedError(
            f"{target} at {layer_id}: {exc}"
        ) from exc


def _execute_torch(
    target: str,
    arguments: list[DynamicValue],
    attributes: dict[str, DynamicValue],
) -> DynamicValue:
    import torch

    if target in _METHOD_SHAPES:
        return getattr(arguments[0], target)(*arguments[1:], **attributes)
    if target in {"cat", "stack"} and arguments:
        tensors = (
            arguments[0]
            if isinstance(arguments[0], (list, tuple))
            else arguments
        )
        return getattr(torch, target)(tensors, **attributes)
    operation = getattr(torch, target, None)
    if operation is None:
        raise AttributeError(f"public torch.{target} is unavailable")
    return operation(*arguments, **attributes)


_METHOD_SHAPES = frozenset(
    {"contiguous", "expand", "permute", "repeat", "view"}
)


def _execute_functional(
    target: str,
    arguments: list[DynamicValue],
    attributes: dict[str, DynamicValue],
) -> DynamicValue:
    from torch.nn import functional

    operation = getattr(functional, target, None)
    if operation is None:
        raise AttributeError(
            f"public torch.nn.functional.{target} is unavailable"
        )
    return operation(*arguments, **attributes)


def _execute_method(
    target: str,
    arguments: list[DynamicValue],
    attributes: dict[str, DynamicValue],
) -> DynamicValue:
    if not arguments:
        raise TypeError("method operation has no receiver")
    method = getattr(arguments[0], target, None)
    if method is None:
        raise AttributeError(f"public Tensor.{target} is unavailable")
    return method(*arguments[1:], **attributes)


def _execute_fft(
    target: str,
    arguments: list[DynamicValue],
    attributes: dict[str, DynamicValue],
) -> DynamicValue:
    import torch

    operation = getattr(torch.fft, target.removeprefix("fft_"), None)
    if operation is None:
        raise AttributeError(f"public torch.fft.{target} is unavailable")
    return operation(*arguments, **attributes)


def _execute_linalg(
    target: str,
    arguments: list[DynamicValue],
    attributes: dict[str, DynamicValue],
) -> DynamicValue:
    import torch

    operation = getattr(torch.linalg, target, None)
    if operation is None:
        raise AttributeError(f"public torch.linalg.{target} is unavailable")
    return operation(*arguments, **attributes)


def _execute_special(
    target: str,
    arguments: list[DynamicValue],
    attributes: dict[str, DynamicValue],
) -> DynamicValue:
    if target == "identity":
        return arguments[0]
    if target == "getitem":
        index = arguments[1]
        if isinstance(index, list):
            index = tuple(index)
        return arguments[0][index]
    if target == "slice":
        dimension = int(attributes.get("dim", 0))
        slices = [slice(None)] * arguments[0].ndim
        slices[dimension] = slice(
            attributes.get("start"),
            attributes.get("end"),
            attributes.get("step"),
        )
        return arguments[0][tuple(slices)]
    raise AttributeError(f"native special operation {target!r} is unavailable")


__all__ = ["execute_extended_einsum_layer"]
