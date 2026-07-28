# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Normalize a trusted make_fx graph into the NVLabs einsum analysis schema."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace

from solar.common.types import DynamicValue, TensorShapes
from solar.einsum.analyzer import EinsumAnalyzer
from solar.einsum.operation_conversion import (
    FORCE_ATEN_SEMANTICS_OPS,
    OperationRepresentation,
    default_operation_representation,
)
from solar.einsum.operation_policy import SUPPORTABLE_OPERATIONS
from solar.errors import UnsupportedOperationError
from solar.graph.contracts import ExtractionKind, OperatorGraphArtifact
from solar.ir.bindings import bind_graph
from solar.ir.contracts import IRKind

_HANDLER_ALIASES = {"addmm": "matmul", "amax": "max", "amin": "min"}
_PASSTHROUGH_OPERATIONS = frozenset(
    {
        "identity",
        "addmm",
        "copy",
        "dequantize",
        "embedding_bag",
        "gather",
        "index_select",
        "maximum",
        "minimum",
        "narrow",
        "ones_like",
        "square",
        "type_as",
        "where",
        "zeros_like",
    },
)
_MISSING = object()


class MakeFXNVLabsConversionError(UnsupportedOperationError):
    """A make_fx layer cannot be represented in the NVLabs analysis schema."""


def convert_make_fx_graph(
    operator: OperatorGraphArtifact,
) -> dict[str, DynamicValue]:
    """Convert one trusted make_fx operator artifact without losing ATen replay."""
    document = operator.document
    traced = document.data
    if traced.get("extraction_kind") != ExtractionKind.MAKE_FX_REFERENCE.value:
        raise RuntimeError("make_fx graph provenance is not trusted")
    converted: dict[str, DynamicValue] = {
        key: deepcopy(value) for key, value in traced.items()
    }
    converted["ir_kind"] = IRKind.NVLABS_EINSUM.value
    converted_layers = {}
    for layer_id, layer in document.require_mapping("layers").items():
        if not isinstance(layer, dict):
            raise ValueError(f"operator layer {layer_id!r} is not a mapping")
        converted_layers[layer_id] = _convert_layer(layer_id, layer)
    converted["layers"] = converted_layers
    bind_graph(converted, operator)
    return converted


def _convert_layer(
    layer_id: str,
    source: Mapping[str, DynamicValue],
) -> dict[str, DynamicValue]:
    layer = deepcopy(dict(source))
    semantic = source.get("semantic_op") or {}
    layer["aten_semantic_op"] = deepcopy(dict(semantic))
    operation = str(semantic.get("target", ""))
    if str(source.get("type", "")).lower() == "start":
        representation = default_operation_representation()
    elif not operation:
        raise MakeFXNVLabsConversionError(
            f"layer {layer_id} has no canonical operation",
        )
    else:
        representation = _representation(layer_id, source, operation)
    layer.update(
        {
            "is_real_einsum": representation.is_real_einsum,
            "is_einsum_supportable": representation.is_einsum_supportable,
            "einsum_equation": representation.equation,
            "operands": representation.operands,
            "elementwise_op": representation.elementwise_op,
            "reduction_op": representation.reduction_op,
        },
    )
    if representation.is_real_einsum:
        layer["semantic_op"] = {
            **deepcopy(dict(semantic)),
            "kind": "einsum",
            "target": "einsum",
            "equation": representation.equation,
            "kwargs": {},
        }
    return layer


def _representation(
    layer_id: str,
    source: Mapping[str, DynamicValue],
    operation: str,
) -> OperationRepresentation:
    handler_operation = _HANDLER_ALIASES.get(operation, operation)
    if not _is_supportable(handler_operation):
        raise MakeFXNVLabsConversionError(
            f"layer {layer_id} uses unsupported NVLabs einsum operation "
            f"{operation!r}",
        )
    try:
        result = OperationRepresentation.from_einsum_op(
            EinsumAnalyzer().get_einsum_op(
                handler_operation,
                _tensor_shapes(source),
                **_handler_kwargs(source, operation),
            ),
        )
    except (TypeError, ValueError):
        result = default_operation_representation()
    if operation in FORCE_ATEN_SEMANTICS_OPS:
        return replace(result, is_real_einsum=False)
    return result


def _is_supportable(operation: str) -> bool:
    return (
        operation in SUPPORTABLE_OPERATIONS
        or operation in _PASSTHROUGH_OPERATIONS
    )


def _tensor_shapes(source: Mapping[str, DynamicValue]) -> TensorShapes:
    shapes = source.get("tensor_shapes") or {}
    return TensorShapes(
        inputs=[list(shape) for shape in shapes.get("inputs") or []],
        outputs=[list(shape) for shape in shapes.get("outputs") or []],
    )


def _handler_kwargs(
    source: Mapping[str, DynamicValue],
    operation: str,
) -> dict[str, DynamicValue]:
    semantic = source.get("semantic_op") or {}
    kwargs = {
        str(key): decoded
        for key, value in (semantic.get("kwargs") or {}).items()
        if (decoded := _literal_value(value)) is not _MISSING
    }
    values = [
        _literal_value(value) for value in semantic.get("arguments") or []
    ]
    for key, value in _positional_parameters(operation, values).items():
        kwargs.setdefault(key, value)
    return kwargs


def _literal_value(value: DynamicValue) -> DynamicValue:
    if isinstance(value, Mapping):
        if "tensor" in value:
            return _MISSING
        if "value" in value:
            return value["value"]
        if "dtype" in value:
            return value["dtype"]
        return _MISSING
    if isinstance(value, list):
        decoded = [_literal_value(item) for item in value]
        return (
            _MISSING if any(item is _MISSING for item in decoded) else decoded
        )
    return value


def _positional_parameters(
    operation: str,
    values: list[DynamicValue],
) -> dict[str, DynamicValue]:
    positions = {
        "sum": ((1, "dims"), (2, "keepdim")),
        "mean": ((1, "dims"), (2, "keepdim")),
        "prod": ((1, "dims"), (2, "keepdim")),
        "amax": ((1, "dims"), (2, "keepdim")),
        "amin": ((1, "dims"), (2, "keepdim")),
        "softmax": ((1, "dim"),),
        "log_softmax": ((1, "dim"),),
        "cumsum": ((1, "dim"),),
        "transpose": ((1, "dim0"), (2, "dim1")),
        "permute": ((1, "dims"),),
    }
    return {
        name: values[index]
        for index, name in positions.get(operation, ())
        if index < len(values) and values[index] is not _MISSING
    }


__all__ = [
    "MakeFXNVLabsConversionError",
    "convert_make_fx_graph",
]
