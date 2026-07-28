# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Extended-einsum intermediate representation backend."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from solar.common.types import TensorShapes
from solar.einsum.analyzer import EinsumAnalyzer
from solar.einsum.operation_conversion import (
    FORCE_ATEN_SEMANTICS_OPS,
    OperationRepresentation,
    default_operation_representation,
)
from solar.einsum.operation_policy import SUPPORTABLE_OPERATIONS
from solar.graph.contracts import OperatorGraphArtifact
from solar.ir.bindings import bind_graph
from solar.ir.contracts import IRBackend, IRKind
from solar.schema_versions import IR_GRAPH_SCHEMA_VERSION
from solar.verification.extended_einsum import execute_extended_layer

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


class ExtendedEinsumIRError(ValueError):
    """An extended-einsum IR graph is malformed or not supportable exactly."""


def convert_operator_graph(
    operator: OperatorGraphArtifact,
    output_dir: str | Path,
) -> Path:
    """Convert a canonical ATen trace into standalone extended-einsum IR."""
    traced = _load_operator_graph(operator)
    converted = _convert_graph(traced)
    bind_graph(converted, operator)
    validate_extended_einsum_graph(converted)
    path = Path(output_dir) / "einsum_graph.yaml"
    path.write_text(yaml.safe_dump(converted, sort_keys=False))
    return path


def validate_extended_einsum_graph(graph: Mapping[str, Any]) -> None:
    """Validate an extended-einsum graph without accepting embedded ATen IR."""
    if graph.get("ir_kind") != "extended_einsum":
        raise ExtendedEinsumIRError("graph is not extended_einsum IR")
    if int(graph.get("schema_version", 0)) != IR_GRAPH_SCHEMA_VERSION:
        raise ExtendedEinsumIRError(
            "extended-einsum graph must use current "
            f"schema_version={IR_GRAPH_SCHEMA_VERSION}",
        )
    layers = graph.get("layers")
    if not isinstance(layers, Mapping) or not layers:
        raise ExtendedEinsumIRError("extended-einsum graph has no layers")
    for layer_id, layer in layers.items():
        _validate_layer(str(layer_id), layer)


def _load_operator_graph(operator: OperatorGraphArtifact) -> Mapping[str, Any]:
    traced = yaml.safe_load(operator.path.read_text()) or {}
    if traced.get("extraction_kind") != "make_fx_reference_v1":
        raise RuntimeError("extended-einsum source provenance is not trusted")
    return traced


def _convert_graph(traced: Mapping[str, Any]) -> dict[str, Any]:
    layers = traced.get("layers") or {}
    return {
        "schema_version": IR_GRAPH_SCHEMA_VERSION,
        "ir_kind": "extended_einsum",
        "model_name": traced.get("model_name"),
        "extraction_kind": traced.get("extraction_kind"),
        "joint_graph": bool(traced.get("joint_graph", False)),
        "outputs": deepcopy(traced.get("outputs") or []),
        "graph_signature": deepcopy(traced.get("graph_signature") or {}),
        "layers": {
            str(layer_id): _convert_layer(str(layer_id), layer)
            for layer_id, layer in layers.items()
        },
    }


def _convert_layer(layer_id: str, source: Mapping[str, Any]) -> dict[str, Any]:
    target = str((source.get("semantic_op") or {}).get("target", ""))
    operation = _operation_name(target, source)
    representation = _representation(layer_id, source, operation)
    layer = _base_layer(source)
    layer.update(
        {
            "is_real_einsum": representation.is_real_einsum,
            "is_einsum_supportable": representation.is_einsum_supportable,
            "einsum_equation": representation.equation,
            "operands": representation.operands,
            "elementwise_op": representation.elementwise_op,
            "reduction_op": representation.reduction_op,
            "extended_op": _extended_operation(
                source, operation, representation
            ),
        },
    )
    return layer


def _base_layer(source: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "type",
        "phase",
        "tensor_names",
        "tensor_shapes",
        "tensor_dtypes",
        "connections",
        "source_input_index",
    )
    return {
        field: deepcopy(source[field]) for field in fields if field in source
    }


def _operation_name(target: str, source: Mapping[str, Any]) -> str:
    if target:
        return target
    if str(source.get("type", "")).lower() == "start":
        return "input"
    raise ExtendedEinsumIRError("operation has no canonical target")


def _representation(
    layer_id: str,
    source: Mapping[str, Any],
    operation: str,
) -> OperationRepresentation:
    if operation == "input":
        return default_operation_representation()
    handler_operation = _HANDLER_ALIASES.get(operation, operation)
    if not _is_supportable(handler_operation):
        raise ExtendedEinsumIRError(
            f"layer {layer_id} uses unsupported extended-einsum operation "
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
        return (
            replace(result, is_real_einsum=False)
            if operation in FORCE_ATEN_SEMANTICS_OPS
            else result
        )
    except (TypeError, ValueError):
        return default_operation_representation()


def _is_supportable(operation: str) -> bool:
    return (
        operation in SUPPORTABLE_OPERATIONS
        or operation in _PASSTHROUGH_OPERATIONS
    )


def _tensor_shapes(source: Mapping[str, Any]) -> TensorShapes:
    shapes = source.get("tensor_shapes") or {}
    return TensorShapes(
        inputs=[list(shape) for shape in shapes.get("inputs") or []],
        outputs=[list(shape) for shape in shapes.get("outputs") or []],
    )


def _handler_kwargs(
    source: Mapping[str, Any], operation: str
) -> dict[str, Any]:
    semantic = source.get("semantic_op") or {}
    kwargs = {
        str(key): _literal_value(value)
        for key, value in (semantic.get("kwargs") or {}).items()
        if _literal_value(value) is not _MISSING
    }
    values = [
        _literal_value(value) for value in semantic.get("arguments") or []
    ]
    positional = _positional_parameters(operation, values)
    for key, value in positional.items():
        kwargs.setdefault(key, value)
    return kwargs


_MISSING = object()


def _literal_value(value: Any) -> Any:
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


def _positional_parameters(operation: str, values: list[Any]) -> dict[str, Any]:
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


def _extended_operation(
    source: Mapping[str, Any],
    operation: str,
    representation: OperationRepresentation,
) -> dict[str, Any]:
    semantic = source.get("semantic_op") or {}
    return {
        "operation": operation,
        "equation": representation.equation,
        "operands": representation.operands,
        "elementwise_op": representation.elementwise_op,
        "reduction_op": representation.reduction_op,
        "is_real_einsum": representation.is_real_einsum,
        "arguments": [
            _encode_argument(item) for item in semantic.get("arguments") or []
        ],
        "kwargs": {
            str(key): _encode_argument(value)
            for key, value in (semantic.get("kwargs") or {}).items()
        },
        "effects": deepcopy((semantic.get("effects") or {})),
    }


def _encode_argument(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "tensor" in value:
            return {"operand": int(value["tensor"])}
        if "value" in value:
            return {"literal": deepcopy(value["value"])}
        if "slice" in value:
            return {
                "slice": [_encode_argument(item) for item in value["slice"]]
            }
        return {str(key): _encode_argument(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_argument(item) for item in value]
    return deepcopy(value)


def _validate_layer(layer_id: str, layer: Any) -> None:
    if not isinstance(layer, Mapping):
        raise ExtendedEinsumIRError(f"layer {layer_id} is not a mapping")
    if "semantic_op" in layer:
        raise ExtendedEinsumIRError(
            f"layer {layer_id} embeds ATen semantic_op in extended-einsum IR",
        )
    _validate_tensor_metadata(layer_id, layer)
    operation = layer.get("extended_op")
    if not isinstance(operation, Mapping):
        raise ExtendedEinsumIRError(f"layer {layer_id} has no extended_op")
    if not str(operation.get("operation", "")):
        raise ExtendedEinsumIRError(f"layer {layer_id} has no operation name")
    if not isinstance(operation.get("arguments"), list):
        raise ExtendedEinsumIRError(f"layer {layer_id} has invalid arguments")
    if not isinstance(operation.get("kwargs"), Mapping):
        raise ExtendedEinsumIRError(
            f"layer {layer_id} has invalid keyword arguments"
        )
    if bool(operation.get("is_real_einsum")) and "->" not in str(
        operation.get("equation", ""),
    ):
        raise ExtendedEinsumIRError(
            f"layer {layer_id} has no extended-einsum equation",
        )


def _validate_tensor_metadata(layer_id: str, layer: Mapping[str, Any]) -> None:
    names = layer.get("tensor_names") or {}
    shapes = layer.get("tensor_shapes") or {}
    dtypes = layer.get("tensor_dtypes") or {}
    for side in ("inputs", "outputs"):
        arity = len(shapes.get(side) or [])
        if (
            len(names.get(side) or []) != arity
            or len(dtypes.get(side) or []) != arity
        ):
            raise ExtendedEinsumIRError(
                f"layer {layer_id} lacks explicit {side} name/shape/dtype metadata",
            )


backend = IRBackend(
    IRKind.EXTENDED_EINSUM,
    validate_extended_einsum_graph,
    convert_operator_graph,
    execute_extended_layer,
)


__all__ = [
    "ExtendedEinsumIRError",
    "backend",
    "convert_operator_graph",
    "validate_extended_einsum_graph",
]
