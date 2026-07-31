# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Versioned, fail-closed semantics for Extended native IR."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from solar.errors import DynamicShapeUnboundedError, SolarError
from solar.ir.extended_einsum.native_registry import (
    NATIVE_OP_REGISTRY,
    NativeOpSpec,
    canonical_native_target,
    native_op_spec,
    validate_native_attributes,
)
from solar.schema_versions import EXTENDED_EINSUM_IR_SCHEMA_VERSION


class SemanticGraphError(ValueError):
    """An Extended graph does not contain complete native semantics."""


def _plain_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _plain_value(item)
            for key, item in value.items()
            if str(key) != "raw_attributes"
        }
    return str(value)


def _canonical_target(layer: Mapping[str, Any]) -> str:
    raw = str(layer.get("type", "")).lower().rsplit(".", maxsplit=1)[-1]
    output_count = len(
        (layer.get("tensor_names") or {}).get("outputs") or [],
    )
    if raw.rstrip("_") in {"max", "min"} and output_count == 2:
        return raw.rstrip("_")
    return canonical_native_target(raw)


def _default_effects(
    spec: NativeOpSpec,
    input_arity: int,
    output_arity: int,
) -> dict[str, Any]:
    aliases = []
    if spec.effects.aliases_input and input_arity and output_arity:
        aliases = [
            {
                "output": index,
                "input": 0,
                "conditional": spec.target in {"contiguous", "reshape"},
            }
            for index in range(output_arity)
        ]
    return {
        "mutates": [0] if spec.effects.mutates_input and input_arity else [],
        "aliases": aliases,
        "atomic": spec.effects.atomic,
        "opaque_library_call": spec.effects.opaque_library_call,
    }


def _normalize_call(
    layer: Mapping[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    names = (layer.get("tensor_names") or {}).get("inputs") or []
    module_args = layer.get("module_args") or {}
    recorded_operands = module_args.get("call_arguments")
    recorded_attributes = module_args.get("call_kwargs")
    operands = (
        _plain_value(recorded_operands)
        if isinstance(recorded_operands, list)
        else [{"tensor": index} for index in range(len(names))]
    )
    attributes = (
        _plain_value(recorded_attributes)
        if isinstance(recorded_attributes, Mapping)
        else {}
    )
    if not isinstance(recorded_attributes, Mapping):
        for source in (module_args, layer.get("additional_info") or {}):
            if not isinstance(source, Mapping):
                continue
            for key, value in source.items():
                if key not in _NON_SEMANTIC_KEYS and value is not None:
                    attributes[str(key)] = _plain_value(value)
    if "dims" in attributes and "dim" not in attributes:
        attributes["dim"] = attributes.pop("dims")
    attributes.pop("_stacklevel", None)
    return list(operands), dict(attributes)


_NON_SEMANTIC_KEYS = frozenset(
    {
        "call_arguments",
        "call_kwargs",
        "function_name",
        "hierarchical_name",
        "raw_attributes",
        "training",
    }
)


def _reverse_operands(raw_target: str, operands: list[Any]) -> None:
    if raw_target not in {
        "__radd__",
        "__rmul__",
        "__rpow__",
        "__rsub__",
        "__rtruediv__",
    }:
        return
    if len(operands) != 2:
        raise SemanticGraphError(
            f"native_attribute_unsupported: {raw_target} requires two operands"
        )
    operands.reverse()


def _observed_effects(
    layer: Mapping[str, Any],
    spec: NativeOpSpec,
    operands: Sequence[Any],
    attributes: Mapping[str, Any],
) -> dict[str, Any]:
    input_arity = len((layer.get("tensor_names") or {}).get("inputs") or [])
    output_arity = len((layer.get("tensor_names") or {}).get("outputs") or [])
    effects = _default_effects(spec, input_arity, output_arity)
    raw_target = str(layer.get("type", ""))
    mutating = (
        raw_target.endswith("_") and not raw_target.endswith("__")
    ) or layer.get("mutates_inputs") is True
    aliases = _plain_value(layer.get("aliases") or [])
    if aliases:
        effects["aliases"] = list(aliases)
    if mutating and operands:
        effects["mutates"] = [0]
        effects["aliases"] = [{"output": 0, "input": 0}]
    out_reference = attributes.get("out")
    if isinstance(out_reference, Mapping) and "tensor" in out_reference:
        out_index = int(out_reference["tensor"])
        effects["mutates"] = [out_index]
        effects["aliases"] = [{"output": 0, "input": out_index}]
    return effects


def build_semantic_operation(layer: Mapping[str, Any]) -> dict[str, Any]:
    """Build one v6 native semantic record from captured public call data."""
    if str(layer.get("type", "")).lower() == "start":
        return {"kind": "input", "target": "input"}
    operands, attributes = _normalize_call(layer)
    target = _canonical_target(layer)
    if target in {"flip", "roll"} and "dim" in attributes:
        attributes["dims"] = attributes.pop("dim")
    if (
        layer.get("is_real_einsum") is True
        and layer.get("einsum_equation")
        and target not in _NATIVE_OPERATION_ONLY
    ):
        contraction_operands = [
            operand
            for operand in operands
            if _contains_tensor_reference(operand)
        ]
        return {
            "kind": "einsum",
            "target": "einsum",
            "equation": str(layer["einsum_equation"]),
            "operands": contraction_operands,
            "attributes": {},
            "effects": _empty_effects(),
        }
    spec = native_op_spec(target)
    validate_native_attributes(spec, attributes)
    raw_target = str(layer.get("type", "")).lower().rsplit(".", maxsplit=1)[-1]
    _reverse_operands(raw_target, operands)
    if (
        target in {"view", "reshape"}
        and len(operands) == 1
        and "shape" not in attributes
    ):
        output_shapes = (layer.get("tensor_shapes") or {}).get("outputs") or []
        if len(output_shapes) == 1:
            attributes["shape"] = _plain_value(output_shapes[0])
    return {
        "kind": "operation",
        "target": target,
        "operands": operands,
        "attributes": attributes,
        "effects": _observed_effects(layer, spec, operands, attributes),
    }


_NATIVE_OPERATION_ONLY = frozenset(
    {
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "scaled_dot_product_attention",
    }
)


def _contains_tensor_reference(value: Any) -> bool:
    if isinstance(value, Mapping):
        return "tensor" in value or any(
            _contains_tensor_reference(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor_reference(item) for item in value)
    return False


def _empty_effects() -> dict[str, Any]:
    return {
        "mutates": [],
        "aliases": [],
        "atomic": False,
        "opaque_library_call": False,
    }


def annotate_semantics(
    graph: dict[str, Any], *, strict: bool
) -> dict[str, Any]:
    """Attach v6 native semantics and optionally validate strictly."""
    graph["schema_version"] = EXTENDED_EINSUM_IR_SCHEMA_VERSION
    dynamic_index = 0
    for layer_id, layer in (graph.get("layers") or {}).items():
        if not isinstance(layer, dict):
            raise SemanticGraphError(f"layer {layer_id} is not a mapping")
        try:
            layer["semantic_op"] = build_semantic_operation(layer)
            if _annotate_dynamic_shapes(
                layer, layer["semantic_op"], dynamic_index
            ):
                dynamic_index += 1
        except (SemanticGraphError, SolarError):
            if strict:
                raise
            layer["semantic_op"] = {
                "kind": "unsupported",
                "target": _canonical_target(layer),
                "reason": "native operation parameters are incomplete",
            }
    if strict:
        validate_semantic_graph(graph)
    return graph


def _annotate_dynamic_shapes(
    layer: dict[str, Any],
    semantic: Mapping[str, Any],
    dynamic_index: int,
) -> bool:
    if (
        semantic.get("kind") != "operation"
        or semantic.get("target") != "nonzero"
    ):
        return False
    shapes = layer.get("tensor_shapes") or {}
    inputs = shapes.get("inputs") or []
    outputs = shapes.get("outputs") or []
    if len(inputs) != 1 or not outputs:
        raise DynamicShapeUnboundedError("nonzero lacks tensor metadata")
    upper = dynamic_shape_upper_numel(inputs[0])
    trace_values = [int(shape[0]) for shape in outputs if shape]
    if not trace_values or len(set(trace_values)) != 1:
        raise DynamicShapeUnboundedError("nonzero outputs disagree on nnz")
    descriptor = {
        "symbol": f"nnz{dynamic_index}",
        "lower": 0,
        "upper": upper,
        "trace_value": trace_values[0],
    }
    for shape in outputs:
        shape[0] = dict(descriptor)
    return True


def validate_semantic_graph(graph: Mapping[str, Any]) -> None:
    """Validate v6 without accepting legacy Extended artifacts."""
    if graph.get("schema_version") != EXTENDED_EINSUM_IR_SCHEMA_VERSION:
        raise SemanticGraphError(
            "extended-einsum graph must use current "
            f"schema_version={EXTENDED_EINSUM_IR_SCHEMA_VERSION}"
        )
    layers = graph.get("layers")
    if not isinstance(layers, Mapping) or not layers:
        raise SemanticGraphError("einsum graph has no layers")
    symbols: dict[str, tuple[int, int, int]] = {}
    for layer_id, layer in layers.items():
        _validate_layer(str(layer_id), layer, symbols)


def _validate_layer(
    layer_id: str,
    layer: Any,
    symbols: dict[str, tuple[int, int, int]],
) -> None:
    if not isinstance(layer, Mapping):
        raise SemanticGraphError(f"layer {layer_id} is not a mapping")
    shapes = layer.get("tensor_shapes") or {}
    dtypes = layer.get("tensor_dtypes") or {}
    names = layer.get("tensor_names") or {}
    for side in ("inputs", "outputs"):
        tensors = shapes.get(side) or []
        if len(dtypes.get(side) or []) != len(tensors) or len(
            names.get(side) or []
        ) != len(tensors):
            raise SemanticGraphError(
                f"layer {layer_id} lacks explicit {side} name/shape/dtype metadata"
            )
        for shape in tensors:
            _validate_shape(shape, layer_id, symbols)
    semantic = layer.get("semantic_op")
    if not isinstance(semantic, Mapping):
        raise SemanticGraphError(f"layer {layer_id} has no semantic_op")
    kind = str(semantic.get("kind", ""))
    if kind == "input":
        return
    if kind == "einsum":
        _validate_einsum(layer_id, semantic)
        return
    if kind != "operation":
        raise SemanticGraphError(f"layer {layer_id} is not executable exactly")
    _validate_native_operation(layer_id, semantic, names)


def _validate_shape(
    shape: Any,
    layer_id: str,
    symbols: dict[str, tuple[int, int, int]],
) -> None:
    if not isinstance(shape, (list, tuple)):
        raise SemanticGraphError(
            f"layer {layer_id} has an invalid tensor shape"
        )
    for dimension in shape:
        if isinstance(dimension, int) and dimension >= 0:
            continue
        if not isinstance(dimension, Mapping):
            raise DynamicShapeUnboundedError(
                f"layer {layer_id} has invalid dimension"
            )
        required = {"symbol", "lower", "upper", "trace_value"}
        if set(dimension) != required:
            raise DynamicShapeUnboundedError(
                f"layer {layer_id} lacks bounded dimension"
            )
        symbol = str(dimension["symbol"])
        lower = int(dimension["lower"])
        upper = int(dimension["upper"])
        trace = int(dimension["trace_value"])
        if not symbol or lower < 0 or lower > trace or trace > upper:
            raise DynamicShapeUnboundedError(
                f"layer {layer_id} has invalid bounds"
            )
        contract = (lower, upper, trace)
        if symbol in symbols and symbols[symbol] != contract:
            raise DynamicShapeUnboundedError(
                f"symbol {symbol!r} is inconsistent"
            )
        symbols[symbol] = contract


def _validate_einsum(
    layer_id: str,
    semantic: Mapping[str, Any],
) -> None:
    equation = str(semantic.get("equation", ""))
    if not equation or "->" not in equation:
        raise SemanticGraphError(
            f"layer {layer_id} has no exact einsum equation"
        )
    if not isinstance(semantic.get("operands"), list):
        raise SemanticGraphError(f"layer {layer_id} lacks ordered operands")
    if not isinstance(semantic.get("attributes"), Mapping):
        raise SemanticGraphError(f"layer {layer_id} lacks named attributes")


def _collect_tensor_references(
    value: Any,
    references: set[int],
) -> None:
    if isinstance(value, Mapping):
        if "tensor" in value:
            references.add(int(value["tensor"]))
        else:
            for item in value.values():
                _collect_tensor_references(item, references)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_tensor_references(item, references)


def _validate_native_operation(
    layer_id: str,
    semantic: Mapping[str, Any],
    names: Mapping[str, Any],
) -> None:
    target = str(semantic.get("target", ""))
    spec = native_op_spec(target)
    operands = semantic.get("operands")
    attributes = semantic.get("attributes")
    if not isinstance(operands, list):
        raise SemanticGraphError(f"layer {layer_id} lacks ordered operands")
    if not isinstance(attributes, Mapping):
        raise SemanticGraphError(f"layer {layer_id} lacks named attributes")
    validate_native_attributes(spec, attributes)
    input_arity = len(names.get("inputs") or [])
    output_arity = len(names.get("outputs") or [])
    _validate_arity(layer_id, spec, input_arity, output_arity)
    references: set[int] = set()
    _collect_tensor_references(operands, references)
    _collect_tensor_references(attributes, references)
    if references != set(range(input_arity)):
        raise SemanticGraphError(
            f"layer {layer_id} does not preserve every ordered tensor operand"
        )
    _validate_effects(layer_id, semantic, input_arity, output_arity)
    _validate_required_parameters(layer_id, target, operands, attributes)


def _validate_arity(
    layer_id: str,
    spec: NativeOpSpec,
    input_arity: int,
    output_arity: int,
) -> None:
    input_min, input_max = spec.input_arity
    output_min, output_max = spec.output_arity
    if input_arity < input_min or (
        input_max is not None and input_arity > input_max
    ):
        raise SemanticGraphError(
            f"layer {layer_id} has invalid input arity for {spec.target}"
        )
    if output_arity < output_min or (
        output_max is not None and output_arity > output_max
    ):
        raise SemanticGraphError(
            f"layer {layer_id} has invalid output arity for {spec.target}"
        )


def _validate_effects(
    layer_id: str,
    semantic: Mapping[str, Any],
    input_arity: int,
    output_arity: int,
) -> None:
    effects = semantic.get("effects")
    if not isinstance(effects, Mapping):
        raise SemanticGraphError(f"layer {layer_id} lacks explicit effects")
    mutations = effects.get("mutates")
    aliases = effects.get("aliases")
    if not isinstance(mutations, list) or not isinstance(aliases, list):
        raise SemanticGraphError(f"layer {layer_id} has invalid effects")
    if any(int(index) not in range(input_arity) for index in mutations):
        raise SemanticGraphError(
            f"layer {layer_id} has invalid mutation target"
        )
    for alias in aliases:
        if (
            not isinstance(alias, Mapping)
            or int(alias.get("input", -1)) not in range(input_arity)
            or int(alias.get("output", -1)) not in range(output_arity)
        ):
            raise SemanticGraphError(
                f"layer {layer_id} has invalid alias effect"
            )


def _validate_required_parameters(
    layer_id: str,
    target: str,
    operands: Sequence[Any],
    attributes: Mapping[str, Any],
) -> None:
    required = {
        "gather": (("dim", 3),),
        "index_add": (("dim", 4),),
        "index_select": (("dim", 3),),
        "log_softmax": (("dim", 2),),
        "repeat_interleave": (("repeats", 2),),
        "softmax": (("dim", 2),),
        "topk": (("k", 2),),
    }
    missing = [
        name
        for name, positional_arity in required.get(target, ())
        if name not in attributes and len(operands) < positional_arity
    ]
    if missing:
        raise SemanticGraphError(
            f"native_attribute_unsupported: layer {layer_id} lacks "
            + ", ".join(missing)
        )


def dynamic_shape_trace(shape: Sequence[Any]) -> tuple[int, ...]:
    """Return the trace-time concrete shape for executor setup."""
    result = []
    for dimension in shape:
        if isinstance(dimension, Mapping):
            result.append(int(dimension["trace_value"]))
        else:
            result.append(int(dimension))
    return tuple(result)


def dynamic_shape_upper_numel(shape: Sequence[Any]) -> int:
    """Return a bounded upper element count for validation and analysis."""
    return math.prod(
        int(dimension["upper"])
        if isinstance(dimension, Mapping)
        else int(dimension)
        for dimension in shape
    )


SUPPORTED_NATIVE_TARGETS = frozenset(NATIVE_OP_REGISTRY)

__all__ = [
    "EXTENDED_EINSUM_IR_SCHEMA_VERSION",
    "SUPPORTED_NATIVE_TARGETS",
    "SemanticGraphError",
    "annotate_semantics",
    "build_semantic_operation",
    "dynamic_shape_trace",
    "dynamic_shape_upper_numel",
    "validate_semantic_graph",
]
