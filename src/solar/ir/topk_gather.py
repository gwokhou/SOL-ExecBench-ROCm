# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Frozen top-k gather read provenance, without tensor sampling.

A gather whose index operand is the index output of a top-k over an
identically shaped operand reads, per index row, ``k`` distinct in-range
positions along the selection dimension: top-k selects distinct positions,
and identical shapes keep those positions in range for the gather source.
The frozen vocabulary proves only that such a gather contributes no
external reads: both of its operands must be computed values, never an
affine view of an external base. Every other gather keeps its
dynamic-index rejection; nothing here generalizes to arbitrary indexing.
"""

from collections.abc import Mapping

from solar.ir.contracts import (
    layer_operation,
    operation_attributes,
    operation_operands,
)
from solar.ir.semantic_values import (
    EXTENDED_VALUE_POLICY,
    SemanticValueDecodeError,
    decode_semantic_value,
)
from solar.types import NodeDict

TOPK_TARGETS = frozenset({"topk"})
TOPK_GATHER_TARGETS = frozenset({"gather"})
TOPK_INDEX_OUTPUT_SLOT = 1
_PASSTHROUGH_TARGETS = frozenset({"identity", "hidden-tensor"})


def _decode_int(value: object) -> int | None:
    """Decode one static integer operand, or keep the case unproved."""
    try:
        decoded = decode_semantic_value(
            value,
            (),
            policy=EXTENDED_VALUE_POLICY,  # type: ignore[arg-type]
        )
    except SemanticValueDecodeError:
        return None
    return decoded if type(decoded) is int else None


def _tensor_operand(
    operands: tuple[object, ...], index: int, names: tuple[str, ...]
) -> str | None:
    operand = operands[index] if 0 <= index < len(operands) else None
    if not isinstance(operand, Mapping):
        return None
    position = operand.get("tensor")
    if type(position) is not int or not 0 <= position < len(names):
        return None
    return names[position]


def _layer_shapes(layer: NodeDict) -> tuple[list[list[int]], list[list[int]]]:
    shapes = layer.get("tensor_shapes") or {}
    inputs = shapes.get("inputs") or []
    outputs = shapes.get("outputs") or []
    return (
        [s for s in inputs if isinstance(s, list)],
        [s for s in outputs if isinstance(s, list)],
    )


def _static_shape(shape: object) -> tuple[int, ...] | None:
    if not isinstance(shape, list):
        return None
    dims: list[int] = []
    for extent in shape:
        if type(extent) is not int:
            return None
        dims.append(extent)
    return tuple(dims)


def _normalize_dim(dim: int, rank: int) -> int | None:
    normalized = dim + rank if dim < 0 else dim
    return normalized if 0 <= normalized < rank else None


def _topk_selection(semantic: Mapping[str, object]) -> tuple[int, int] | None:
    """Decode the static ``(k, dim)`` of a top-k selection from either dialect."""
    operands = tuple(operation_operands(semantic))
    attributes = operation_attributes(semantic)
    k = _decode_int(attributes.get("k")) if "k" in attributes else None
    if k is None and len(operands) > 1:
        k = _decode_int(operands[1])
    dim = _decode_int(attributes.get("dim")) if "dim" in attributes else None
    if dim is None and len(operands) > 2:
        dim = _decode_int(operands[2])
    if dim is None:
        dim = -1
    if k is None or dim is None or k < 1:
        return None
    return k, dim


def _gather_operands(
    semantic: Mapping[str, object], names: tuple[str, ...]
) -> tuple[str | None, int | None, str | None]:
    """Decode ``(source, dim, index)`` of a gather from either dialect."""
    operands = tuple(operation_operands(semantic))
    attributes = operation_attributes(semantic)
    source = _tensor_operand(operands, 0, names)
    index = _tensor_operand(operands, 2, names) if len(operands) == 3 else None
    dim = _decode_int(attributes.get("dim")) if "dim" in attributes else None
    if dim is None and len(operands) > 1:
        dim = _decode_int(operands[1])
    return source, dim, index


def topk_gather_read_proved(
    gather_layer: NodeDict,
    layers: Mapping[str, NodeDict],
    external_views: Mapping[str, object],
    producer_of: Mapping[str, str],
) -> bool:
    """Prove the frozen top-k gather class contributes no external reads.

    The index operand must reduce, through single-output passthrough layers
    only, to the index output of a top-k with static ``k`` whose operand
    shape equals the gather source shape along every dimension. Both gather
    operands must be computed values outside ``external_views``; a gather
    directly over an external base stays rejected because its data-dependent
    selection has no affine region representation.
    """
    semantic = layer_operation(gather_layer)
    names_layer = gather_layer.get("tensor_names") or {}
    inputs = tuple(names_layer.get("inputs") or ())
    source_name, gather_dim, index_name = _gather_operands(semantic, inputs)
    if source_name is None or index_name is None or gather_dim is None:
        return False
    if source_name in external_views or index_name in external_views:
        return False
    input_shapes, _ = _layer_shapes(gather_layer)
    source_shape = (
        _static_shape(input_shapes[0]) if input_shapes and inputs else None
    )
    index_shape = (
        _static_shape(input_shapes[inputs.index(index_name)])
        if len(input_shapes) == len(inputs)
        else None
    )
    if source_shape is None or index_shape is None:
        return False
    gather_axis = _normalize_dim(gather_dim, len(source_shape))
    if gather_axis is None:
        return False

    name = index_name
    for _ in range(len(layers) + 1):
        layer_id = producer_of.get(name)
        if layer_id is None:
            return False
        layer = layers.get(layer_id)
        if layer is None:
            return False
        layer_semantic = layer_operation(layer)
        target = str(layer_semantic.get("target", ""))
        outputs = tuple((layer.get("tensor_names") or {}).get("outputs") or ())
        passthrough_inputs = tuple(
            (layer.get("tensor_names") or {}).get("inputs") or ()
        )
        if (
            target in _PASSTHROUGH_TARGETS
            and len(outputs) == 1
            and len(passthrough_inputs) == 1
            and outputs[0] == name
        ):
            name = passthrough_inputs[0]
            continue
        if target in TOPK_TARGETS:
            selection = _topk_selection(layer_semantic)
            if selection is None or len(outputs) != 2:
                return False
            k, topk_dim = selection
            if outputs[TOPK_INDEX_OUTPUT_SLOT] != name:
                return False
            topk_inputs = tuple(
                (layer.get("tensor_names") or {}).get("inputs") or ()
            )
            operands = tuple(operation_operands(layer_semantic))
            operand_name = _tensor_operand(operands, 0, topk_inputs)
            if operand_name is None:
                return False
            operand_shapes, _ = _layer_shapes(layer)
            position = topk_inputs.index(operand_name)
            operand_shape = (
                _static_shape(operand_shapes[position])
                if position < len(operand_shapes)
                else None
            )
            if operand_shape is None or operand_shape != source_shape:
                return False
            topk_axis = _normalize_dim(topk_dim, len(operand_shape))
            if topk_axis != gather_axis:
                return False
            expected_index = (
                source_shape[:gather_axis]
                + (k,)
                + source_shape[gather_axis + 1 :]
            )
            return (
                1 <= k <= source_shape[gather_axis]
                and index_shape == expected_index
            )
        return False
    return False


__all__ = [
    "TOPK_GATHER_TARGETS",
    "TOPK_INDEX_OUTPUT_SLOT",
    "TOPK_TARGETS",
    "topk_gather_read_proved",
]
