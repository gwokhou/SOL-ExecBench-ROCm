# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Build exact Orojenesis proof views without changing executable semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

from solar.analysis.resources import is_mfma_operation
from solar.ir.contracts import layer_operation
from solar.ir.extended_einsum.operations.analyzer import EinsumAnalyzer
from solar.types import GraphValue, NodeDict, TensorShapes

_PROOF_INPUTS: dict[str, tuple[str, tuple[int, ...]]] = {
    "addmm": ("matmul", (1, 2)),
    "bmm": ("bmm", (0, 1)),
    "linear": ("linear", (0, 1)),
    "matmul": ("matmul", (0, 1)),
    "mm": ("mm", (0, 1)),
}

_CONVOLUTION_TARGETS = frozenset({"conv1d", "conv2d", "conv3d"})
_MISSING = object()


def requires_tile_evidence(layer: Mapping[str, GraphValue]) -> bool:
    """Return whether a layer contains contraction work needing tile evidence."""
    semantic = layer_operation(layer)
    target = str(semantic.get("target") or layer.get("type") or "").lower()
    return (
        semantic.get("kind") == "einsum"
        or is_mfma_operation(target)
        or target == "scaled_dot_product_attention"
    )


def _select(
    values: Sequence[GraphValue],
    indices: Sequence[int],
) -> list[GraphValue] | None:
    try:
        return [deepcopy(values[index]) for index in indices]
    except IndexError:
        return None


def _select_input_metadata(
    proof: NodeDict,
    indices: tuple[int, ...],
) -> tuple[list[GraphValue], list[GraphValue]] | None:
    containers = ("tensor_names", "tensor_shapes", "tensor_dtypes")
    for container_name in containers:
        container = proof.get(container_name)
        if not isinstance(container, dict):
            return None
        inputs = container.get("inputs")
        if not isinstance(inputs, list):
            return None
        selected = _select(inputs, indices)
        if selected is None:
            return None
        container["inputs"] = selected
    tensor_types = proof.get("tensor_types")
    if isinstance(tensor_types, dict) and isinstance(
        tensor_types.get("inputs"),
        list,
    ):
        selected_types = _select(tensor_types["inputs"], indices)
        if selected_types is None:
            return None
        tensor_types["inputs"] = selected_types
    shapes = proof["tensor_shapes"]
    return list(shapes["inputs"]), list(shapes.get("outputs") or [])


def _literal(value: GraphValue) -> GraphValue:
    if isinstance(value, Mapping) and set(value) == {"value"}:
        return value["value"]
    if isinstance(value, list):
        items = tuple(_literal(item) for item in value)
        return _MISSING if _MISSING in items else items
    return _MISSING


def _convolution_kwargs(
    semantic: Mapping[str, GraphValue],
    input_shapes: list[GraphValue],
    output_shapes: list[GraphValue],
) -> dict[str, GraphValue] | None:
    """Return exact handler parameters for the reviewed direct-conv subset."""
    target = str(semantic.get("target") or "").lower()
    if target not in _CONVOLUTION_TARGETS:
        return None
    dimensions = int(target[-2])
    unit = (1,) * dimensions
    zero = (0,) * dimensions
    values = _convolution_values(semantic, unit=unit, zero=zero)
    if values is None or len(input_shapes) != 2 or len(output_shapes) != 1:
        return None
    stride, padding, dilation, transposed, output_padding, groups = values
    if (
        stride != unit
        or padding != zero
        or dilation != unit
        or transposed is not False
        or output_padding != zero
        or type(groups) is not int
    ):
        return None
    input_shape, weight_shape = input_shapes
    output_shape = output_shapes[0]
    if not all(
        isinstance(shape, list)
        and len(shape) == dimensions + 2
        and all(type(size) is int and size > 0 for size in shape)
        for shape in (input_shape, weight_shape, output_shape)
    ):
        return None
    in_channels = input_shape[1]
    out_channels = weight_shape[0]
    standard = groups == 1 and weight_shape[1] == in_channels
    depthwise = (
        target in {"conv1d", "conv2d"}
        and groups == in_channels == out_channels == output_shape[1]
        and weight_shape[1] == 1
    )
    expected_spatial = [
        input_shape[index] - weight_shape[index] + 1
        for index in range(2, dimensions + 2)
    ]
    if (
        not (standard or depthwise)
        or output_shape[0] != input_shape[0]
        or output_shape[1] != out_channels
        or output_shape[2:] != expected_spatial
    ):
        return None
    return {
        "stride": unit,
        "padding": zero,
        "dilation": unit,
        "module_args": {
            "groups": groups,
            "in_channels": in_channels,
            "out_channels": out_channels,
        },
    }


def _convolution_values(
    semantic: Mapping[str, GraphValue],
    *,
    unit: tuple[int, ...],
    zero: tuple[int, ...],
) -> tuple[GraphValue, ...] | None:
    """Read exact convolution controls from ATen or default native semantics."""
    arguments = semantic.get("arguments")
    if isinstance(arguments, list) and len(arguments) >= 9:
        values = tuple(_literal(arguments[index]) for index in range(3, 9))
        return None if _MISSING in values else values
    operands = semantic.get("operands")
    attributes = semantic.get("attributes")
    if (
        semantic.get("kind") == "operation"
        and isinstance(operands, list)
        and len(operands) in {2, 3}
        and attributes == {}
    ):
        return unit, zero, unit, False, zero, 1
    return None


def build_orojenesis_proof_layer(
    layer: Mapping[str, GraphValue],
    *,
    analyzer: EinsumAnalyzer,
) -> NodeDict | None:
    """Return an exact einsum proof view for a supported contraction layer."""
    semantic = layer_operation(layer)
    if semantic.get("kind") == "einsum":
        equation = str(semantic.get("equation", ""))
        return deepcopy(dict(layer)) if "->" in equation else None

    target = str(semantic.get("target") or layer.get("type") or "").lower()
    proof_spec = (
        (target, (0, 1))
        if target in _CONVOLUTION_TARGETS
        else _PROOF_INPUTS.get(target)
    )
    if proof_spec is None:
        return None
    operation, indices = proof_spec
    proof = deepcopy(dict(layer))
    selected_shapes = _select_input_metadata(proof, indices)
    if selected_shapes is None:
        return None
    input_shapes, output_shapes = selected_shapes
    if len(output_shapes) != 1:
        return None
    operation_kwargs: dict[str, GraphValue] = {}
    if target in _CONVOLUTION_TARGETS:
        parameters = _convolution_kwargs(
            semantic,
            input_shapes,
            output_shapes,
        )
        if parameters is None:
            return None
        operation_kwargs = parameters
    try:
        operation_proof = analyzer.get_einsum_op(
            operation,
            TensorShapes(inputs=input_shapes, outputs=output_shapes),
            **operation_kwargs,
        )
    except (TypeError, ValueError):
        return None
    if (
        not operation_proof.is_real_einsum
        or not operation_proof.is_einsum_supportable
        or "->" not in operation_proof.equation
    ):
        return None
    proof_source_kind = (
        "aten"
        if isinstance(semantic.get("arguments"), list)
        else "extended_native"
    )
    proof["semantic_op"] = {
        "kind": "einsum",
        "target": "einsum",
        "equation": operation_proof.equation,
        "arguments": [{"tensor": index} for index in range(len(indices))],
        "kwargs": {},
        "effects": deepcopy(dict(semantic.get("effects") or {})),
        "proof_source": {"kind": proof_source_kind, "target": target},
    }
    proof["einsum_equation"] = operation_proof.equation
    proof["is_real_einsum"] = True
    return proof


def build_orojenesis_proof_graph(
    all_layers: Mapping[str, NodeDict],
    active_layers: Mapping[str, NodeDict],
    *,
    analyzer: EinsumAnalyzer,
) -> tuple[dict[str, NodeDict], dict[str, NodeDict], tuple[str, ...]]:
    """Build a proof-only graph and list contractions without an exact view."""
    proof_graph_layers = dict(all_layers)
    proof_layers: dict[str, NodeDict] = {}
    unsupported: list[str] = []
    for layer_id, layer in active_layers.items():
        if not requires_tile_evidence(layer):
            continue
        proof = build_orojenesis_proof_layer(layer, analyzer=analyzer)
        if proof is None:
            unsupported.append(layer_id)
            continue
        proof_graph_layers[layer_id] = proof
        proof_layers[layer_id] = proof
    return proof_graph_layers, proof_layers, tuple(sorted(unsupported))


__all__ = [
    "build_orojenesis_proof_graph",
    "build_orojenesis_proof_layer",
    "requires_tile_evidence",
]
