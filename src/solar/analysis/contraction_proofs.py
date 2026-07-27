# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Build exact Orojenesis proof views without changing executable semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

from solar.analysis.resources import is_mfma_operation
from solar.common.types import GraphValue, NodeDict, TensorShapes
from solar.einsum.analyzer import EinsumAnalyzer
from solar.ir.contracts import layer_operation

_PROOF_INPUTS: dict[str, tuple[str, tuple[int, ...]]] = {
    "addmm": ("matmul", (1, 2)),
    "bmm": ("bmm", (0, 1)),
    "linear": ("linear", (0, 1)),
    "matmul": ("matmul", (0, 1)),
    "mm": ("mm", (0, 1)),
}


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
    proof_spec = _PROOF_INPUTS.get(target)
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
    try:
        operation_proof = analyzer.get_einsum_op(
            operation,
            TensorShapes(inputs=input_shapes, outputs=output_shapes),
        )
    except (TypeError, ValueError):
        return None
    if (
        not operation_proof.is_real_einsum
        or not operation_proof.is_einsum_supportable
        or "->" not in operation_proof.equation
    ):
        return None
    proof["semantic_op"] = {
        "kind": "einsum",
        "target": "einsum",
        "equation": operation_proof.equation,
        "arguments": [{"tensor": index} for index in range(len(indices))],
        "kwargs": {},
        "effects": deepcopy(dict(semantic.get("effects") or {})),
        "proof_source": {"kind": "aten", "target": target},
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
