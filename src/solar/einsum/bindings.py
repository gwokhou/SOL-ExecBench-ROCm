# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Exact source-input and reference-output bindings for converted graphs."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from solar.common.types import DynamicValue, NodeDict
from solar.einsum.semantics import validate_semantic_graph
from solar.graph.extraction import OperatorGraphArtifact, TensorSignature


def accept_semantic_operator_graph(
    traced: Mapping[str, DynamicValue],
    operator: OperatorGraphArtifact,
    output: Path,
) -> Path:
    """Validate and persist an exact make_fx ATen graph."""
    converted: NodeDict = deepcopy(dict(traced))
    starts = _start_layers(converted)
    source_indices = list(operator.used_source_indices)
    if len(starts) != len(source_indices):
        raise RuntimeError("make_fx graph input arity does not match reference")
    for layer, source_index in zip(starts, source_indices):
        layer["source_input_index"] = source_index
    converted["source_input_indices"] = bind_inputs(converted, operator)
    _validate_declared_outputs(converted, operator.reference_outputs)
    validate_semantic_graph(converted)
    einsum_path = output / "einsum_graph.yaml"
    einsum_path.write_text(yaml.safe_dump(converted, sort_keys=False))
    return einsum_path


def _validate_declared_outputs(
    graph: Mapping[str, DynamicValue], expected: Sequence[TensorSignature]
) -> None:
    declared = graph.get("outputs")
    if not isinstance(declared, list) or len(declared) != len(expected):
        raise RuntimeError("make_fx graph output arity does not match reference")
    metadata: dict[str, TensorSignature] = {}
    for layer in (graph.get("layers") or {}).values():
        names = (layer.get("tensor_names") or {}).get("outputs") or []
        shapes = (layer.get("tensor_shapes") or {}).get("outputs") or []
        dtypes = (layer.get("tensor_dtypes") or {}).get("outputs") or []
        if len(names) != len(shapes) or len(names) != len(dtypes):
            raise RuntimeError("make_fx graph output metadata is incomplete")
        metadata.update(
            {
                str(name): TensorSignature(tuple(shape), str(dtype))
                for name, shape, dtype in zip(names, shapes, dtypes)
            }
        )
    if any(metadata.get(str(name)) != value for name, value in zip(declared, expected)):
        raise RuntimeError("make_fx graph outputs do not match reference metadata")


def _start_layers(graph: Mapping[str, DynamicValue]) -> list[NodeDict]:
    return [
        layer
        for layer in (graph.get("layers") or {}).values()
        if str(layer.get("type", "")).lower() == "start"
    ]


def bind_inputs(
    graph: Mapping[str, DynamicValue], operator: OperatorGraphArtifact
) -> list[int]:
    """Bind executor tensor slots to their exact source argument indices."""
    slots = _graph_input_slots(graph)
    ordered = list(operator.used_source_indices)
    if len(slots) != len(ordered):
        raise RuntimeError(
            "cannot bind source arguments to graph inputs: "
            f"observed={ordered}, graph_inputs={len(slots)}"
        )
    signatures = dict(operator.source_inputs)
    bindings: dict[str, int] = {}
    used: set[int] = set()
    unresolved: list[NodeDict] = []
    for slot in slots:
        source_index = slot.get("source_input_index")
        if source_index is None:
            if slot.get("source_binding") == "torchview_input_order":
                raise RuntimeError("trace input provenance is incomplete")
            unresolved.append(slot)
            continue
        source_index = int(source_index)
        if source_index not in ordered or source_index in used:
            raise RuntimeError("trace input provenance is invalid")
        _validate_input_binding(slot, source_index, signatures)
        bindings[str(slot["tensor_name"])] = source_index
        used.add(source_index)
    remaining = [index for index in ordered if index not in used]
    candidates = [_input_candidates(slot, remaining, signatures) for slot in unresolved]
    inferred: list[list[int]] = []
    _search_bindings(unresolved, candidates, 0, [], set(remaining), {}, inferred)
    if len(inferred) != 1:
        reason = "no" if not inferred else "ambiguous"
        raise RuntimeError(f"{reason} exact recovered-input binding")
    for slot, source_index in zip(unresolved, inferred[0]):
        bindings[str(slot["tensor_name"])] = source_index
    if len(bindings) != len(slots):
        raise RuntimeError("graph input provenance is incomplete")
    return [bindings[str(slot["tensor_name"])] for slot in slots]


def _graph_input_slots(graph: Mapping[str, DynamicValue]) -> list[NodeDict]:
    """Return start and external tensor slots in executor consumption order."""
    layers = graph.get("layers") or {}
    produced = {
        str(name)
        for layer in layers.values()
        for name in ((layer.get("tensor_names") or {}).get("outputs") or [])
    }
    slots: list[NodeDict] = []
    for layer in _start_layers(graph):
        slots.extend(_layer_slots(layer, side="outputs"))
    seen: set[str] = set()
    for layer in layers.values():
        if str(layer.get("type", "")).lower() == "start":
            continue
        for slot in _layer_slots(layer, side="inputs"):
            name = str(slot["tensor_name"])
            if name in produced or name in seen:
                continue
            seen.add(name)
            external = dict(slot)
            external["source_binding"] = "external_tensor_order"
            slots.append(external)
    return slots


def _layer_slots(layer: Mapping[str, DynamicValue], *, side: str) -> list[NodeDict]:
    names = (layer.get("tensor_names") or {}).get(side) or []
    shapes = (layer.get("tensor_shapes") or {}).get(side) or []
    dtypes = (layer.get("tensor_dtypes") or {}).get(side) or []
    if len(names) != len(shapes) or len(names) != len(dtypes):
        raise RuntimeError(f"graph {side} lacks exact name/shape/dtype metadata")
    binding = str(layer.get("source_binding") or "torchview_input_order")
    source_input_index = layer.get("source_input_index")
    slots: list[NodeDict] = [
        {
            "tensor_name": str(name),
            "tensor_shapes": {"outputs": [shape]},
            "tensor_dtypes": {"outputs": [dtype]},
            "source_binding": binding,
        }
        for name, shape, dtype in zip(names, shapes, dtypes)
    ]
    if source_input_index is not None:
        if len(slots) != 1:
            raise RuntimeError("source input provenance requires one tensor slot")
        slots[0]["source_input_index"] = int(source_input_index)
    return slots


def _input_candidates(
    slot: Mapping[str, DynamicValue],
    indices: Sequence[int],
    inputs: Mapping[int, TensorSignature],
) -> list[int]:
    shapes = (slot.get("tensor_shapes") or {}).get("outputs") or []
    dtypes = (slot.get("tensor_dtypes") or {}).get("outputs") or []
    if len(shapes) != 1 or len(dtypes) != 1:
        raise RuntimeError("graph input lacks exact shape/dtype metadata")
    return [
        index
        for index in indices
        if tuple(shapes[0]) == inputs[index].shape
        and str(dtypes[0]) == inputs[index].dtype
    ]


def _search_bindings(
    slots: Sequence[Mapping[str, DynamicValue]],
    candidates: Sequence[Sequence[int]],
    position: int,
    chosen: list[int],
    remaining: set[int],
    last_ordered: Mapping[str, int],
    results: list[list[int]],
) -> None:
    if len(results) > 1:
        return
    if position == len(candidates):
        results.append(list(chosen))
        return
    binding_group = str(slots[position].get("source_binding") or "")
    for source_index in candidates[position]:
        if source_index not in remaining or source_index <= last_ordered.get(
            binding_group, -1
        ):
            continue
        chosen.append(source_index)
        next_ordered = dict(last_ordered)
        next_ordered[binding_group] = source_index
        _search_bindings(
            slots,
            candidates,
            position + 1,
            chosen,
            remaining - {source_index},
            next_ordered,
            results,
        )
        chosen.pop()


def _validate_input_binding(
    slot: Mapping[str, DynamicValue],
    source_index: int,
    inputs: Mapping[int, TensorSignature],
) -> None:
    shapes = (slot.get("tensor_shapes") or {}).get("outputs") or []
    dtypes = (slot.get("tensor_dtypes") or {}).get("outputs") or []
    if len(shapes) != 1 or len(dtypes) != 1:
        raise RuntimeError("graph input lacks exact shape/dtype metadata")
    signature = inputs.get(source_index)
    if (
        signature is None
        or tuple(shapes[0]) != signature.shape
        or str(dtypes[0]) != signature.dtype
    ):
        raise RuntimeError("trace input provenance does not match graph metadata")


def bind_outputs(
    graph: Mapping[str, DynamicValue],
    traced: Mapping[str, DynamicValue],
    expected: Sequence[TensorSignature],
) -> list[str]:
    """Bind declared graph outputs to exact traced reference slots."""
    candidates = _output_candidates(graph, traced)
    if len(candidates) != len(expected):
        raise RuntimeError("cannot preserve exact reference output arity")
    declared: list[str] = []
    for value in expected:
        matches = [
            index
            for index, (_, shape, dtype) in enumerate(candidates)
            if tuple(shape) == value.shape and dtype == value.dtype
        ]
        if not matches:
            raise RuntimeError("traced output metadata does not match reference")
        declared.append(candidates.pop(matches[0])[0])
    return declared


def _output_candidates(
    graph: Mapping[str, DynamicValue], traced: Mapping[str, DynamicValue]
) -> list[tuple[str, list[int], str]]:
    layers = graph.get("layers") or {}
    traced_layers = traced.get("layers") or {}
    output_nodes = [
        (layer_id, layer)
        for layer_id, layer in traced_layers.items()
        if str(layer.get("type", "")).lower() == "output-tensor"
    ]
    result: list[tuple[str, list[int], str]] = []
    for output_id, output in output_nodes:
        producers = (output.get("connections") or {}).get("inputs") or []
        if len(producers) != 1:
            raise RuntimeError("cannot bind exact traced graph output")
        original_id = str(producers[0])
        producer = _resolve_output_producer(layers, original_id)
        names = (producer.get("tensor_names") or {}).get("outputs") or []
        shapes = (producer.get("tensor_shapes") or {}).get("outputs") or []
        dtypes = (producer.get("tensor_dtypes") or {}).get("outputs") or []
        original_outputs = (
            (traced_layers.get(original_id) or {}).get("connections") or {}
        ).get("outputs") or []
        if output_id not in original_outputs:
            raise RuntimeError("cannot identify traced graph output slot")
        index = original_outputs.index(output_id)
        if index >= len(names) or index >= len(shapes) or index >= len(dtypes):
            raise RuntimeError("converted graph output arity does not match trace")
        result.append((str(names[index]), list(shapes[index]), str(dtypes[index])))
    return result


def _resolve_output_producer(
    layers: Mapping[str, DynamicValue], original_id: str
) -> Mapping[str, DynamicValue]:
    """Resolve the terminal layer of a reviewed split/expanded operation."""
    current_id = original_id
    if current_id not in layers:
        candidates = [
            layer_id
            for layer_id, layer in layers.items()
            if layer_id.startswith(f"{original_id}.")
            and not ((layer.get("connections") or {}).get("outputs") or [])
        ]
        if len(candidates) != 1:
            raise RuntimeError("cannot bind exact traced graph output")
        current_id = candidates[0]
    visited: set[str] = set()
    while current_id not in visited:
        visited.add(current_id)
        layer = layers[current_id]
        successors = [
            str(candidate)
            for candidate in ((layer.get("connections") or {}).get("outputs") or [])
            if str(candidate) in layers and str(candidate).startswith(f"{original_id}.")
        ]
        if len(successors) != 1:
            return layer
        current_id = successors[0]
    raise RuntimeError("converted output remap contains a cycle")


__all__ = [
    "OperatorGraphArtifact",
    "accept_semantic_operator_graph",
    "bind_inputs",
    "bind_outputs",
]
