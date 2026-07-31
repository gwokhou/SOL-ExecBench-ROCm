# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the pinned Timeloop/Orojenesis mapper implementation."""

# pylint: disable=missing-function-docstring,unspecified-encoding,too-many-locals,too-many-statements,too-many-branches,too-many-lines,too-many-boolean-expressions

from __future__ import annotations

import contextlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solar.analysis.orojenesis.configuration import (
    MULTI_EINSUM_BATCH_COMPOSITION,
    MULTI_EINSUM_FANOUT_COMPOSITION,
    MULTI_EINSUM_LAYOUT_COMPOSITION,
)
from solar.analysis.orojenesis.curves import (
    compose_multi_einsum_region_curve as _compose_multi_einsum_region_curve,
)
from solar.analysis.orojenesis.errors import OrojenesisError
from solar.analysis.orojenesis.multi_einsum import (
    _region_matmul_descriptor,
    _shape_product,
    find_multi_einsum_chains,
)
from solar.ir.contracts import (
    CONTRACTION_KIND,
    INPUT_KIND,
    layer_operation,
    operation_operands,
)
from solar.schema_versions import OROJENESIS_MULTI_EINSUM_REGION_SCHEMA_VERSION


@dataclass(frozen=True)
class _ViewMetadata:
    target: str
    semantic: Mapping[str, Any]
    input_name: str
    output_name: str
    input_shape: list[int]
    output_shape: list[int]
    dtype: str


def _view_metadata(layer: Mapping[str, Any]) -> _ViewMetadata | None:
    semantic = layer_operation(layer)
    target = str(semantic.get("target", ""))
    if semantic.get("kind") in {INPUT_KIND, CONTRACTION_KIND} or target not in {
        "view",
        "transpose",
        "permute",
        "squeeze",
        "unsqueeze",
    }:
        return None
    effects = semantic.get("effects") or {}
    if (
        effects.get("mutates")
        or effects.get("atomic")
        or effects.get("opaque_library_call")
    ):
        return None
    names = layer.get("tensor_names") or {}
    shapes = layer.get("tensor_shapes") or {}
    dtypes = layer.get("tensor_dtypes") or {}
    input_names = [str(name) for name in names.get("inputs") or []]
    output_names = [str(name) for name in names.get("outputs") or []]
    input_shapes = [list(shape) for shape in shapes.get("inputs") or []]
    output_shapes = [list(shape) for shape in shapes.get("outputs") or []]
    input_dtypes = [str(dtype) for dtype in dtypes.get("inputs") or []]
    output_dtypes = [str(dtype) for dtype in dtypes.get("outputs") or []]
    if not (
        len(input_names) == len(output_names) == 1
        and len(input_shapes) == len(output_shapes) == 1
        and len(input_dtypes) == len(output_dtypes) == 1
        and input_dtypes[0] == output_dtypes[0]
        and _shape_product(input_shapes[0]) == _shape_product(output_shapes[0])
    ):
        return None
    aliases = effects.get("aliases")
    if (
        not isinstance(aliases, list)
        or len(aliases) != 1
        or not isinstance(aliases[0], Mapping)
    ):
        return None
    alias = aliases[0]
    if (
        alias.get("input") != 0
        or alias.get("output") != 0
        or alias.get("conditional") is not False
    ):
        return None
    return _ViewMetadata(
        target=target,
        semantic=semantic,
        input_name=input_names[0],
        output_name=output_names[0],
        input_shape=input_shapes[0],
        output_shape=output_shapes[0],
        dtype=input_dtypes[0],
    )


def _view_axis_map(metadata: _ViewMetadata) -> list[int] | None:
    target = metadata.target
    input_shape = metadata.input_shape
    output_shape = metadata.output_shape
    if target in {"view", "squeeze", "unsqueeze"}:
        input_flat = [_shape_product(input_shape[:-1]), input_shape[-1]]
        output_flat = [
            _shape_product(output_shape[:-1]),
            output_shape[-1],
        ]
        if input_flat != output_flat:
            return None
        return [0, 1]
    if len(input_shape) != 2 or len(output_shape) != 2:
        return None
    literal_arguments = [
        item.get("value")
        for item in operation_operands(metadata.semantic)
        if isinstance(item, Mapping) and "value" in item
    ]
    if target == "transpose":
        axis_map = _transpose_axis_map(literal_arguments)
    else:
        axis_map = _permutation_axis_map(literal_arguments)
    if axis_map is None:
        return None
    expected_output_shape = [input_shape[index] for index in axis_map]
    return axis_map if output_shape == expected_output_shape else None


def _transpose_axis_map(arguments: Sequence[Any]) -> list[int] | None:
    if len(arguments) != 2 or any(
        not isinstance(value, int) for value in arguments
    ):
        return None
    dimensions = [int(value) % 2 for value in arguments]
    if dimensions == [0, 1] or dimensions == [1, 0]:
        return [1, 0]
    if dimensions[0] == dimensions[1]:
        return [0, 1]
    return None


def _permutation_axis_map(arguments: Sequence[Any]) -> list[int] | None:
    permutation = arguments[-1] if arguments else None
    if (
        not isinstance(permutation, (list, tuple))
        or len(permutation) != 2
        or any(not isinstance(value, int) for value in permutation)
    ):
        return None
    axis_map = [int(value) % 2 for value in permutation]
    return axis_map if sorted(axis_map) == [0, 1] else None


def _internal_zero_copy_view(layer: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = _view_metadata(layer)
    if metadata is None:
        return None
    axis_map = _view_axis_map(metadata)
    if axis_map is None:
        return None
    return {
        "target": metadata.target,
        "input": metadata.input_name,
        "output": metadata.output_name,
        "input_shape": metadata.input_shape,
        "output_shape": metadata.output_shape,
        "dtype": metadata.dtype,
        "axis_map": axis_map,
    }


def _region_axis_map(
    producer: Mapping[str, Any],
    consumer: Mapping[str, Any],
    bridges: Sequence[str],
    views: Mapping[str, Mapping[str, Any]],
) -> list[int] | None:
    producer_shape = [int(producer["m"]), int(producer["n"])]
    consumer_shape = [int(consumer["m"]), int(consumer["k"])]
    axis_map = [0, 1]
    for bridge in bridges:
        bridge_map = list(views[bridge]["axis_map"])
        axis_map = [axis_map[index] for index in bridge_map]
    mapped_shape = [producer_shape[index] for index in axis_map]
    return axis_map if mapped_shape == consumer_shape else None


@dataclass(frozen=True)
class _RegionDiscovery:
    layers: dict[str, Mapping[str, Any]]
    producers: dict[str, str]
    consumers: dict[str, list[str]]
    descriptors: dict[str, dict[str, Any]]
    views: dict[str, dict[str, Any]]


def _region_discovery(
    layers: Mapping[str, Mapping[str, Any]],
) -> _RegionDiscovery:
    layer_map = {str(key): value for key, value in layers.items()}
    producers = {
        str(name): str(layer_id)
        for layer_id, layer in layer_map.items()
        for name in (layer.get("tensor_names") or {}).get("outputs") or []
    }
    consumers: dict[str, list[str]] = defaultdict(list)
    descriptors: dict[str, dict[str, Any]] = {}
    views: dict[str, dict[str, Any]] = {}
    for layer_id, layer in layer_map.items():
        for name in (layer.get("tensor_names") or {}).get("inputs") or []:
            consumers[str(name)].append(layer_id)
        with contextlib.suppress(OrojenesisError):
            descriptors[layer_id] = _region_matmul_descriptor(layer_id, layer)
        view = _internal_zero_copy_view(layer)
        if view is not None:
            views[layer_id] = view
    return _RegionDiscovery(
        layers=layer_map,
        producers=producers,
        consumers=dict(consumers),
        descriptors=descriptors,
        views=views,
    )


def _trace_region_tensor(
    discovery: _RegionDiscovery,
    tensor: str,
) -> tuple[str | None, list[str]]:
    path: list[str] = []
    current = str(tensor)
    seen: set[str] = set()
    while True:
        producer = discovery.producers.get(current)
        if producer is None or producer in seen:
            return None, []
        seen.add(producer)
        if producer in discovery.descriptors:
            return producer, list(reversed(path))
        if producer not in discovery.views:
            return producer, list(reversed(path))
        view = discovery.views[producer]
        if len(discovery.consumers.get(str(view["output"])) or []) != 1:
            return None, []
        path.append(producer)
        current = str(view["input"])


def _candidate_region_edges(
    discovery: _RegionDiscovery,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], set[str]]:
    edges: list[dict[str, Any]] = []
    entry_bridges: dict[str, list[str]] = {}
    valid_nodes = set(discovery.descriptors)
    for consumer_id, descriptor in discovery.descriptors.items():
        producer_id, bridges = _trace_region_tensor(
            discovery,
            str(descriptor["input"]),
        )
        if producer_id is None or producer_id not in discovery.descriptors:
            source = discovery.layers.get(str(producer_id), {})
            if str(source.get("type", "")).lower() != "start":
                valid_nodes.discard(consumer_id)
            else:
                entry_bridges[consumer_id] = bridges
            continue
        axis_map = _region_axis_map(
            discovery.descriptors[producer_id],
            descriptor,
            bridges,
            discovery.views,
        )
        if axis_map is None:
            valid_nodes.discard(consumer_id)
            continue
        edges.append(
            {
                "producer": producer_id,
                "consumer": consumer_id,
                "tensor": str(discovery.descriptors[producer_id]["output"]),
                "bridges": bridges,
                "axis_map": axis_map,
                "layer_path": [producer_id, *bridges, consumer_id],
            },
        )
    return edges, entry_bridges, valid_nodes


def _filtered_region_edges(
    edges: Sequence[Mapping[str, Any]],
    valid_nodes: set[str],
) -> list[dict[str, Any]]:
    return [
        dict(edge)
        for edge in edges
        if edge["producer"] in valid_nodes and edge["consumer"] in valid_nodes
    ]


def _region_edge_maps(
    edges: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    predecessors = {
        str(edge["consumer"]): str(edge["producer"]) for edge in edges
    }
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        successors[str(edge["producer"])].append(str(edge["consumer"]))
    return predecessors, dict(successors)


def _validate_region_endpoints(
    discovery: _RegionDiscovery,
    edges: Sequence[Mapping[str, Any]],
    entry_bridges: dict[str, list[str]],
    valid_nodes: set[str],
) -> None:
    predecessors, _ = _region_edge_maps(edges)
    for node_id in list(valid_nodes):
        descriptor = discovery.descriptors[node_id]
        weight_producer, weight_bridges = _trace_region_tensor(
            discovery,
            str(descriptor["weight"]),
        )
        weight_source = discovery.layers.get(str(weight_producer), {})
        if str(weight_source.get("type", "")).lower() != "start":
            valid_nodes.discard(node_id)
            continue
        descriptor["weight_bridges"] = weight_bridges
        if node_id in predecessors:
            continue
        activation_producer, bridges = _trace_region_tensor(
            discovery,
            str(descriptor["input"]),
        )
        activation_source = discovery.layers.get(
            str(activation_producer),
            {},
        )
        if str(activation_source.get("type", "")).lower() != "start":
            valid_nodes.discard(node_id)
        else:
            entry_bridges[node_id] = bridges


def _region_components(
    edges: Sequence[Mapping[str, Any]],
) -> list[set[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        left, right = str(edge["producer"]), str(edge["consumer"])
        neighbors[left].add(right)
        neighbors[right].add(left)
    components: list[set[str]] = []
    visited: set[str] = set()
    for seed in sorted(neighbors):
        if seed in visited:
            continue
        stack = [seed]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted(neighbors[node], reverse=True))
        visited.update(component)
        components.append(component)
    return components


def _component_schedule(
    component: set[str],
    predecessors: Mapping[str, str],
    successors: Mapping[str, Sequence[str]],
) -> tuple[list[str], list[str]] | None:
    roots = sorted(node for node in component if node not in predecessors)
    if len(roots) != 1:
        return None
    schedule: list[str] = []
    ready = list(roots)
    while ready:
        node = ready.pop(0)
        schedule.append(node)
        ready.extend(sorted(successors.get(node) or []))
    return (roots, schedule) if len(schedule) == len(component) else None


def _region_kind(
    component: set[str],
    descriptors: Mapping[str, Mapping[str, Any]],
    successors: Mapping[str, Sequence[str]],
) -> tuple[str, str]:
    if any(len(successors.get(node) or []) > 1 for node in component):
        return "matmul_fanout_tree", MULTI_EINSUM_FANOUT_COMPOSITION
    if any(descriptors[node]["kind"] == "batched_matmul" for node in component):
        return (
            "broadcast_batch_linear_matmul",
            MULTI_EINSUM_BATCH_COMPOSITION,
        )
    return "linear_matmul_with_axis_maps", MULTI_EINSUM_LAYOUT_COMPOSITION


def _region_physical_paths(
    schedule: Sequence[str],
    edges: Sequence[Mapping[str, Any]],
    descriptors: Mapping[str, Mapping[str, Any]],
    entry_bridges: Mapping[str, Sequence[str]],
) -> list[list[str]]:
    paths = [list(edge["layer_path"]) for edge in edges]
    for node in schedule:
        entry = list(entry_bridges.get(node) or [])
        if entry:
            paths.append([*entry, node])
        weight = list(descriptors[node].get("weight_bridges") or [])
        if weight:
            paths.append([*weight, node])
    return paths


def _component_region(
    component: set[str],
    edges: Sequence[Mapping[str, Any]],
    discovery: _RegionDiscovery,
    entry_bridges: Mapping[str, Sequence[str]],
    legacy_sets: set[tuple[str, ...]],
) -> dict[str, Any] | None:
    predecessors, successors = _region_edge_maps(edges)
    component_edges = [
        edge
        for edge in edges
        if edge["producer"] in component and edge["consumer"] in component
    ]
    if len(component) < 2 or len(component_edges) != len(component) - 1:
        return None
    schedule_result = _component_schedule(
        component,
        predecessors,
        successors,
    )
    if schedule_result is None:
        return None
    roots, schedule = schedule_result
    leaves = sorted(node for node in component if not successors.get(node))
    if any(
        discovery.consumers.get(str(discovery.descriptors[node]["output"]))
        for node in leaves
    ):
        return None
    if (
        tuple(schedule) in legacy_sets
        and all(not edge["bridges"] for edge in component_edges)
        and all(
            discovery.descriptors[node]["kind"] == "matmul"
            for node in component
        )
    ):
        return None
    kind, composition = _region_kind(
        component,
        discovery.descriptors,
        successors,
    )
    ordered_edges = [
        edge
        for producer in schedule
        for consumer in schedule
        for edge in component_edges
        if str(edge["producer"]) == producer
        and str(edge["consumer"]) == consumer
    ]
    return {
        "schema_version": (OROJENESIS_MULTI_EINSUM_REGION_SCHEMA_VERSION),
        "kind": kind,
        "composition": composition,
        "nodes": [discovery.descriptors[node] for node in schedule],
        "edges": ordered_edges,
        "roots": roots,
        "leaves": leaves,
        "schedule": schedule,
        "physical_paths": _region_physical_paths(
            schedule,
            component_edges,
            discovery.descriptors,
            entry_bridges,
        ),
    }


def find_multi_einsum_regions(
    layers: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Find endpoint-proven MatMul regions beyond the legacy direct chain."""
    discovery = _region_discovery(layers)
    edges, entry_bridges, valid_nodes = _candidate_region_edges(discovery)
    edges = _filtered_region_edges(edges, valid_nodes)
    _validate_region_endpoints(
        discovery,
        edges,
        entry_bridges,
        valid_nodes,
    )
    edges = _filtered_region_edges(edges, valid_nodes)
    legacy_sets = {
        tuple(chain) for chain in find_multi_einsum_chains(discovery.layers)
    }
    regions: list[dict[str, Any]] = []
    for component in _region_components(edges):
        region = _component_region(
            component,
            edges,
            discovery,
            entry_bridges,
            legacy_sets,
        )
        if region is not None:
            regions.append(region)
    return regions


def multi_einsum_region_problem(region: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a supported extended MatMul region."""
    schema_version = OROJENESIS_MULTI_EINSUM_REGION_SCHEMA_VERSION
    try:
        descriptor = json.loads(json.dumps(region, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise OrojenesisError(
            "multi-einsum region is not serializable",
        ) from exc
    if descriptor.get("schema_version") != schema_version:
        raise OrojenesisError("unsupported multi-einsum region schema")
    compositions = {
        "linear_matmul_with_axis_maps": MULTI_EINSUM_LAYOUT_COMPOSITION,
        "broadcast_batch_linear_matmul": MULTI_EINSUM_BATCH_COMPOSITION,
        "matmul_fanout_tree": MULTI_EINSUM_FANOUT_COMPOSITION,
    }
    if compositions.get(str(descriptor.get("kind"))) != descriptor.get(
        "composition",
    ):
        raise OrojenesisError("multi-einsum region composition mismatch")
    nodes = descriptor.get("nodes") or []
    schedule = [str(item) for item in descriptor.get("schedule") or []]
    node_ids = [str(node.get("id")) for node in nodes]
    if (
        len(nodes) < 2
        or schedule != node_ids
        or len(node_ids) != len(set(node_ids))
    ):
        raise OrojenesisError("multi-einsum region schedule is invalid")
    for node in nodes:
        if (
            str(node.get("kind")) not in {"matmul", "batched_matmul"}
            or any(int(node.get(name, 0)) <= 0 for name in ("m", "k", "n"))
            or not str(node.get("dtype", ""))
        ):
            raise OrojenesisError("multi-einsum region node is invalid")
    positions = {node_id: index for index, node_id in enumerate(schedule)}
    predecessors: dict[str, str] = {}
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in descriptor.get("edges") or []:
        producer = str(edge.get("producer"))
        consumer = str(edge.get("consumer"))
        axis_map = edge.get("axis_map")
        if (
            producer not in positions
            or consumer not in positions
            or positions[producer] >= positions[consumer]
            or consumer in predecessors
            or axis_map not in ([0, 1], [1, 0])
            or list(edge.get("layer_path") or [])[0:1] != [producer]
            or list(edge.get("layer_path") or [])[-1:] != [consumer]
        ):
            raise OrojenesisError("multi-einsum region edge is invalid")
        predecessors[consumer] = producer
        successors[producer].append(consumer)
    roots = sorted(
        node_id for node_id in schedule if node_id not in predecessors
    )
    leaves = sorted(
        node_id for node_id in schedule if not successors.get(node_id)
    )
    if (
        len(roots) != 1
        or len(predecessors) != len(nodes) - 1
        or descriptor.get("roots") != roots
        or descriptor.get("leaves") != leaves
    ):
        raise OrojenesisError("multi-einsum region is not an arborescence")
    if descriptor["composition"] != MULTI_EINSUM_FANOUT_COMPOSITION and any(
        len(items) > 1 for items in successors.values()
    ):
        raise OrojenesisError("linear multi-einsum region contains fan-out")
    if descriptor["composition"] == MULTI_EINSUM_BATCH_COMPOSITION and not any(
        node.get("kind") == "batched_matmul" for node in nodes
    ):
        raise OrojenesisError(
            "batched multi-einsum region has no batch dimension",
        )
    return descriptor


def multi_einsum_region_mapper_role(
    region: Mapping[str, Any],
    node_id: str,
) -> str:
    """Choose the pinned FFMT constraint variant for a region node."""
    descriptor = multi_einsum_region_problem(region)
    schedule = [str(item) for item in descriptor["schedule"]]
    edges = descriptor["edges"]
    predecessors = {
        str(edge["consumer"]): str(edge["producer"]) for edge in edges
    }
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        successors[str(edge["producer"])].append(str(edge["consumer"]))
    if node_id not in schedule:
        raise OrojenesisError("multi-einsum region mapper node is unknown")
    if node_id not in predecessors:
        return "first"
    if not successors.get(node_id):
        return "second_last" if len(schedule) == 2 else "last"
    parent = predecessors[node_id]
    if parent not in predecessors:
        return "second"
    return "middle"


def compose_multi_einsum_region_curve(
    region: Mapping[str, Any],
    raw_paths: Mapping[str, Sequence[str | Path]],
    *,
    row_tiles_by_node: Mapping[str, Sequence[int]],
    word_bytes: int,
) -> list[dict[str, Any]]:
    """Compose replayable mapping assignments for a linear or fan-out region."""
    descriptor = multi_einsum_region_problem(region)
    return _compose_multi_einsum_region_curve(
        descriptor,
        raw_paths,
        row_tiles_by_node=row_tiles_by_node,
        word_bytes=word_bytes,
    )
