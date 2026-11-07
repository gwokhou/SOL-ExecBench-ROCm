# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Translate validated SOLAR layer mappings into performance descriptors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from sol_execbench.core.bench.performance_model.models import (
    CompositeGraphDescriptor,
    CompositeGraphEdge,
    CompositeGraphNode,
    FusionRegion,
    MatmulDescriptor,
    ReductionDescriptor,
    ReductionOperation,
    SoftmaxDescriptor,
    TensorDType,
    UnsupportedDescriptor,
    WorkloadKind,
)
from sol_execbench.core.solar_bridge.semantic_primitives import (
    ELEMENTWISE_TYPES,
    INDEXED_READ_TYPES,
    INDEXED_UPDATE_TYPES,
    MATMUL_TYPES,
    REDUCTION_TYPES,
    SOFTMAX_TYPES,
    TRANSPOSE_TYPES,
    cross_entropy_descriptor,
    decomposed_cross_entropy_descriptor,
    elementwise_descriptor,
    indexed_read_descriptor,
    indexed_update_descriptor,
    matmul_descriptor,
    primitive_layer_descriptor,
    reduction_descriptor,
    softmax_descriptor,
    tensor_names,
    transpose_descriptor,
)

_IGNORED_TYPES = frozenset(
    {
        "",
        "start",
        "input",
        "clone",
        "contiguous",
        "expand",
        "getitem",
        "identity",
        "reshape",
        "squeeze",
        "unsqueeze",
        "view",
    }
)
_MAX_COMPOSITE_NODES = 32


@dataclass(frozen=True, slots=True, kw_only=True)
class DescriptorClassification:
    """One successful semantic descriptor classification."""

    descriptor: object
    workload_kind: WorkloadKind
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class _DescriptorRequest:
    layers: Mapping[str, object]
    metadata: Mapping[str, object]
    operation_layers: list[tuple[Mapping[str, object], str]]
    operation_types: tuple[str, ...]


class DescriptorRule(Protocol):
    """Ordered classifier rule for one descriptor family."""

    def __call__(
        self, request: _DescriptorRequest
    ) -> DescriptorClassification | None: ...


def semantic_descriptor(
    layers: Mapping[str, object],
    metadata: Mapping[str, object],
) -> tuple[object, WorkloadKind, list[str]]:
    """Classify validated SOLAR layers into one typed descriptor family."""
    operation_layers = _operation_layers(layers)
    request = _DescriptorRequest(
        layers=layers,
        metadata=metadata,
        operation_layers=operation_layers,
        operation_types=tuple(item[1] for item in operation_layers),
    )
    classification = next(
        (result for rule in _DESCRIPTOR_RULES if (result := rule(request))),
        _unsupported_rule(request),
    )
    return (
        classification.descriptor,
        classification.workload_kind,
        list(classification.reason_codes),
    )


def _classification(
    descriptor: object | None,
    kind: WorkloadKind,
) -> DescriptorClassification | None:
    if descriptor is None:
        return None
    return DescriptorClassification(descriptor=descriptor, workload_kind=kind)


def _special_rule(
    request: _DescriptorRequest,
) -> DescriptorClassification | None:
    special = _special_graph_descriptor(
        request.layers, request.operation_layers, request.metadata
    )
    if special is None:
        return None
    descriptor, kind, _reasons = special
    return _classification(descriptor, kind)


def _elementwise_rule(
    request: _DescriptorRequest,
) -> DescriptorClassification | None:
    if not request.operation_types or not (
        set(request.operation_types) <= ELEMENTWISE_TYPES
    ):
        return None
    return _classification(
        elementwise_descriptor(request.operation_layers),
        WorkloadKind.ELEMENTWISE,
    )


def _single_operation_rule(
    request: _DescriptorRequest,
) -> DescriptorClassification | None:
    if len(request.operation_layers) != 1:
        return None
    layer = request.operation_layers[0][0]
    operation = request.operation_types[0]
    if operation in TRANSPOSE_TYPES:
        return _classification(
            transpose_descriptor(layer), WorkloadKind.TRANSPOSE
        )
    if operation in REDUCTION_TYPES:
        return _classification(
            reduction_descriptor(layer, REDUCTION_TYPES[operation]),
            WorkloadKind.REDUCTION,
        )
    if operation in MATMUL_TYPES:
        return _classification(
            matmul_descriptor(layer, batched=operation == "bmm"),
            WorkloadKind.MATMUL,
        )
    if operation in SOFTMAX_TYPES:
        return _classification(
            softmax_descriptor(layer, SOFTMAX_TYPES[operation]),
            WorkloadKind.SOFTMAX,
        )
    return None


def _decomposed_cross_entropy_rule(
    request: _DescriptorRequest,
) -> DescriptorClassification | None:
    decomposed_cross_entropy = decomposed_cross_entropy_descriptor(
        request.operation_layers
    )
    return _classification(decomposed_cross_entropy, WorkloadKind.CROSS_ENTROPY)


def _native_cross_entropy_rule(
    request: _DescriptorRequest,
) -> DescriptorClassification | None:
    if len(request.operation_layers) != 1 or request.operation_types != (
        "cross_entropy",
    ):
        return None
    return _classification(
        cross_entropy_descriptor(request.operation_layers[0][0]),
        WorkloadKind.CROSS_ENTROPY,
    )


def _indexed_rule(
    request: _DescriptorRequest,
) -> DescriptorClassification | None:
    if len(request.operation_layers) != 1:
        return None
    layer = request.operation_layers[0][0]
    operation = request.operation_types[0]
    if operation in INDEXED_READ_TYPES:
        return _classification(
            indexed_read_descriptor(layer, INDEXED_READ_TYPES[operation]),
            WorkloadKind.INDEXED_READ,
        )
    if operation in INDEXED_UPDATE_TYPES:
        return _classification(
            indexed_update_descriptor(layer, INDEXED_UPDATE_TYPES[operation]),
            WorkloadKind.INDEXED_UPDATE,
        )
    return None


def _composite_rule(
    request: _DescriptorRequest,
) -> DescriptorClassification | None:
    composite = _composite_descriptor(
        request.layers, request.operation_layers, request.metadata
    )
    if composite is None:
        return None
    kind = {
        "transformer_block": WorkloadKind.TRANSFORMER,
        "concurrent_graph": WorkloadKind.CONCURRENT,
    }.get(composite.graph_class, WorkloadKind.COMPOSITE)
    return _classification(composite, kind)


def _unsupported_rule(_request: _DescriptorRequest) -> DescriptorClassification:
    reasons = ("unsupported_workload_descriptor",)
    return DescriptorClassification(
        descriptor=UnsupportedDescriptor(reason_codes=list(reasons)),
        workload_kind=WorkloadKind.UNSUPPORTED,
        reason_codes=reasons,
    )


_DESCRIPTOR_RULES: tuple[DescriptorRule, ...] = (
    _special_rule,
    _elementwise_rule,
    _single_operation_rule,
    _decomposed_cross_entropy_rule,
    _native_cross_entropy_rule,
    _indexed_rule,
    _composite_rule,
)


def _special_graph_descriptor(
    layers: Mapping[str, object],
    operation_layers: list[tuple[Mapping[str, object], str]],
    metadata: Mapping[str, object],
) -> tuple[object, WorkloadKind, list[str]] | None:
    special_graph = _declared_special_graph(metadata)
    if special_graph is None:
        return None
    composite = _composite_descriptor(layers, operation_layers, metadata)
    if composite is None or composite.graph_class != special_graph:
        return None
    kind = (
        WorkloadKind.TRANSFORMER
        if special_graph == "transformer_block"
        else WorkloadKind.CONCURRENT
    )
    return composite, kind, []


def _declared_special_graph(
    metadata: Mapping[str, object],
) -> Literal["transformer_block", "concurrent_graph"] | None:
    semantics = metadata.get("performance_semantics")
    if not isinstance(semantics, Mapping):
        return None
    graph_class = semantics.get("graph_class")
    if graph_class == "transformer_block":
        return "transformer_block"
    if graph_class == "concurrent_graph":
        return "concurrent_graph"
    return None


def _operation_layers(
    layers: Mapping[str, object],
) -> list[tuple[Mapping[str, object], str]]:
    result: list[tuple[Mapping[str, object], str]] = []
    for value in layers.values():
        if not isinstance(value, Mapping):
            continue
        layer = cast("Mapping[str, object]", value)
        operation = str(layer.get("type", "")).lower()
        alias_only = _is_metadata_or_alias_only(layer)
        if operation not in _IGNORED_TYPES and (
            not alias_only or operation in TRANSPOSE_TYPES
        ):
            result.append((layer, operation))
    return result


def _is_metadata_or_alias_only(layer: Mapping[str, object]) -> bool:
    resources = layer.get("resources")
    return (
        isinstance(resources, Mapping)
        and resources.get("classification") == "exempt"
        and resources.get("exemption_reason") == "metadata_or_alias_only"
    )


def _composite_descriptor(
    layers: Mapping[str, object],
    operation_layers: list[tuple[Mapping[str, object], str]],
    metadata: Mapping[str, object],
) -> CompositeGraphDescriptor | None:
    if not 2 <= len(operation_layers) <= _MAX_COMPOSITE_NODES:
        return None
    operation_names = [
        name
        for name, layer in layers.items()
        if any(layer is item for item, _operation in operation_layers)
    ]
    if len(operation_names) != len(operation_layers):
        return None
    nodes: list[CompositeGraphNode] = []
    for name, (layer, operation) in zip(
        operation_names,
        operation_layers,
        strict=True,
    ):
        descriptor = primitive_layer_descriptor(layer, operation)
        inputs = tensor_names(layer, "inputs")
        outputs = tensor_names(layer, "outputs")
        if descriptor is None or not outputs:
            return None
        nodes.append(
            CompositeGraphNode(
                node_id=name,
                layer_names=[name],
                descriptor=descriptor,
                input_tensors=inputs,
                output_tensors=outputs,
            )
        )
    regions = fusion_regions(metadata.get("fusion"))
    edges = _composite_edges(layers, operation_names, regions)
    if edges is None:
        return None
    schedule = _topological_schedule(operation_names, edges)
    if schedule is None:
        return None
    return CompositeGraphDescriptor(
        graph_class=_composite_graph_class(metadata, nodes),
        nodes=nodes,
        edges=edges,
        schedule=schedule,
    )


def _composite_edges(
    layers: Mapping[str, object],
    operation_names: list[str],
    regions: list[FusionRegion],
) -> list[CompositeGraphEdge] | None:
    admitted = set(operation_names)
    region_by_layer = {
        layer_name: region.region_id
        for region in regions
        for layer_name in region.layer_names
    }
    edges: list[CompositeGraphEdge] = []
    for consumer in operation_names:
        predecessors = _admitted_predecessors(
            layers,
            consumer,
            admitted,
        )
        if predecessors is None:
            return None
        for producer in predecessors:
            producer_layer = layers[producer]
            if not isinstance(producer_layer, Mapping):
                return None
            tensors = tensor_names(
                cast("Mapping[str, object]", producer_layer),
                "outputs",
            )
            if len(tensors) != 1:
                return None
            edges.append(
                CompositeGraphEdge(
                    producer=producer,
                    consumer=consumer,
                    tensor=tensors[0],
                    materialized=(
                        region_by_layer.get(producer)
                        != region_by_layer.get(consumer)
                    ),
                )
            )
    identities = {(edge.producer, edge.consumer, edge.tensor) for edge in edges}
    return edges if len(identities) == len(edges) else None


def _admitted_predecessors(
    layers: Mapping[str, object],
    consumer: str,
    admitted: set[str],
) -> list[str] | None:
    """Resolve operation predecessors through transparent IR-only layers."""
    pending = [consumer]
    visited = {consumer}
    resolved: list[str] = []
    while pending:
        current = pending.pop()
        layer = layers.get(current)
        if not isinstance(layer, Mapping):
            return None
        connections = layer.get("connections")
        if not isinstance(connections, Mapping):
            return None
        raw = connections.get("inputs")
        if not isinstance(raw, list):
            return None
        for predecessor in (str(item) for item in raw):
            if predecessor in admitted:
                resolved.append(predecessor)
            elif predecessor in layers and predecessor not in visited:
                visited.add(predecessor)
                pending.append(predecessor)
    return list(dict.fromkeys(resolved))


def _topological_schedule(
    node_ids: list[str],
    edges: list[CompositeGraphEdge],
) -> list[str] | None:
    incoming = dict.fromkeys(node_ids, 0)
    successors = {node_id: [] for node_id in node_ids}
    for edge in edges:
        incoming[edge.consumer] += 1
        successors[edge.producer].append(edge.consumer)
    roots = sorted(node_id for node_id, count in incoming.items() if count == 0)
    if len(roots) != 1:
        return None
    ready = roots
    schedule: list[str] = []
    while ready:
        node_id = ready.pop(0)
        schedule.append(node_id)
        for successor in sorted(successors[node_id]):
            incoming[successor] -= 1
            if incoming[successor] == 0:
                ready.append(successor)
        ready.sort()
    return schedule if len(schedule) == len(node_ids) else None


def _is_minigpt_semantics(
    metadata: Mapping[str, object],
    nodes: list[CompositeGraphNode],
) -> bool:
    semantics = metadata.get("performance_semantics")
    if not isinstance(semantics, Mapping):
        return False
    values = cast("Mapping[str, object]", semantics)
    sequence_length = values.get("sequence_length")
    descriptors = [node.descriptor for node in nodes]
    return (
        values.get("graph_class") == "transformer_block"
        and values.get("hidden_size") == 768
        and values.get("num_heads") == 8
        and isinstance(sequence_length, int)
        and not isinstance(sequence_length, bool)
        and 0 < sequence_length <= 1024
        and values.get("dtype") in {"float32", "torch.float32"}
        and any(
            isinstance(descriptor, MatmulDescriptor)
            and descriptor.input_dtype is TensorDType.FLOAT32
            and 768 in {descriptor.m, descriptor.n, descriptor.k}
            for descriptor in descriptors
        )
        and any(
            isinstance(descriptor, SoftmaxDescriptor)
            and descriptor.reduction_width <= 1024
            for descriptor in descriptors
        )
        and any(
            isinstance(descriptor, ReductionDescriptor)
            and descriptor.operation is ReductionOperation.LAYER_NORM
            and descriptor.reduction_width == 768
            for descriptor in descriptors
        )
    )


def _composite_graph_class(
    metadata: Mapping[str, object],
    nodes: list[CompositeGraphNode],
) -> Literal["composite_graph", "transformer_block", "concurrent_graph"]:
    if _is_minigpt_semantics(metadata, nodes):
        return "transformer_block"
    semantics = metadata.get("performance_semantics")
    if (
        isinstance(semantics, Mapping)
        and semantics.get("graph_class") == "concurrent_graph"
    ):
        return "concurrent_graph"
    return "composite_graph"


def fusion_regions(raw_fusion: object) -> list[FusionRegion]:
    """Parse declared fusion regions from validated analysis metadata."""
    if not isinstance(raw_fusion, Mapping):
        return []
    raw_regions = raw_fusion.get("regions")
    if not isinstance(raw_regions, list):
        return []
    return [
        _fusion_region(cast("Mapping[object, object]", region), index)
        for index, region in enumerate(raw_regions)
        if isinstance(region, Mapping)
    ]


def _fusion_region(region: Mapping[object, object], index: int) -> FusionRegion:
    raw_layers = region.get("layers")
    return FusionRegion(
        region_id=str(region.get("id") or f"semantic_region_{index}"),
        layer_names=(
            [str(layer) for layer in raw_layers]
            if isinstance(raw_layers, list)
            else []
        ),
    )


def fusion_regions_cover_operations(
    layers: Mapping[str, object],
    regions: list[FusionRegion],
) -> bool:
    """Return whether fusion regions cite every operation exactly once."""
    operation_names = {
        name
        for name, layer in layers.items()
        if isinstance(layer, Mapping)
        and str(layer.get("type", "")).lower() not in _IGNORED_TYPES
    }
    cited = [
        layer_name
        for region in regions
        for layer_name in region.layer_names
        if layer_name in operation_names
    ]
    return set(cited) == operation_names and len(cited) == len(set(cited))


__all__ = [
    "fusion_regions",
    "fusion_regions_cover_operations",
    "semantic_descriptor",
]
