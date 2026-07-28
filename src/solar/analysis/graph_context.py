"""Typed preparation and topology primitives for graph analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from solar.analysis.graph_rules import (
    BOOL_DTYPES,
    QUANTIZED_PAYLOAD_PASSTHROUGH,
    TRANSPARENT_OPS,
)
from solar.common.types import DynamicValue
from solar.ir.contracts import layer_operation
from solar.rocm.architecture import ArchitectureProfile

PathLike = str | Path


def product(shape: list[int]) -> int:
    """Return the product of concrete tensor dimensions."""
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return int(result)


@dataclass(frozen=True, slots=True)
class AnalysisJob:
    """Inputs controlling one graph-analysis run."""

    graph_path: PathLike
    output_dir: PathLike
    precision: str
    copy_graph: bool
    strict: bool
    architecture: str | Path | ArchitectureProfile | None
    orojenesis_runner: DynamicValue | None
    require_orojenesis: bool


@dataclass(frozen=True, slots=True)
class PreparedAnalysis:
    """Validated graph inputs and resolved analysis configuration."""

    source: Path
    output_dir: Path
    graph: dict[str, DynamicValue]
    all_layers: dict[str, DynamicValue]
    declared_graph_outputs: set[str]
    semantic_graph: bool
    semantic_complete: bool
    strict: bool
    requested_precision: str
    fallback_precision: str
    element_size: float
    profile: ArchitectureProfile | None


@dataclass(frozen=True, slots=True)
class GraphTopology:
    """Producer, consumer, and layer classifications for an IR graph."""

    layers: dict[str, DynamicValue]
    start_node_ids: set[str]
    bool_start_node_ids: set[str]
    all_layer_ids: set[str]
    transparent_layer_ids: set[str]
    tensor_producers: dict[str, str]
    tensor_consumers: dict[str, set[str]]
    intermediate_tensors: set[str]
    bool_layers: set[str]
    dead_end_layers: set[str]

    def trace_source_through_views(self, layer_id: str) -> str:
        """Trace backward through transparent view layers to the real source."""
        visited: set[str] = set()
        current = layer_id
        while current in self.transparent_layer_ids and current not in visited:
            visited.add(current)
            connections = (self.layers[current].get("connections") or {}).get(
                "inputs",
            ) or []
            if not connections:
                break
            current = connections[0]
        return current

    def has_real_consumer(self, layer_id: str) -> bool:
        """Return whether an output reaches a non-transparent graph layer."""
        visited: set[str] = set()
        queue = [layer_id]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            connections = (
                self.layers.get(current, {}).get("connections") or {}
            ).get("outputs") or []
            for output_id in connections:
                if output_id in self.transparent_layer_ids:
                    queue.append(output_id)
                elif output_id in self.all_layer_ids:
                    return True
        return False

    def source_is_orphan(self, connection_id: str) -> bool:
        """Return whether a connection traces to no graph or start node."""
        source = (
            self.trace_source_through_views(connection_id)
            if connection_id in self.transparent_layer_ids
            else connection_id
        )
        return (
            source not in self.all_layer_ids
            and source not in self.start_node_ids
        )

    def dequantized_payload_precision(
        self,
        tensor_name: str,
        profile: ArchitectureProfile | None,
        fallback_precision: str,
        visited: set[str] | None = None,
    ) -> str | None:
        """Trace one contraction payload to an exact low-precision cast."""
        if (
            profile is None
            or (producer_id := self.tensor_producers.get(tensor_name)) is None
        ):
            return None
        seen = set(visited or ())
        if producer_id in seen:
            return None
        seen.add(producer_id)
        producer = self.layers[producer_id]
        semantic = layer_operation(producer)
        target = str(semantic.get("target", ""))
        names = (producer.get("tensor_names") or {}).get("inputs") or []
        dtypes = producer.get("tensor_dtypes") or {}
        inputs = list(dtypes.get("inputs") or [])
        outputs = list(dtypes.get("outputs") or [])
        if target == "to" and inputs and outputs:
            source = profile.tensor_precision(inputs[0], fallback_precision)
            destination = profile.tensor_precision(
                outputs[0],
                fallback_precision,
            )
            return (
                source
                if source in {"fp8", "int8", "int4"} and source != destination
                else None
            )
        if target in QUANTIZED_PAYLOAD_PASSTHROUGH and names:
            return self.dequantized_payload_precision(
                str(names[0]),
                profile,
                fallback_precision,
                seen,
            )
        if target == "mul":
            candidates = {
                candidate
                for name in names
                if (
                    candidate := self.dequantized_payload_precision(
                        str(name),
                        profile,
                        fallback_precision,
                        seen,
                    )
                )
                is not None
            }
            if len(candidates) == 1:
                return candidates.pop()
        return None


def build_graph_topology(
    all_layers: dict[str, DynamicValue],
) -> GraphTopology:
    """Build immutable producer and consumer indices for graph layers."""
    layers, start_ids, bool_start_ids = _partition_graph_layers(all_layers)
    all_layer_ids = set(layers)
    transparent_ids = {
        layer_id
        for layer_id, layer in layers.items()
        if str(layer.get("type", "")).lower() in TRANSPARENT_OPS
    }
    producers, consumers = _build_tensor_indices(layers)
    topology = GraphTopology(
        layers=layers,
        start_node_ids=start_ids,
        bool_start_node_ids=bool_start_ids,
        all_layer_ids=all_layer_ids,
        transparent_layer_ids=transparent_ids,
        tensor_producers=producers,
        tensor_consumers=consumers,
        intermediate_tensors={
            name for name in producers if bool(consumers.get(name))
        },
        bool_layers=_propagate_bool_layers(layers, bool_start_ids),
        dead_end_layers=set(),
    )
    return replace(
        topology,
        dead_end_layers={
            layer_id
            for layer_id in layers
            if not topology.has_real_consumer(layer_id)
        },
    )


def _partition_graph_layers(
    all_layers: dict[str, DynamicValue],
) -> tuple[
    dict[str, DynamicValue],
    set[str],
    set[str],
]:
    layers: dict[str, DynamicValue] = {}
    start_node_ids: set[str] = set()
    bool_start_node_ids: set[str] = set()
    for layer_id, layer in all_layers.items():
        if str(layer.get("type", "")).lower() != "start":
            layers[layer_id] = layer
            continue
        start_node_ids.add(layer_id)
        output_dtypes = (layer.get("tensor_dtypes") or {}).get("outputs") or []
        if output_dtypes and all(
            str(dtype) in BOOL_DTYPES for dtype in output_dtypes
        ):
            bool_start_node_ids.add(layer_id)
    return layers, start_node_ids, bool_start_node_ids


def _build_tensor_indices(
    layers: dict[str, DynamicValue],
) -> tuple[dict[str, str], dict[str, set[str]]]:
    producers: dict[str, str] = {}
    consumers: dict[str, set[str]] = {}
    for layer_id, layer in layers.items():
        names = layer.get("tensor_names") or {}
        for output_name in names.get("outputs") or []:
            producers[output_name] = layer_id
    for layer_id, layer in layers.items():
        names = layer.get("tensor_names") or {}
        for input_name in names.get("inputs") or []:
            if input_name in producers:
                consumers.setdefault(input_name, set()).add(layer_id)
    return producers, consumers


def _propagate_bool_layers(
    layers: dict[str, DynamicValue],
    bool_start_node_ids: set[str],
) -> set[str]:
    bool_layers: set[str] = set()
    changed = bool(bool_start_node_ids)
    while changed:
        changed = False
        for layer_id, layer in layers.items():
            if layer_id in bool_layers:
                continue
            input_ids = list(
                (layer.get("connections") or {}).get("inputs") or [],
            )
            if input_ids and all(
                item in bool_start_node_ids or item in bool_layers
                for item in input_ids
            ):
                bool_layers.add(layer_id)
                changed = True
    return bool_layers
