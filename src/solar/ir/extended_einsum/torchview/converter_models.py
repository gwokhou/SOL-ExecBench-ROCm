"""Shared contracts for the Torchview extended-einsum converter."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

PathLike = str | Path


class RawConnections(TypedDict):
    """Untrusted connection fields accepted from Torchview YAML."""

    inputs: NotRequired[list[str]]
    outputs: NotRequired[list[str]]


class ConversionLayer(TypedDict, total=False):
    """Normalized mutable layer representation used during conversion."""

    type: str
    node_class: str
    input_shapes: list[list[int]]
    output_shapes: list[list[int]]
    input_dtypes: list[str]
    output_dtypes: list[str]
    input_types: list[str]
    output_types: list[str]
    module_args: dict[str, Any]
    connections: RawConnections
    weight_nodes: list[str]
    weight_shapes: list[list[int]]
    output_slots: list[dict[str, Any]]
    source_input_index: int


class ConversionGraph(TypedDict, total=False):
    """Typed top-level shape of a graph moving through conversion."""

    schema_version: str
    model_name: str
    layers: dict[str, ConversionLayer]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class ConversionConfig:
    """Immutable converter configuration shared by one façade."""

    debug: bool = False
    enable_agent: bool = False
    api_key: str | None = None
    cache_dir: str = "./solar_handlers_cache"
    strict: bool = False


@dataclass(slots=True, kw_only=True)
class ConversionState:
    """Per-conversion mutable producer metadata."""

    tensor_to_producer: dict[str, str] = field(default_factory=dict)
    tensor_to_producer_slot: dict[str, int] = field(default_factory=dict)


class ConversionError(ValueError):
    """The PyTorch graph cannot be converted without approximation."""


def _string_list(value: object, *, field_name: str) -> list[str]:
    """Normalize one untrusted sequence of identifiers."""
    if value is None:
        return []
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ConversionError(f"{field_name} must be a list of strings")
    return [cast("str", item) for item in value]


def _shape_list(value: object, *, field_name: str) -> list[list[int]]:
    """Normalize one untrusted list of integer tensor shapes."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConversionError(f"{field_name} must be a list of shapes")
    shapes: list[list[int]] = []
    for shape in value:
        if not isinstance(shape, list) or not all(
            isinstance(dimension, int) for dimension in shape
        ):
            raise ConversionError(
                f"{field_name} entries must be lists of integers"
            )
        shapes.append([cast("int", dimension) for dimension in shape])
    return shapes


def normalize_conversion_graph(payload: object) -> ConversionGraph:
    """Validate raw YAML/JSON and return the converter's typed boundary."""
    if not isinstance(payload, Mapping):
        raise ConversionError("graph payload must be a mapping")
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, Mapping):
        raise ConversionError("graph payload must contain a layers mapping")

    graph = dict(payload)
    layers: dict[str, ConversionLayer] = {}
    for layer_id, raw_layer in raw_layers.items():
        if not isinstance(layer_id, str) or not isinstance(raw_layer, Mapping):
            raise ConversionError(
                "layer ids and layer payloads must be mappings"
            )
        layer = dict(raw_layer)
        for field_name in (
            "input_dtypes",
            "output_dtypes",
            "input_types",
            "output_types",
            "weight_nodes",
        ):
            layer[field_name] = _string_list(
                raw_layer.get(field_name), field_name=field_name
            )
        for field_name in ("input_shapes", "output_shapes", "weight_shapes"):
            layer[field_name] = _shape_list(
                raw_layer.get(field_name), field_name=field_name
            )
        module_args = raw_layer.get("module_args")
        if module_args is not None and not isinstance(module_args, Mapping):
            raise ConversionError("module_args must be a mapping")
        layer["module_args"] = dict(module_args or {})
        connections = raw_layer.get("connections")
        if connections is not None and not isinstance(connections, Mapping):
            raise ConversionError("connections must be a mapping")
        connection_mapping = connections or {}
        layer["connections"] = {
            "inputs": _string_list(
                connection_mapping.get("inputs"),
                field_name="connections.inputs",
            ),
            "outputs": _string_list(
                connection_mapping.get("outputs"),
                field_name="connections.outputs",
            ),
        }
        for field_name in ("type", "node_class"):
            value = raw_layer.get(field_name)
            if value is not None and not isinstance(value, str):
                raise ConversionError(f"{field_name} must be a string")
        layers[layer_id] = cast("ConversionLayer", layer)
    graph["layers"] = layers
    return cast("ConversionGraph", graph)


@dataclass(frozen=True, slots=True, kw_only=True)
class ConvertedTensorMetadata:
    """Normalized tensor metadata collected while converting one layer."""

    tensor_names: dict[str, list[str]]
    tensor_types: dict[str, list[str]]
    tensor_shapes: dict[str, list[list[int]]]
    tensor_dtypes: dict[str, Any]
    activation_connections: list[str]
    output_connections: list[str]
    additional_info: dict[str, Any]


__all__ = [
    "ConversionConfig",
    "ConversionError",
    "ConversionGraph",
    "ConversionLayer",
    "ConversionState",
    "ConvertedTensorMetadata",
    "PathLike",
    "RawConnections",
    "normalize_conversion_graph",
]
