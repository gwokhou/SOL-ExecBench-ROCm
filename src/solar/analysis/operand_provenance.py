# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Prove graph-external provenance for contraction operands."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from solar.analysis.graph_rules import (
    LOW_PRECISION_DEQUANT_DTYPES,
    RECOMPUTABLE_OPERAND_TARGETS,
    TILE_LOCAL_SCALAR_POSTPROCESS_TARGETS,
    TILE_RECOMPUTABLE_OPERAND_TARGETS,
)
from solar.ir.contracts import CONTRACTION_KIND, INPUT_KIND, layer_operation
from solar.types import NodeDict

type SourceTrace = tuple[set[str], bool, bool]


@dataclass(frozen=True, slots=True)
class _ProducedTensor:
    layer: NodeDict
    output_index: int


class _RegionOutputTracer:
    """Prove a contraction output reaches a region boundary tile-locally."""

    def __init__(
        self,
        region: Mapping[str, Any],
        layers: Mapping[str, NodeDict],
    ) -> None:
        region_layers = {str(item) for item in region.get("layers") or []}
        self._external_outputs = {
            str(item) for item in region.get("external_outputs") or []
        }
        self._layers = {
            layer_id: layer
            for layer_id, layer in layers.items()
            if layer_id in region_layers
        }
        self._consumers: dict[str, list[NodeDict]] = {}
        for layer in self._layers.values():
            for name in (layer.get("tensor_names") or {}).get("inputs") or []:
                self._consumers.setdefault(str(name), []).append(layer)

    @staticmethod
    def _safe_pointwise_step(
        layer: NodeDict,
        input_name: str,
        shape: list[Any],
        dtype: str,
    ) -> str | None:
        semantic = layer_operation(layer)
        effects = semantic.get("effects") or {}
        if (
            semantic.get("target") not in TILE_LOCAL_SCALAR_POSTPROCESS_TARGETS
            or effects.get("mutates")
            or effects.get("aliases")
            or effects.get("atomic")
            or effects.get("opaque_library_call")
        ):
            return None
        names = layer.get("tensor_names") or {}
        shapes = layer.get("tensor_shapes") or {}
        dtypes = layer.get("tensor_dtypes") or {}
        inputs = [str(item) for item in names.get("inputs") or []]
        outputs = [str(item) for item in names.get("outputs") or []]
        if inputs != [input_name] or len(outputs) != 1:
            return None
        if (
            list(shapes.get("inputs") or []) != [shape]
            or list(shapes.get("outputs") or []) != [shape]
            or [str(item) for item in dtypes.get("inputs") or []] != [dtype]
            or [str(item) for item in dtypes.get("outputs") or []] != [dtype]
        ):
            return None
        return outputs[0]

    def trace(self, output_name: str, shape: list[Any], dtype: str) -> bool:
        visited: set[str] = set()
        current = output_name
        while current not in self._external_outputs:
            if current in visited:
                return False
            visited.add(current)
            consumers = self._consumers.get(current) or []
            if len(consumers) != 1:
                return False
            next_name = self._safe_pointwise_step(
                consumers[0], current, shape, dtype
            )
            if next_name is None:
                return False
            current = next_name
        return True


class _OperandSourceTracer:
    def __init__(self, layers: dict[str, NodeDict]) -> None:
        self._producers = {
            str(name): _ProducedTensor(producer, output_index)
            for producer in layers.values()
            for output_index, name in enumerate(
                (producer.get("tensor_names") or {}).get("outputs") or [],
            )
        }

    def _trace_alias(
        self,
        aliases: list[NodeDict],
        input_names: list[str],
        visited: set[str],
    ) -> SourceTrace | None:
        input_index = int(aliases[0].get("input", -1))
        if input_index not in range(len(input_names)):
            return None
        return self.trace(str(input_names[input_index]), visited)

    @staticmethod
    def _is_recomputable(
        semantic: Mapping[str, Any], effects: NodeDict
    ) -> bool:
        return bool(
            semantic.get("kind") not in {INPUT_KIND, CONTRACTION_KIND}
            and semantic.get("target") in RECOMPUTABLE_OPERAND_TARGETS
            and not effects.get("mutates")
            and not effects.get("atomic")
            and not effects.get("opaque_library_call"),
        )

    @staticmethod
    def _is_low_precision_dequantization(producer: NodeDict) -> bool:
        dtypes = producer.get("tensor_dtypes") or {}
        inputs = [
            str(item).removeprefix("torch.")
            for item in dtypes.get("inputs") or []
        ]
        outputs = [
            str(item).removeprefix("torch.")
            for item in dtypes.get("outputs") or []
        ]
        return bool(
            inputs
            and outputs
            and inputs[0] in LOW_PRECISION_DEQUANT_DTYPES
            and inputs[0] != outputs[0],
        )

    def _trace_recomputation(
        self,
        producer: NodeDict,
        semantic: Mapping[str, Any],
        input_names: list[str],
        visited: set[str],
    ) -> SourceTrace | None:
        sources: set[str] = set()
        saw_low_precision = False
        materialized = (
            semantic.get("target") not in TILE_RECOMPUTABLE_OPERAND_TARGETS
        )
        for input_name in input_names:
            traced = self.trace(str(input_name), visited)
            if traced is None:
                return None
            traced_sources, traced_low_precision, traced_materialized = traced
            sources.update(traced_sources)
            saw_low_precision |= traced_low_precision
            materialized |= traced_materialized
        if semantic.get("target") == "to":
            saw_low_precision |= self._is_low_precision_dequantization(producer)
        return sources, saw_low_precision, materialized

    def trace(self, name: str, visited: set[str]) -> SourceTrace | None:
        if name in visited:
            return None
        produced = self._producers.get(name)
        if produced is None:
            return set(), False, False
        producer = produced.layer
        if str(producer.get("type", "")).lower() == "start":
            output_dtypes = (producer.get("tensor_dtypes") or {}).get(
                "outputs",
            ) or []
            if produced.output_index not in range(len(output_dtypes)):
                return None
            return {str(output_dtypes[produced.output_index])}, False, False
        semantic = layer_operation(producer)
        effects = semantic.get("effects") or {}
        aliases = [
            alias
            for alias in effects.get("aliases") or []
            if int(alias.get("output", -1)) == produced.output_index
            and not bool(alias.get("conditional", False))
        ]
        input_names = list(
            (producer.get("tensor_names") or {}).get("inputs") or [],
        )
        next_visited = visited | {name}
        if len(aliases) == 1:
            return self._trace_alias(aliases, input_names, next_visited)
        if not input_names or not self._is_recomputable(semantic, effects):
            return None
        return self._trace_recomputation(
            producer,
            semantic,
            input_names,
            next_visited,
        )

    def trace_operands(self, layer: NodeDict) -> list[SourceTrace] | None:
        result: list[SourceTrace] = []
        for name in (layer.get("tensor_names") or {}).get("inputs") or []:
            traced = self.trace(str(name), set())
            if traced is None:
                return None
            _, saw_low_precision, materialized = traced
            if materialized and not saw_low_precision:
                return None
            result.append(traced)
        return result


def _contraction_operand_sources(
    layer: NodeDict,
    layers: dict[str, NodeDict],
) -> list[SourceTrace] | None:
    return _OperandSourceTracer(layers).trace_operands(layer)


def contraction_operands_are_graph_external(
    layer: NodeDict,
    layers: dict[str, NodeDict],
) -> bool:
    """Return whether contraction operands are external or safely recomputable."""
    return _contraction_operand_sources(layer, layers) is not None


def contraction_has_region_boundary_proof(
    layer: NodeDict,
    region: Mapping[str, Any],
    layers: Mapping[str, NodeDict],
) -> bool:
    """Prove mapper I/O is scoped by materialized fusion-region boundaries."""
    names = layer.get("tensor_names") or {}
    shapes = layer.get("tensor_shapes") or {}
    dtypes = layer.get("tensor_dtypes") or {}
    inputs = {str(item) for item in names.get("inputs") or []}
    external_inputs = {
        str(item) for item in region.get("external_inputs") or []
    }
    outputs = [str(item) for item in names.get("outputs") or []]
    output_shapes = list(shapes.get("outputs") or [])
    output_dtypes = [str(item) for item in dtypes.get("outputs") or []]
    if (
        not inputs
        or not inputs.issubset(external_inputs)
        or len(outputs) != 1
        or len(output_shapes) != 1
        or len(output_dtypes) != 1
    ):
        return False
    return _RegionOutputTracer(region, layers).trace(
        outputs[0],
        list(output_shapes[0]),
        output_dtypes[0],
    )


def contraction_external_source_dtypes(
    layer: NodeDict,
    layers: dict[str, NodeDict],
) -> set[str]:
    """Return graph-input dtypes for a proven recomputable contraction region."""
    traced = _contraction_operand_sources(layer, layers)
    if traced is None:
        return set()
    return {dtype for sources, _, _ in traced for dtype in sources}


__all__ = [
    "contraction_external_source_dtypes",
    "contraction_has_region_boundary_proof",
    "contraction_operands_are_graph_external",
]
