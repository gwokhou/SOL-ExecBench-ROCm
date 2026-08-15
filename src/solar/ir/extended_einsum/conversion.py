# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict operator-graph to executable-einsum conversion boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import yaml

from solar.errors import (
    ReferenceOutputBindingError,
    SourceInputBindingError,
    StrictConversionError,
)
from solar.graph.contracts import (
    ExtractionKind,
    OperatorGraphArtifact,
    TensorSignature,
)
from solar.ir.contracts import IRGraphArtifact, IRKind
from solar.ir.extended_einsum.semantics import validate_semantic_graph
from solar.ir.extended_einsum.torchview.converter import PyTorchToEinsum
from solar.types import DynamicValue


def convert_operator_graph(
    operator: OperatorGraphArtifact,
    output_dir: str | Path,
) -> IRGraphArtifact:
    """Convert one operator artifact and preserve exact argument/output bindings."""
    output = Path(output_dir)
    traced = operator.document.data
    try:
        extraction = ExtractionKind(str(traced.get("extraction_kind", "")))
        converter = _SOURCE_CONVERTERS[extraction]
    except (KeyError, ValueError) as exc:
        raise StrictConversionError(
            "unsupported operator graph provenance "
            f"{traced.get('extraction_kind')!r}",
        ) from exc
    converted = converter(operator, output, traced)
    validate_extended_einsum_graph(converted)
    einsum_path = output / "einsum_graph.yaml"
    einsum_path.write_text(yaml.safe_dump(converted, sort_keys=False))
    return IRGraphArtifact(path=einsum_path, kind=IRKind.EXTENDED_EINSUM)


def _convert_torchview_graph(
    operator: OperatorGraphArtifact,
    output: Path,
    traced: Mapping[str, DynamicValue],
) -> dict[str, DynamicValue]:
    converted = PyTorchToEinsum(strict=True).convert(
        operator.path, output, copy_graph=False, enable_rename=False
    )
    if converted is None:
        raise StrictConversionError(
            "strict graph conversion produced no einsum graph",
        )
    source_node_remap = converted.pop("_source_node_remap", {})
    converted["ir_kind"] = IRKind.EXTENDED_EINSUM.value
    converted["source_input_indices"] = _bind_inputs(converted, operator)
    _repair_reference_output_dtypes(
        converted,
        traced,
        operator.reference_outputs,
        source_node_remap,
    )
    converted["outputs"] = _bind_outputs(
        converted,
        traced,
        operator.reference_outputs,
        source_node_remap,
    )
    return converted


def _repair_reference_output_dtypes(
    graph: dict[str, DynamicValue],
    traced: Mapping[str, DynamicValue],
    expected: Sequence[TensorSignature],
    source_node_remap: Mapping[str, DynamicValue],
) -> None:
    """Repair known Torchview output-dtype loss from exact reference slots."""
    output_nodes = [
        layer
        for layer in (traced.get("layers") or {}).values()
        if str(layer.get("type", "")).lower() == "output-tensor"
    ]
    if len(output_nodes) != len(expected):
        return
    layers = graph.get("layers") or {}
    for output, signature in zip(output_nodes, expected, strict=True):
        producers = (output.get("connections") or {}).get("inputs") or []
        if len(producers) != 1:
            continue
        producer_id = source_node_remap.get(producers[0], producers[0])
        producer = layers.get(producer_id)
        if not isinstance(producer, dict):
            continue
        shapes = (producer.get("tensor_shapes") or {}).get("outputs") or []
        dtypes = (producer.get("tensor_dtypes") or {}).get("outputs") or []
        raw_slot = (output.get("module_args") or {}).get("producer_output_slot")
        slot = int(raw_slot) if raw_slot is not None else 0
        if (
            slot in range(len(shapes))
            and slot in range(len(dtypes))
            and _trace_shape(shapes[slot]) == signature.shape
        ):
            dtypes[slot] = signature.dtype


def _trace_shape(shape: Sequence[DynamicValue]) -> tuple[int, ...]:
    return tuple(
        int(dimension["trace_value"])
        if isinstance(dimension, Mapping)
        else int(dimension)
        for dimension in shape
    )


_SOURCE_CONVERTERS: dict[
    ExtractionKind,
    Callable[
        [
            OperatorGraphArtifact,
            Path,
            Mapping[str, DynamicValue],
        ],
        dict[str, DynamicValue],
    ],
] = {
    ExtractionKind.TORCHVIEW: _convert_torchview_graph,
}


def validate_extended_einsum_graph(
    graph: Mapping[str, DynamicValue],
) -> None:
    """Validate the extended-einsum discriminator and executable schema."""
    if graph.get("ir_kind") != IRKind.EXTENDED_EINSUM.value:
        raise ValueError("graph is not extended-einsum IR")
    validate_semantic_graph(graph)


def _start_layers(
    graph: Mapping[str, DynamicValue],
) -> list[Mapping[str, DynamicValue]]:
    return [
        layer
        for layer in (graph.get("layers") or {}).values()
        if str(layer.get("type", "")).lower() == "start"
    ]


def _bind_inputs(
    graph: Mapping[str, DynamicValue], operator: OperatorGraphArtifact
) -> list[int]:
    starts = _start_layers(graph)
    expected = list(operator.used_source_indices)
    if len(starts) != len(expected):
        raise SourceInputBindingError(
            "cannot bind source arguments to graph inputs: "
            f"observed={expected}, starts={len(starts)}"
        )
    signatures = dict(operator.source_inputs)
    result: list[int] = []
    for layer in starts:
        raw_index = layer.get("source_input_index")
        if raw_index is None:
            raise SourceInputBindingError(
                "Torchview graph input lacks an exact source index",
            )
        source_index = int(raw_index)
        if source_index not in expected or source_index in result:
            raise SourceInputBindingError(
                "Torchview graph input has an invalid source index",
            )
        _validate_input_signature(layer, source_index, signatures)
        result.append(source_index)
    if result != expected:
        raise SourceInputBindingError(
            "Torchview graph source indices do not match reference order",
        )
    return result


def _validate_input_signature(
    layer: Mapping[str, DynamicValue],
    source_index: int,
    inputs: Mapping[int, TensorSignature],
) -> None:
    shapes = (layer.get("tensor_shapes") or {}).get("outputs") or []
    dtypes = (layer.get("tensor_dtypes") or {}).get("outputs") or []
    signature = inputs.get(source_index)
    if (
        len(shapes) != 1
        or len(dtypes) != 1
        or signature is None
        or tuple(shapes[0]) != signature.shape
        or str(dtypes[0]) != signature.dtype
    ):
        raise SourceInputBindingError(
            "Torchview source index does not match graph input metadata",
        )


def _bind_outputs(
    graph: Mapping[str, DynamicValue],
    traced: Mapping[str, DynamicValue],
    expected: Sequence[TensorSignature],
    source_node_remap: Mapping[str, DynamicValue] | None = None,
) -> list[str]:
    candidates = _output_candidates(graph, traced, source_node_remap)
    if len(candidates) != len(expected):
        raise ReferenceOutputBindingError(
            "cannot preserve exact reference output arity",
        )
    declared: list[str] = []
    for value in expected:
        matches = [
            index
            for index, (_, shape, dtype) in enumerate(candidates)
            if _trace_shape(shape) == value.shape and dtype == value.dtype
        ]
        if not matches:
            raise ReferenceOutputBindingError(
                "traced output metadata does not match reference"
            )
        declared.append(candidates.pop(matches[0])[0])
    return declared


def _output_candidates(
    graph: Mapping[str, DynamicValue],
    traced: Mapping[str, DynamicValue],
    source_node_remap: Mapping[str, DynamicValue] | None = None,
) -> list[tuple[str, list[int], str]]:
    layers = graph.get("layers") or {}
    source_node_remap = source_node_remap or {}
    output_nodes = [
        layer
        for layer in (traced.get("layers") or {}).values()
        if str(layer.get("type", "")).lower() == "output-tensor"
    ]
    result: list[tuple[str, list[int], str]] = []
    for output in output_nodes:
        producers = (output.get("connections") or {}).get("inputs") or []
        if len(producers) != 1:
            raise ReferenceOutputBindingError(
                "cannot bind exact traced graph output",
            )
        producer_id = source_node_remap.get(producers[0], producers[0])
        if producer_id not in layers:
            raise ReferenceOutputBindingError(
                "cannot bind exact traced graph output",
            )
        producer = layers[producer_id]
        names = (producer.get("tensor_names") or {}).get("outputs") or []
        shapes = (producer.get("tensor_shapes") or {}).get("outputs") or []
        dtypes = (producer.get("tensor_dtypes") or {}).get("outputs") or []
        raw_slot = (output.get("module_args") or {}).get(
            "producer_output_slot",
        )
        if raw_slot is None:
            if len(names) == len(shapes) == len(dtypes) == 1:
                raw_slot = 0
            else:
                raise ReferenceOutputBindingError(
                    "traced graph output lacks an exact producer output slot"
                )
        slot = int(raw_slot)
        if (
            slot < 0
            or slot >= len(names)
            or slot >= len(shapes)
            or slot >= len(dtypes)
        ):
            raise ReferenceOutputBindingError(
                "traced graph output selects an unavailable producer slot"
            )
        result.append(
            (str(names[slot]), list(shapes[slot]), str(dtypes[slot])),
        )
    return result


__all__ = ["convert_operator_graph", "validate_extended_einsum_graph"]
