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
from solar.ir.extended_einsum.make_fx_conversion import convert_make_fx_graph
from solar.ir.extended_einsum.semantics import validate_semantic_graph
from solar.ir.extended_einsum.torchview.converter import PyTorchToEinsum
from solar.types import DynamicValue

_REVIEWED_HANDLERS = Path(__file__).parent / "reviewed_handlers"


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
    return IRGraphArtifact(einsum_path, IRKind.EXTENDED_EINSUM)


def _convert_make_fx_graph(
    operator: OperatorGraphArtifact,
    output: Path,
    traced: Mapping[str, DynamicValue],
) -> dict[str, DynamicValue]:
    del output, traced
    return convert_make_fx_graph(operator)


def _convert_torchview_graph(
    operator: OperatorGraphArtifact,
    output: Path,
    traced: Mapping[str, DynamicValue],
) -> dict[str, DynamicValue]:
    converted = PyTorchToEinsum(
        strict=True, cache_dir=str(_REVIEWED_HANDLERS)
    ).convert(operator.path, output, copy_graph=False, enable_rename=False)
    if converted is None:
        raise StrictConversionError(
            "strict graph conversion produced no einsum graph",
        )
    converted["ir_kind"] = IRKind.EXTENDED_EINSUM.value
    converted["source_input_indices"] = _bind_inputs(converted, operator)
    converted["outputs"] = _bind_outputs(
        converted, traced, operator.reference_outputs
    )
    return converted


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
    ExtractionKind.MAKE_FX_REFERENCE: _convert_make_fx_graph,
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
    ordered = list(operator.used_source_indices)
    if len(starts) != len(ordered):
        raise SourceInputBindingError(
            "cannot bind source arguments to graph inputs: "
            f"observed={ordered}, starts={len(starts)}"
        )
    signatures = dict(operator.source_inputs)
    candidates = [
        _input_candidates(layer, ordered, signatures) for layer in starts
    ]
    bindings: list[list[int]] = []
    _search_bindings(starts, candidates, 0, [], set(ordered), -1, bindings)
    if len(bindings) != 1:
        reason = "no" if not bindings else "ambiguous"
        raise SourceInputBindingError(
            f"{reason} exact source-to-graph input binding",
        )
    return bindings[0]


def _input_candidates(
    layer: Mapping[str, DynamicValue],
    indices: Sequence[int],
    inputs: Mapping[int, TensorSignature],
) -> list[int]:
    shapes = (layer.get("tensor_shapes") or {}).get("outputs") or []
    dtypes = (layer.get("tensor_dtypes") or {}).get("outputs") or []
    if len(shapes) != 1 or len(dtypes) != 1:
        raise SourceInputBindingError(
            "graph input lacks exact shape/dtype metadata",
        )
    return [
        index
        for index in indices
        if tuple(shapes[0]) == inputs[index].shape
        and str(dtypes[0]) == inputs[index].dtype
    ]


def _search_bindings(
    starts: Sequence[Mapping[str, DynamicValue]],
    candidates: Sequence[Sequence[int]],
    position: int,
    chosen: list[int],
    remaining: set[int],
    last_ordered: int,
    results: list[list[int]],
) -> None:
    if len(results) > 1:
        return
    if position == len(candidates):
        results.append(list(chosen))
        return
    ordered_start = (
        starts[position].get("source_binding") == "torchview_input_order"
    )
    for source_index in candidates[position]:
        if source_index not in remaining:
            continue
        if ordered_start and source_index <= last_ordered:
            continue
        chosen.append(source_index)
        _search_bindings(
            starts,
            candidates,
            position + 1,
            chosen,
            remaining - {source_index},
            source_index if ordered_start else last_ordered,
            results,
        )
        chosen.pop()


def _bind_outputs(
    graph: Mapping[str, DynamicValue],
    traced: Mapping[str, DynamicValue],
    expected: Sequence[TensorSignature],
) -> list[str]:
    candidates = _output_candidates(graph, traced)
    if len(candidates) != len(expected):
        raise ReferenceOutputBindingError(
            "cannot preserve exact reference output arity",
        )
    declared: list[str] = []
    for value in expected:
        matches = [
            index
            for index, (_, shape, dtype) in enumerate(candidates)
            if tuple(shape) == value.shape and dtype == value.dtype
        ]
        if not matches:
            raise ReferenceOutputBindingError(
                "traced output metadata does not match reference"
            )
        declared.append(candidates.pop(matches[0])[0])
    return declared


def _output_candidates(
    graph: Mapping[str, DynamicValue], traced: Mapping[str, DynamicValue]
) -> list[tuple[str, list[int], str]]:
    layers = graph.get("layers") or {}
    output_nodes = [
        layer
        for layer in (traced.get("layers") or {}).values()
        if str(layer.get("type", "")).lower() == "output-tensor"
    ]
    result: list[tuple[str, list[int], str]] = []
    for output in output_nodes:
        producers = (output.get("connections") or {}).get("inputs") or []
        if len(producers) != 1 or producers[0] not in layers:
            raise ReferenceOutputBindingError(
                "cannot bind exact traced graph output",
            )
        producer = layers[producers[0]]
        names = (producer.get("tensor_names") or {}).get("outputs") or []
        shapes = (producer.get("tensor_shapes") or {}).get("outputs") or []
        dtypes = (producer.get("tensor_dtypes") or {}).get("outputs") or []
        if len(names) != 1 or len(shapes) != 1 or len(dtypes) != 1:
            raise ReferenceOutputBindingError(
                "traced graph output producer is not single-output"
            )
        result.append((str(names[0]), list(shapes[0]), str(dtypes[0])))
    return result


__all__ = ["convert_operator_graph", "validate_extended_einsum_graph"]
