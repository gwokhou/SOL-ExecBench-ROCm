# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict operator-graph to executable-einsum conversion boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from solar.einsum.pytorch_to_einsum import PyTorchToEinsum
from solar.graph.extraction import OperatorGraphArtifact, TensorSignature

_REVIEWED_HANDLERS = Path(__file__).parent.parent / "handlers"


@dataclass(frozen=True)
class EinsumGraphArtifact:
    """Canonical executable einsum graph produced by strict conversion."""

    path: Path


def convert_operator_graph(
    operator: OperatorGraphArtifact,
    *,
    output_dir: str | Path,
) -> EinsumGraphArtifact:
    """Convert one operator artifact and preserve exact argument/output bindings."""
    output = Path(output_dir)
    converted = PyTorchToEinsum(strict=True, cache_dir=str(_REVIEWED_HANDLERS)).convert(
        operator.path, output, copy_graph=False, enable_rename=False
    )
    if converted is None:
        raise RuntimeError("strict graph conversion produced no einsum graph")
    converted["source_input_indices"] = _bind_inputs(converted, operator)
    traced = yaml.safe_load(operator.path.read_text()) or {}
    converted["outputs"] = _bind_outputs(converted, traced, operator.reference_outputs)
    einsum_path = output / "einsum_graph.yaml"
    einsum_path.write_text(yaml.safe_dump(converted, sort_keys=False))
    return EinsumGraphArtifact(einsum_path)


def _start_layers(graph: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        layer
        for layer in (graph.get("layers") or {}).values()
        if str(layer.get("type", "")).lower() == "start"
    ]


def _bind_inputs(
    graph: Mapping[str, Any], operator: OperatorGraphArtifact
) -> list[int]:
    starts = _start_layers(graph)
    ordered = list(operator.used_source_indices)
    if len(starts) != len(ordered):
        raise RuntimeError(
            "cannot bind source arguments to graph inputs: "
            f"observed={ordered}, starts={len(starts)}"
        )
    signatures = dict(operator.source_inputs)
    candidates = [_input_candidates(layer, ordered, signatures) for layer in starts]
    bindings: list[list[int]] = []
    _search_bindings(starts, candidates, 0, [], set(ordered), -1, bindings)
    if len(bindings) != 1:
        reason = "no" if not bindings else "ambiguous"
        raise RuntimeError(f"{reason} exact source-to-graph input binding")
    return bindings[0]


def _input_candidates(
    layer: Mapping[str, Any],
    indices: Sequence[int],
    inputs: Mapping[int, TensorSignature],
) -> list[int]:
    shapes = (layer.get("tensor_shapes") or {}).get("outputs") or []
    dtypes = (layer.get("tensor_dtypes") or {}).get("outputs") or []
    if len(shapes) != 1 or len(dtypes) != 1:
        raise RuntimeError("graph input lacks exact shape/dtype metadata")
    return [
        index
        for index in indices
        if tuple(shapes[0]) == inputs[index].shape
        and str(dtypes[0]) == inputs[index].dtype
    ]


def _search_bindings(
    starts: Sequence[Mapping[str, Any]],
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
    ordered_start = starts[position].get("source_binding") == "torchview_input_order"
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
    graph: Mapping[str, Any],
    traced: Mapping[str, Any],
    expected: Sequence[TensorSignature],
) -> list[str]:
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
    graph: Mapping[str, Any], traced: Mapping[str, Any]
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
            raise RuntimeError("cannot bind exact traced graph output")
        producer = layers[producers[0]]
        names = (producer.get("tensor_names") or {}).get("outputs") or []
        shapes = (producer.get("tensor_shapes") or {}).get("outputs") or []
        dtypes = (producer.get("tensor_dtypes") or {}).get("outputs") or []
        if len(names) != 1 or len(shapes) != 1 or len(dtypes) != 1:
            raise RuntimeError("traced graph output producer is not single-output")
        result.append((str(names[0]), list(shapes[0]), str(dtypes[0])))
    return result


__all__ = ["EinsumGraphArtifact", "convert_operator_graph"]
