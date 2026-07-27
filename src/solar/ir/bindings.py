# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Source-input and reference-output bindings shared by IR backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from solar.graph.extraction import OperatorGraphArtifact, TensorSignature


def bind_inputs(
    graph: Mapping[str, Any],
    operator: OperatorGraphArtifact,
) -> list[int]:
    """Validate exact source indices recorded on canonical start layers."""
    starts = _start_layers(graph)
    expected = list(operator.used_source_indices)
    if len(starts) != len(expected):
        raise RuntimeError("make_fx graph input arity does not match reference")
    signatures = dict(operator.source_inputs)
    result: list[int] = []
    for layer in starts:
        source_index = layer.get("source_input_index")
        if source_index is None:
            raise RuntimeError("make_fx input provenance is incomplete")
        source_index = int(source_index)
        if source_index not in expected or source_index in result:
            raise RuntimeError("make_fx input provenance is invalid")
        _validate_input_signature(layer, source_index, signatures)
        result.append(source_index)
    if set(result) != set(expected):
        raise RuntimeError("make_fx input provenance is incomplete")
    return result


def validate_declared_outputs(
    graph: Mapping[str, Any],
    expected: Sequence[TensorSignature],
) -> None:
    """Ensure the graph outputs retain traced reference metadata exactly."""
    declared = graph.get("outputs")
    if not isinstance(declared, list) or len(declared) != len(expected):
        raise RuntimeError(
            "make_fx graph output arity does not match reference"
        )
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
                for name, shape, dtype in zip(
                    names, shapes, dtypes, strict=True
                )
            },
        )
    if any(
        metadata.get(str(name)) != signature
        for name, signature in zip(declared, expected, strict=True)
    ):
        raise RuntimeError(
            "make_fx graph outputs do not match reference metadata"
        )


def bind_graph(
    graph: dict[str, Any],
    operator: OperatorGraphArtifact,
) -> dict[str, Any]:
    """Attach and validate the public source-input binding for one IR graph."""
    graph["source_input_indices"] = bind_inputs(graph, operator)
    validate_declared_outputs(graph, operator.reference_outputs)
    return graph


def _start_layers(graph: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        layer
        for layer in (graph.get("layers") or {}).values()
        if str(layer.get("type", "")).lower() == "start"
    ]


def _validate_input_signature(
    layer: Mapping[str, Any],
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
        raise RuntimeError(
            "make_fx input provenance does not match graph metadata",
        )


__all__ = ["bind_graph", "bind_inputs", "validate_declared_outputs"]
