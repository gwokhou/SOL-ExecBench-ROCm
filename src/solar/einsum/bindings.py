# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Exact source-input and reference-output checks for canonical operator graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
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
    """Validate and persist one exact make_fx ATen graph."""
    converted: NodeDict = deepcopy(dict(traced))
    converted["source_input_indices"] = bind_inputs(converted, operator)
    _validate_declared_outputs(converted, operator.reference_outputs)
    validate_semantic_graph(converted)
    einsum_path = output / "einsum_graph.yaml"
    einsum_path.write_text(yaml.safe_dump(converted, sort_keys=False))
    return einsum_path


def bind_inputs(
    graph: Mapping[str, DynamicValue],
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


def _start_layers(graph: Mapping[str, DynamicValue]) -> list[NodeDict]:
    return [
        layer
        for layer in (graph.get("layers") or {}).values()
        if str(layer.get("type", "")).lower() == "start"
    ]


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
        raise RuntimeError(
            "make_fx input provenance does not match graph metadata",
        )


def _validate_declared_outputs(
    graph: Mapping[str, DynamicValue],
    expected: Sequence[TensorSignature],
) -> None:
    declared = graph.get("outputs")
    if not isinstance(declared, list) or len(declared) != len(expected):
        raise RuntimeError(
            "make_fx graph output arity does not match reference",
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
            "make_fx graph outputs do not match reference metadata",
        )


__all__ = [
    "accept_semantic_operator_graph",
    "bind_inputs",
]
