# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Canonical exact-ATen extraction for reference callables."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml


def trace_make_fx_reference(
    reference: Callable[..., Any],
    inputs: tuple[Any, ...],
    *,
    output: Path,
    name: str,
) -> tuple[dict[int, Any], list[int], Path]:
    """Capture and serialize an exact ATen reference graph."""
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx

    from solar.graph.reference_serializer import ReferenceGraphSerializer

    tensor_inputs = {
        index: value
        for index, value in enumerate(inputs)
        if isinstance(value, torch.Tensor)
    }
    tensor_indices = list(tensor_inputs)

    def tensor_reference(*tensor_values: Any) -> Any:
        replacements = iter(tensor_values)
        arguments = tuple(
            next(replacements) if isinstance(value, torch.Tensor) else value
            for value in inputs
        )
        return reference(*arguments)

    try:
        graph_module = make_fx(tensor_reference)(*tensor_inputs.values())
        placeholders = [
            node
            for node in graph_module.graph.nodes
            if node.op == "placeholder"
        ]
        if len(placeholders) != len(tensor_indices):
            raise RuntimeError(
                "make_fx tensor input arity does not match reference",
            )
        used = [
            (node, source_index)
            for node, source_index in zip(
                placeholders, tensor_indices, strict=True
            )
            if node.users
        ]
        graph = ReferenceGraphSerializer().serialize_fx_reference(
            graph_module,
            name,
        )
        for node, source_index in used:
            layer = graph["layers"].get(node.name)
            if layer is None:
                raise RuntimeError("make_fx input provenance is incomplete")
            layer["source_input_index"] = source_index
    except Exception as extraction_error:
        raise RuntimeError(
            f"make_fx reference extraction failed: {extraction_error}",
        ) from extraction_error
    operator_path = output / "operator_graph.yaml"
    operator_path.write_text(yaml.safe_dump(graph, sort_keys=False))
    return tensor_inputs, [index for _, index in used], operator_path


__all__ = ["trace_make_fx_reference"]
