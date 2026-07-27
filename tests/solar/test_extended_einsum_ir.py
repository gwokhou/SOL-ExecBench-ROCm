"""Validation contract for the standalone extended-einsum IR backend.

These tests pin the strict isolation between the extended-einsum and ATen IR
dialects: an extended-einsum graph must never embed ATen ``semantic_op`` data,
and every operation layer must carry a complete ``extended_op`` payload.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from solar.graph.extraction import extract_operator_graph
from solar.ir.conversion import convert_operator_graph
from solar.ir.extended_einsum import (
    ExtendedEinsumIrError,
    validate_extended_einsum_graph,
)


def _matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left @ right


def _extended_graph(tmp_path: Path) -> dict:
    inputs = (
        torch.arange(6.0).reshape(2, 3),
        torch.arange(12.0).reshape(3, 4),
    )
    operator = extract_operator_graph(
        _matmul,
        inputs,
        device="cpu",
        output_dir=tmp_path,
        name="matmul",
    )
    artifact = convert_operator_graph(operator, output_dir=tmp_path)
    assert artifact.kind.value == "extended_einsum"
    return yaml.safe_load(artifact.path.read_text())


def _operation_layer(graph: dict) -> tuple[str, dict]:
    for layer_id, layer in graph["layers"].items():
        if "extended_op" in layer:
            return layer_id, layer
    raise AssertionError("extended-einsum graph has no operation layer")


def test_validate_accepts_well_formed_extended_einsum_graph(
    tmp_path: Path,
) -> None:
    validate_extended_einsum_graph(_extended_graph(tmp_path))


def test_validate_rejects_graph_not_marked_extended_einsum(
    tmp_path: Path,
) -> None:
    graph = _extended_graph(tmp_path)
    graph["ir_kind"] = "aten"
    with pytest.raises(ExtendedEinsumIrError, match="not extended_einsum IR"):
        validate_extended_einsum_graph(graph)


def test_validate_rejects_embedded_aten_semantic_op(tmp_path: Path) -> None:
    """Strict isolation: extended-einsum IR must not carry ATen semantic_op."""
    graph = _extended_graph(tmp_path)
    layer_id, _ = _operation_layer(graph)
    graph["layers"][layer_id]["semantic_op"] = {"kind": "aten", "target": "add"}
    with pytest.raises(ExtendedEinsumIrError, match="embeds ATen semantic_op"):
        validate_extended_einsum_graph(graph)


def test_validate_rejects_layer_without_extended_op(tmp_path: Path) -> None:
    graph = _extended_graph(tmp_path)
    layer_id, _ = _operation_layer(graph)
    del graph["layers"][layer_id]["extended_op"]
    with pytest.raises(ExtendedEinsumIrError, match="no extended_op"):
        validate_extended_einsum_graph(graph)


def test_validate_rejects_operation_without_name(tmp_path: Path) -> None:
    graph = _extended_graph(tmp_path)
    _, layer = _operation_layer(graph)
    layer["extended_op"]["operation"] = ""
    with pytest.raises(ExtendedEinsumIrError, match="no operation name"):
        validate_extended_einsum_graph(graph)


def test_validate_rejects_real_einsum_without_arrow_equation(
    tmp_path: Path,
) -> None:
    graph = _extended_graph(tmp_path)
    for layer in graph["layers"].values():
        operation = layer.get("extended_op") or {}
        if operation.get("is_real_einsum"):
            operation["equation"] = "no-arrow-here"
            break
    else:  # pragma: no cover - matmul always converts to a real einsum
        raise AssertionError("extended-einsum matmul graph lacks a real einsum")
    with pytest.raises(
        ExtendedEinsumIrError, match="no extended-einsum equation"
    ):
        validate_extended_einsum_graph(graph)


def test_validate_rejects_invalid_arguments_container(tmp_path: Path) -> None:
    graph = _extended_graph(tmp_path)
    _, layer = _operation_layer(graph)
    layer["extended_op"]["arguments"] = "not-a-list"
    with pytest.raises(ExtendedEinsumIrError, match="invalid arguments"):
        validate_extended_einsum_graph(graph)
