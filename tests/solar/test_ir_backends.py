from __future__ import annotations

from pathlib import Path

import torch
import yaml

from solar.contracts import AnalysisRequest, ConversionRequest
from solar.graph.contracts import DEFAULT_EXTRACTION_KIND, ExtractionKind
from solar.graph.extraction import extract_operator_graph
from solar.ir.contracts import (
    DEFAULT_IR_KIND,
    IRKind,
    IRLifecycle,
)
from solar.ir.conversion import convert_operator_graph
from solar.ir.registry import ir_lifecycle, ir_lifecycles
from solar.verification.verify import IRGraphExecutor


def _matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left @ right


def test_extended_einsum_is_default_and_aten_remains_interchangeable(
    tmp_path: Path,
) -> None:
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
        extraction_kind=ExtractionKind.MAKE_FX_REFERENCE,
    )

    extended_einsum = convert_operator_graph(operator, output_dir=tmp_path)
    aten = convert_operator_graph(
        operator,
        output_dir=tmp_path,
        ir_kind=IRKind.ATEN,
    )
    extended_einsum_graph = yaml.safe_load(extended_einsum.path.read_text())
    aten_graph = yaml.safe_load(aten.path.read_text())

    assert extended_einsum.kind is IRKind.EXTENDED_EINSUM
    assert extended_einsum.path.name == "einsum_graph.yaml"
    assert extended_einsum_graph["ir_kind"] == IRKind.EXTENDED_EINSUM
    assert all(
        "semantic_op" in layer
        for layer in extended_einsum_graph["layers"].values()
    )
    assert all(
        "extended_op" not in layer
        for layer in extended_einsum_graph["layers"].values()
    )
    assert any(
        layer.get("is_real_einsum")
        for layer in extended_einsum_graph["layers"].values()
    )
    assert aten.kind is IRKind.ATEN
    assert aten.path.name == "aten_graph.yaml"
    assert aten_graph["ir_kind"] == IRKind.ATEN
    assert any(
        "semantic_op" in layer for layer in aten_graph["layers"].values()
    )

    expected = _matmul(*inputs)
    for graph in (extended_einsum_graph, aten_graph):
        torch.testing.assert_close(
            IRGraphExecutor(graph, ir_lifecycle(graph["ir_kind"]))(*inputs),
            expected,
        )


def test_analysis_request_defaults_to_extended_einsum(tmp_path: Path) -> None:
    request = AnalysisRequest(
        conversion=ConversionRequest(
            analysis_id="ir-default",
            reference=lambda value: value,
            input_factory=lambda seed: (torch.tensor(float(seed)),),
            reference_name="tests#identity",
            reference_sha256="a" * 64,
        ),
        architecture={},
        output_dir=tmp_path / "result",
    )

    assert DEFAULT_IR_KIND is IRKind.EXTENDED_EINSUM
    assert DEFAULT_EXTRACTION_KIND is ExtractionKind.TORCHVIEW
    assert request.extraction_kind is ExtractionKind.TORCHVIEW
    assert request.ir_kind is IRKind.EXTENDED_EINSUM
    make_fx = AnalysisRequest(
        conversion=ConversionRequest(
            analysis_id="ir-make-fx-default",
            reference=lambda value: value,
            input_factory=lambda seed: (torch.tensor(float(seed)),),
            reference_name="tests#identity",
            reference_sha256="b" * 64,
            extraction_kind=ExtractionKind.MAKE_FX_REFERENCE,
        ),
        architecture={},
        output_dir=tmp_path / "make-fx",
    )
    assert make_fx.ir_kind is IRKind.EXTENDED_EINSUM


def test_every_lifecycle_shares_one_interface_on_the_same_data(
    tmp_path: Path,
) -> None:
    """Both IR dialects implement one lifecycle on trusted make_fx data."""
    lifecycles = ir_lifecycles()
    assert {lifecycle.kind for lifecycle in lifecycles} == {
        IRKind.ATEN,
        IRKind.EXTENDED_EINSUM,
    }
    assert all(isinstance(lifecycle, IRLifecycle) for lifecycle in lifecycles)
    assert ir_lifecycle(DEFAULT_IR_KIND).kind is IRKind.EXTENDED_EINSUM

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
        extraction_kind=ExtractionKind.MAKE_FX_REFERENCE,
    )
    expected = _matmul(*inputs)
    for lifecycle in lifecycles:
        output_dir = tmp_path / lifecycle.kind.value
        output_dir.mkdir()
        artifact = lifecycle.convert(operator, output_dir)
        graph = yaml.safe_load(artifact.path.read_text())
        assert artifact.kind is lifecycle.kind
        lifecycle.validate(graph)
        assert graph["ir_kind"] == lifecycle.kind
        torch.testing.assert_close(
            IRGraphExecutor(graph, lifecycle)(*inputs),
            expected,
        )
