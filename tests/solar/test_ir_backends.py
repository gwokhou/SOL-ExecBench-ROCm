from __future__ import annotations

from pathlib import Path

import torch
import yaml

from solar.contracts import AnalysisRequest, ConversionRequest
from solar.graph.contracts import DEFAULT_EXTRACTION_KIND, ExtractionKind
from solar.graph.extraction import extract_operator_graph
from solar.ir.contracts import (
    DEFAULT_IR_KIND,
    DEFAULT_IR_PATH,
    IRBackend,
    IRKind,
    IRPath,
)
from solar.ir.conversion import convert_operator_graph
from solar.ir.registry import ir_backend, ir_backends
from solar.verification.registry import verification_backend
from solar.verification.verify import IRGraphExecutor


def _matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left @ right


def test_fixed_ir_paths_preserve_the_same_matmul_semantics(
    tmp_path: Path,
) -> None:
    inputs = (
        torch.arange(6.0).reshape(2, 3),
        torch.arange(12.0).reshape(3, 4),
    )
    artifacts = {}
    for ir_path in IRPath:
        output = tmp_path / ir_path.value
        operator = extract_operator_graph(
            _matmul,
            inputs,
            device="cpu",
            output_dir=output,
            name="matmul",
            extraction_kind=ir_path.extraction_kind,
        )
        artifacts[ir_path] = convert_operator_graph(
            operator,
            output_dir=output,
            ir_kind=ir_path.ir_kind,
        )
    extended_einsum = artifacts[IRPath.TORCHVIEW_EXTENDED_EINSUM]
    aten = artifacts[IRPath.MAKE_FX_ATEN]
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
            IRGraphExecutor(graph, verification_backend(graph["ir_kind"]))(
                *inputs
            ),
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
    assert DEFAULT_IR_PATH is IRPath.TORCHVIEW_EXTENDED_EINSUM
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
            ir_path=IRPath.MAKE_FX_ATEN,
        ),
        architecture={},
        output_dir=tmp_path / "make-fx",
    )
    assert make_fx.ir_path is IRPath.MAKE_FX_ATEN
    assert make_fx.ir_kind is IRKind.ATEN


def test_every_fixed_path_shares_one_backend_interface(
    tmp_path: Path,
) -> None:
    """Both IR dialects implement one backend on trusted make_fx data."""
    backends = ir_backends()
    assert {backend.kind for backend in backends} == {
        IRKind.ATEN,
        IRKind.EXTENDED_EINSUM,
    }
    assert all(isinstance(backend, IRBackend) for backend in backends)
    assert ir_backend(DEFAULT_IR_KIND).kind is IRKind.EXTENDED_EINSUM

    inputs = (
        torch.arange(6.0).reshape(2, 3),
        torch.arange(12.0).reshape(3, 4),
    )
    expected = _matmul(*inputs)
    for ir_path in IRPath:
        backend = ir_backend(ir_path.ir_kind)
        output_dir = tmp_path / ir_path.value
        output_dir.mkdir()
        operator = extract_operator_graph(
            _matmul,
            inputs,
            device="cpu",
            output_dir=output_dir,
            name="matmul",
            extraction_kind=ir_path.extraction_kind,
        )
        artifact = backend.convert(operator, output_dir)
        graph = yaml.safe_load(artifact.path.read_text())
        assert artifact.kind is backend.kind
        backend.validate(graph)
        assert graph["ir_kind"] == backend.kind
        torch.testing.assert_close(
            IRGraphExecutor(
                graph,
                verification_backend(ir_path.ir_kind),
            )(*inputs),
            expected,
        )
