from __future__ import annotations

from pathlib import Path

import torch
import yaml

from solar.contracts import AnalysisRequest
from solar.graph.extraction import extract_operator_graph
from solar.ir.contracts import (
    DEFAULT_IR_KIND,
    IRBackend,
    IRKind,
)
from solar.ir.conversion import convert_operator_graph
from solar.ir.registry import ir_backend, ir_backends
from solar.routes import DEFAULT_ROUTE, Route
from solar.verification.verify import IRGraphExecutor


def _matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left @ right


def test_nvlabs_einsum_is_default_and_aten_remains_interchangeable(
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
    )

    nvlabs = convert_operator_graph(operator, output_dir=tmp_path)
    aten = convert_operator_graph(
        operator,
        output_dir=tmp_path,
        representation=IRKind.ATEN,
    )
    nvlabs_graph = yaml.safe_load(nvlabs.path.read_text())
    aten_graph = yaml.safe_load(aten.path.read_text())

    assert nvlabs.kind is IRKind.NVLABS_EINSUM
    assert nvlabs.path.name == "einsum_graph.yaml"
    assert nvlabs_graph["ir_kind"] == IRKind.NVLABS_EINSUM
    assert all(
        "semantic_op" in layer for layer in nvlabs_graph["layers"].values()
    )
    assert all(
        "extended_op" not in layer for layer in nvlabs_graph["layers"].values()
    )
    assert any(
        layer.get("is_real_einsum") for layer in nvlabs_graph["layers"].values()
    )
    assert aten.kind is IRKind.ATEN
    assert aten.path.name == "aten_graph.yaml"
    assert aten_graph["ir_kind"] == IRKind.ATEN
    assert any(
        "semantic_op" in layer for layer in aten_graph["layers"].values()
    )

    expected = _matmul(*inputs)
    for graph in (nvlabs_graph, aten_graph):
        torch.testing.assert_close(IRGraphExecutor(graph)(*inputs), expected)


def test_analysis_request_defaults_to_nvlabs_einsum(tmp_path: Path) -> None:
    request = AnalysisRequest(
        analysis_id="ir-default",
        reference=lambda value: value,
        input_factory=lambda seed: (torch.tensor(float(seed)),),
        reference_name="tests#identity",
        reference_sha256="a" * 64,
        architecture={},
        output_dir=tmp_path / "result",
    )

    assert DEFAULT_IR_KIND is IRKind.NVLABS_EINSUM
    assert DEFAULT_ROUTE is Route.NVLABS
    assert request.route is Route.NVLABS
    assert request.representation is IRKind.NVLABS_EINSUM
    mainline = AnalysisRequest(
        analysis_id="ir-mainline-default",
        reference=lambda value: value,
        input_factory=lambda seed: (torch.tensor(float(seed)),),
        reference_name="tests#identity",
        reference_sha256="b" * 64,
        architecture={},
        output_dir=tmp_path / "mainline",
        route=Route.MAINLINE,
    )
    assert mainline.representation is IRKind.NVLABS_EINSUM


def test_every_backend_shares_one_interface_on_the_same_data(
    tmp_path: Path,
) -> None:
    """Both retained backends implement one interface on trusted make_fx data."""
    backends = ir_backends()
    assert {backend.kind for backend in backends} == {
        IRKind.ATEN,
        IRKind.NVLABS_EINSUM,
    }
    assert all(isinstance(backend, IRBackend) for backend in backends)
    assert ir_backend(DEFAULT_IR_KIND).kind is IRKind.NVLABS_EINSUM

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
    expected = _matmul(*inputs)
    for backend in backends:
        output_dir = tmp_path / backend.kind.value
        output_dir.mkdir()
        artifact = backend.convert(operator, output_dir)
        graph = yaml.safe_load(artifact.path.read_text())
        assert artifact.kind is backend.kind
        backend.validate(graph)
        assert graph["ir_kind"] == backend.kind
        torch.testing.assert_close(IRGraphExecutor(graph)(*inputs), expected)
