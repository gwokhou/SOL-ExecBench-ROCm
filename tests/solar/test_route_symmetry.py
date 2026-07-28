from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from solar.api import (
    AnalysisRequest,
    AnalysisResult,
    ConversionReadinessRequest,
    ConversionRequest,
    analyze,
    audit_conversion,
)
from solar.graph import registry as graph_registry
from solar.graph.contracts import (
    ExtractionKind,
    GraphBackend,
    OperatorGraphArtifact,
)
from solar.graph.extraction import extract_operator_graph
from solar.graph.registry import extraction_backends
from solar.ir.contracts import DEFAULT_IR_KIND, IRKind
from solar.ir.conversion import convert_operator_graph
from solar.ir.registry import ir_lifecycle, ir_lifecycles
from solar.routes import DEFAULT_ROUTE, Route, route_spec
from solar.schema_versions import IR_VERIFICATION_SCHEMA_VERSION


def _matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left @ right


def _inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(4, 8, generator=generator),
        torch.randn(8, 6, generator=generator),
    )


def _analysis_request(output: Path, route: Route) -> AnalysisRequest:
    return AnalysisRequest(
        conversion=ConversionRequest(
            analysis_id=f"route:{route}:matmul",
            reference=_matmul,
            input_factory=_inputs,
            reference_name="tests.test_route_symmetry#matmul",
            reference_sha256="a" * 64,
            route=route,
        ),
        architecture="RX_9060_XT",
        output_dir=output,
    )


@pytest.mark.parametrize(
    ("route", "extraction"),
    [
        (Route.NVLABS, ExtractionKind.TORCHVIEW),
        (Route.MAINLINE, ExtractionKind.MAKE_FX_REFERENCE),
    ],
)
def test_routes_publish_symmetric_nvlabs_artifacts(
    tmp_path: Path,
    route: Route,
    extraction: ExtractionKind,
) -> None:
    result = analyze(_analysis_request(tmp_path / route.value, route))

    assert isinstance(result, AnalysisResult)
    assert result.sol_score_eligible
    assert not result.publication_eligible
    assert result.bound.kind == "roofline_eq1_v1"
    artifact_paths = {artifact.path for artifact in result.artifacts}
    assert {
        "operator_graph.yaml",
        "einsum_graph.yaml",
        "conversion-attestation.yaml",
        "solar-analysis.yaml",
    } <= artifact_paths
    assert artifact_paths == {
        path.relative_to(result.output_dir).as_posix()
        for path in result.output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.yaml"
    }
    operator = yaml.safe_load(
        (result.output_dir / "operator_graph.yaml").read_text(),
    )
    graph = yaml.safe_load(
        (result.output_dir / "einsum_graph.yaml").read_text(),
    )
    attestation = yaml.safe_load(
        (result.output_dir / "conversion-attestation.yaml").read_text(),
    )
    manifest = yaml.safe_load(
        (result.output_dir / "manifest.yaml").read_text(),
    )
    assert operator["extraction_kind"] == extraction.value
    assert graph["ir_kind"] == IRKind.NVLABS_EINSUM.value
    assert (
        attestation["predicate"]["verifier"] == IR_VERIFICATION_SCHEMA_VERSION
    )
    assert manifest["schema_version"] == 3
    assert manifest["analysis_contract"]["route"] == route.value
    assert manifest["analysis_contract"]["representation"] == (
        IRKind.NVLABS_EINSUM.value
    )
    assert manifest["sol_score_eligible"] is True


@pytest.mark.parametrize("route", list(Route))
def test_readiness_passes_for_both_routes(
    tmp_path: Path,
    route: Route,
) -> None:
    result = audit_conversion(
        ConversionReadinessRequest(
            conversion=ConversionRequest(
                analysis_id=f"readiness:{route}",
                reference=_matmul,
                input_factory=_inputs,
                reference_name="tests.test_route_symmetry#matmul",
                reference_sha256="b" * 64,
                route=route,
            ),
            architecture="RX_9060_XT",
            output_dir=tmp_path / f"readiness-{route}",
        ),
    )

    assert result.ready
    graph = yaml.safe_load(
        (Path(result.output_dir) / "einsum_graph.yaml").read_text(),
    )
    assert graph["ir_kind"] == IRKind.NVLABS_EINSUM.value


def test_nvlabs_is_the_default_and_backends_are_first_class(
    tmp_path: Path,
) -> None:
    request = _analysis_request(tmp_path / "default", DEFAULT_ROUTE)

    assert DEFAULT_ROUTE is Route.NVLABS
    assert DEFAULT_IR_KIND is IRKind.NVLABS_EINSUM
    assert request.route is Route.NVLABS
    assert request.representation is IRKind.NVLABS_EINSUM
    assert {lifecycle.kind for lifecycle in ir_lifecycles()} == {
        IRKind.ATEN,
        IRKind.NVLABS_EINSUM,
    }
    assert {backend.kind for backend in extraction_backends()} == {
        ExtractionKind.MAKE_FX_REFERENCE,
        ExtractionKind.TORCHVIEW,
    }
    assert ir_lifecycle(DEFAULT_IR_KIND).verify.__module__ == (
        "solar.ir.nvlabs_einsum.backend"
    )


def test_route_profiles_only_select_extraction_strategy() -> None:
    assert route_spec(Route.NVLABS).extraction is ExtractionKind.TORCHVIEW
    assert (
        route_spec(Route.MAINLINE).extraction
        is ExtractionKind.MAKE_FX_REFERENCE
    )


def test_ir_lifecycle_rejects_an_incompatible_extraction(
    tmp_path: Path,
) -> None:
    operator = extract_operator_graph(
        _matmul,
        _inputs(5),
        device="cpu",
        output_dir=tmp_path,
        name="incompatible",
        extraction=ExtractionKind.TORCHVIEW,
    )

    with pytest.raises(
        RuntimeError,
        match="'aten' does not support extraction 'torchview_v1'",
    ):
        convert_operator_graph(
            operator,
            output_dir=tmp_path,
            representation=IRKind.ATEN,
        )


def test_routes_emit_equivalent_nvlabs_contraction_semantics(
    tmp_path: Path,
) -> None:
    signatures: dict[Route, list[tuple[object, ...]]] = {}
    for route in Route:
        output = tmp_path / route.value
        operator = extract_operator_graph(
            _matmul,
            _inputs(7),
            device="cpu",
            output_dir=output,
            name=f"{route.value}-matmul",
            extraction=route_spec(route).extraction,
        )
        artifact = convert_operator_graph(operator, output_dir=output)
        graph = yaml.safe_load(artifact.path.read_text())
        signatures[route] = sorted(
            (
                semantic["equation"],
                tuple(
                    tuple(shape) for shape in layer["tensor_shapes"]["inputs"]
                ),
                tuple(
                    tuple(shape) for shape in layer["tensor_shapes"]["outputs"]
                ),
            )
            for layer in graph["layers"].values()
            if (semantic := layer.get("semantic_op") or {}).get("kind")
            == "einsum"
        )

    assert signatures[Route.NVLABS] == signatures[Route.MAINLINE]


def test_graph_extraction_dispatches_through_registry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected = OperatorGraphArtifact(tmp_path / "registered.yaml", (), (), ())
    seen: dict[str, object] = {}

    def extract(reference, inputs, **options):
        seen.update(
            reference=reference,
            inputs=inputs,
            options=options,
        )
        return expected

    stub = GraphBackend(ExtractionKind.MAKE_FX_REFERENCE, extract)
    monkeypatch.setitem(
        graph_registry._EXTRACTION_LOADERS,
        ExtractionKind.MAKE_FX_REFERENCE,
        lambda: stub,
    )

    result = extract_operator_graph(
        _matmul,
        _inputs(9),
        device="cpu",
        output_dir=tmp_path,
        name="registry-dispatch",
    )

    assert result is expected
    assert seen["reference"] is _matmul
    assert seen["options"] == {
        "device": "cpu",
        "output_dir": tmp_path,
        "name": "registry-dispatch",
    }
