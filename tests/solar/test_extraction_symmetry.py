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
    DEFAULT_EXTRACTION_KIND,
    ExtractionKind,
    GraphBackend,
    OperatorGraphArtifact,
)
from solar.graph.extraction import extract_operator_graph
from solar.graph.registry import extraction_backends
from solar.ir.contracts import DEFAULT_IR_KIND, DEFAULT_IR_PATH, IRKind, IRPath
from solar.ir.conversion import convert_operator_graph
from solar.ir.registry import ir_backends
from solar.schema_versions import SchemaVersion as SolarSchemaVersion
from solar.verification.registry import verification_backend


def _matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left @ right


def _inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(4, 8, generator=generator),
        torch.randn(8, 6, generator=generator),
    )


def _analysis_request(output: Path, ir_path: IRPath) -> AnalysisRequest:
    return AnalysisRequest(
        conversion=ConversionRequest(
            analysis_id=f"ir_path:{ir_path}:matmul",
            reference=_matmul,
            input_factory=_inputs,
            reference_name="tests.test_extraction_symmetry#matmul",
            reference_sha256="a" * 64,
            ir_path=ir_path,
        ),
        architecture="RX_9060_XT",
        output_dir=output,
    )


@pytest.mark.parametrize(
    "ir_path",
    list(IRPath),
)
def test_fixed_paths_publish_their_canonical_ir_artifacts(
    tmp_path: Path,
    ir_path: IRPath,
) -> None:
    result = analyze(_analysis_request(tmp_path / ir_path.value, ir_path))

    assert isinstance(result, AnalysisResult)
    assert result.sol_score_eligible
    assert not result.publication_eligible
    assert result.bound.kind == "roofline_eq1_v1"
    artifact_paths = {artifact.path for artifact in result.artifacts}
    assert {
        "operator_graph.yaml",
        ir_path.graph_filename,
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
        (result.output_dir / ir_path.graph_filename).read_text(),
    )
    attestation = yaml.safe_load(
        (result.output_dir / "conversion-attestation.yaml").read_text(),
    )
    manifest = yaml.safe_load(
        (result.output_dir / "manifest.yaml").read_text(),
    )
    assert operator["extraction_kind"] == ir_path.extraction_kind.value
    assert graph["ir_kind"] == ir_path.ir_kind.value
    assert (
        attestation["predicate"]["verifier"]
        == SolarSchemaVersion.IR_VERIFICATION
    )
    assert manifest["analysis_contract"]["ir_path"] == ir_path.value
    assert (
        manifest["analysis_contract"]["extraction_kind"]
        == ir_path.extraction_kind.value
    )
    assert manifest["analysis_contract"]["ir_kind"] == ir_path.ir_kind.value
    assert manifest["sol_score_eligible"] is True


@pytest.mark.parametrize("ir_path", list(IRPath))
def test_readiness_passes_for_both_fixed_paths(
    tmp_path: Path,
    ir_path: IRPath,
) -> None:
    result = audit_conversion(
        ConversionReadinessRequest(
            conversion=ConversionRequest(
                analysis_id=f"readiness:{ir_path}",
                reference=_matmul,
                input_factory=_inputs,
                reference_name="tests.test_extraction_symmetry#matmul",
                reference_sha256="b" * 64,
                ir_path=ir_path,
            ),
            architecture="RX_9060_XT",
            output_dir=tmp_path / f"readiness-{ir_path}",
        ),
    )

    assert result.ready
    graph = yaml.safe_load(
        (Path(result.output_dir) / ir_path.graph_filename).read_text(),
    )
    assert graph["ir_kind"] == ir_path.ir_kind.value


def test_extended_einsum_is_default_and_backends_are_first_class(
    tmp_path: Path,
) -> None:
    request = _analysis_request(tmp_path / "default", DEFAULT_IR_PATH)

    assert DEFAULT_EXTRACTION_KIND is ExtractionKind.TORCHVIEW
    assert DEFAULT_IR_KIND is IRKind.EXTENDED_EINSUM
    assert DEFAULT_IR_PATH is IRPath.TORCHVIEW_EXTENDED_EINSUM
    assert request.ir_path is IRPath.TORCHVIEW_EXTENDED_EINSUM
    assert request.extraction_kind is ExtractionKind.TORCHVIEW
    assert request.ir_kind is IRKind.EXTENDED_EINSUM
    assert {backend.kind for backend in ir_backends()} == {
        IRKind.ATEN,
        IRKind.EXTENDED_EINSUM,
    }
    assert {backend.kind for backend in extraction_backends()} == {
        ExtractionKind.MAKE_FX_REFERENCE,
        ExtractionKind.TORCHVIEW,
    }
    assert verification_backend(DEFAULT_IR_KIND).execute.__module__ == (
        "solar.verification.extended"
    )


def test_extraction_kinds_are_distinct() -> None:
    assert ExtractionKind.TORCHVIEW is ExtractionKind.TORCHVIEW
    assert ExtractionKind.MAKE_FX_REFERENCE is ExtractionKind.MAKE_FX_REFERENCE


@pytest.mark.parametrize(
    ("extraction_kind", "ir_kind"),
    [
        (ExtractionKind.TORCHVIEW, IRKind.ATEN),
        (ExtractionKind.MAKE_FX_REFERENCE, IRKind.EXTENDED_EINSUM),
    ],
)
def test_ir_backend_rejects_cross_path_combinations(
    tmp_path: Path,
    extraction_kind: ExtractionKind,
    ir_kind: IRKind,
) -> None:
    operator = extract_operator_graph(
        _matmul,
        _inputs(5),
        device="cpu",
        output_dir=tmp_path,
        name="incompatible",
        extraction_kind=extraction_kind,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            f"{ir_kind.value!r} does not support extraction "
            f"{extraction_kind.value!r}"
        ),
    ):
        convert_operator_graph(
            operator,
            output_dir=tmp_path,
            ir_kind=ir_kind,
        )


def test_fixed_paths_emit_equivalent_contraction_semantics(
    tmp_path: Path,
) -> None:
    signatures: dict[IRPath, list[tuple[object, ...]]] = {}
    for ir_path in IRPath:
        output = tmp_path / ir_path.value
        operator = extract_operator_graph(
            _matmul,
            _inputs(7),
            device="cpu",
            output_dir=output,
            name=f"{ir_path.value}-matmul",
            extraction_kind=ir_path.extraction_kind,
        )
        artifact = convert_operator_graph(
            operator,
            output_dir=output,
            ir_kind=ir_path.ir_kind,
        )
        graph = yaml.safe_load(artifact.path.read_text())
        signatures[ir_path] = sorted(
            (
                tuple(
                    tuple(shape) for shape in layer["tensor_shapes"]["inputs"]
                ),
                tuple(
                    tuple(shape) for shape in layer["tensor_shapes"]["outputs"]
                ),
            )
            for layer in graph["layers"].values()
            if layer.get("is_real_einsum")
            or (layer.get("semantic_op") or {}).get("target")
            in {"matmul", "mm"}
        )

    assert (
        signatures[IRPath.TORCHVIEW_EXTENDED_EINSUM]
        == signatures[IRPath.MAKE_FX_ATEN]
    )


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
        extraction_kind=ExtractionKind.MAKE_FX_REFERENCE,
    )

    assert result is expected
    assert seen["reference"] is _matmul
    assert seen["options"] == {
        "device": "cpu",
        "output_dir": tmp_path,
        "name": "registry-dispatch",
    }
