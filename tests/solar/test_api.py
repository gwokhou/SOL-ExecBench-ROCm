from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import yaml

import solar.api as api
import solar.pipeline as pipeline
from solar.analysis.orojenesis import OrojenesisError
from solar.api import (
    AnalysisFailure,
    AnalysisRequest,
    AnalysisResult,
    ConversionRequest,
    VerificationPolicy,
)
from solar.contracts import SolarStage
from solar.graph.extraction import OperatorGraphArtifact
from solar.ir.contracts import IRKind
from solar.ir.conversion import IRGraphArtifact
from solar.ir.registry import ir_lifecycle
from solar.routes import Route
from solar.verification import VerificationError


class _Profile:
    def to_dict(self):
        return {"name": "test", "gfx_target": "gfx1200"}


def _request(output: Path) -> AnalysisRequest:
    return AnalysisRequest(
        conversion=ConversionRequest(
            analysis_id="problem:workload",
            reference=lambda value: value,
            input_factory=lambda seed: (seed,),
            reference_name="definition.json#reference",
            reference_sha256="a" * 64,
            representation=IRKind.ATEN,
            route=Route.MAINLINE,
        ),
        architecture="RX_9060_XT",
        output_dir=output,
    )


def _matmul_request(
    output: Path,
    *,
    require_orojenesis: bool = False,
    orojenesis_home: str | None = None,
) -> AnalysisRequest:
    def reference(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left @ right

    def input_factory(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        left = torch.full((2, 3), float(seed % 5 + 1))
        return left, torch.arange(12.0).reshape(3, 4)

    return AnalysisRequest(
        conversion=ConversionRequest(
            analysis_id="matmul:formal-policy",
            reference=reference,
            input_factory=input_factory,
            reference_name="tests.test_api#matmul",
            reference_sha256="b" * 64,
            representation=IRKind.ATEN,
            route=Route.MAINLINE,
        ),
        architecture="RX_9060_XT",
        output_dir=output,
        require_orojenesis=require_orojenesis,
        orojenesis_home=orojenesis_home,
    )


def _conv_request(
    output: Path,
    *,
    require_orojenesis: bool = True,
) -> AnalysisRequest:
    def reference(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(value, weight, padding=1)

    def input_factory(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.full((1, 2, 4, 4), float(seed % 5 + 1))
        return value, torch.ones(3, 2, 3, 3)

    return AnalysisRequest(
        conversion=ConversionRequest(
            analysis_id="conv2d:formal-policy",
            reference=reference,
            input_factory=input_factory,
            reference_name="tests.test_api#conv2d",
            reference_sha256="c" * 64,
            representation=IRKind.ATEN,
            route=Route.MAINLINE,
        ),
        architecture="RX_9060_XT",
        output_dir=output,
        require_orojenesis=require_orojenesis,
    )


def test_analyze_publishes_only_complete_atomic_artifact_set(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "result"
    monkeypatch.setattr(
        api.ArchitectureProfile,
        "load",
        lambda value: _Profile(),
    )

    def extract(
        reference,
        inputs,
        *,
        device,
        output_dir,
        name,
        extraction,
    ):
        del reference, inputs, device, name, extraction
        root = Path(output_dir)
        operator = root / "operator_graph.yaml"
        operator.write_text("layers: {}\n")
        return OperatorGraphArtifact(operator, (), (), ())

    def convert(operator, *, output_dir, representation=None):
        del operator, representation
        einsum = Path(output_dir) / "einsum_graph.yaml"
        einsum.write_text("layers: {}\n")
        return IRGraphArtifact(einsum, IRKind.ATEN)

    def verify(**kwargs):
        Path(kwargs["output_path"]).write_text("predicate: passed\n")

    analysis = {
        "schema_version": 3,
        "total": {"lower_bound_seconds": 0.001, "compute_resource": "mfma"},
        "metadata": {"bound_kind": "capacity_constrained_tile_aware_v1"},
    }
    monkeypatch.setattr(
        pipeline,
        "extract_request_graph",
        lambda request, root: extract(
            request.reference,
            tuple(request.input_factory(request.trace_seed)),
            device=request.device,
            output_dir=root,
            name=request.analysis_id,
            extraction="make_fx_reference_v1",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "convert_request_graph",
        lambda request, operator, root: convert(
            operator,
            output_dir=root,
            representation=request.representation,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "verify_request_graph",
        lambda request, graph, output_path: verify(output_path=output_path),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_request_graph",
        lambda request, profile, root, graph: analysis,
    )

    result = api.analyze(_request(output))

    assert isinstance(result, AnalysisResult)
    assert result.bound.seconds == 0.001
    assert {path.name for path in output.iterdir()} == {
        "operator_graph.yaml",
        "einsum_graph.yaml",
        "conversion-attestation.yaml",
        "solar-analysis.yaml",
        "manifest.yaml",
    }
    manifest = yaml.safe_load((output / "manifest.yaml").read_text())
    assert manifest["schema_version"] == 3
    assert "candidate_runtime" not in manifest
    assert "score" not in manifest
    assert manifest["analysis_contract"]["precision"] == "fp16"
    assert manifest["analysis_contract"]["require_orojenesis"] is False
    assert manifest["sol_score_eligible"] is True
    assert manifest["publication_eligible"] is True


def test_analyze_failure_leaves_no_partial_output(tmp_path, monkeypatch):
    output = tmp_path / "result"
    monkeypatch.setattr(
        api.ArchitectureProfile,
        "load",
        lambda value: _Profile(),
    )
    monkeypatch.setattr(
        pipeline,
        "extract_request_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("unsupported"),
        ),
    )

    result = api.analyze(_request(output))

    assert isinstance(result, AnalysisFailure)
    assert result.stage == "graph_extraction"
    assert result.reason_code == "graph_extraction_failed"
    assert not output.exists()


def test_conversion_failure_has_its_own_stable_stage(tmp_path, monkeypatch):
    output = tmp_path / "result"
    monkeypatch.setattr(
        api.ArchitectureProfile,
        "load",
        lambda value: _Profile(),
    )

    def extract(
        reference,
        inputs,
        *,
        device,
        output_dir,
        name,
        extraction,
    ):
        del reference, inputs, device, name, extraction
        operator = Path(output_dir) / "operator_graph.yaml"
        operator.write_text("layers: {}\n")
        return OperatorGraphArtifact(operator, (), (), ())

    monkeypatch.setattr(
        pipeline,
        "extract_request_graph",
        lambda request, root: extract(
            request.reference,
            tuple(request.input_factory(request.trace_seed)),
            device=request.device,
            output_dir=root,
            name=request.analysis_id,
            extraction="make_fx_reference_v1",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "convert_request_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("unsupported op"),
        ),
    )

    result = api.analyze(_request(output))

    assert isinstance(result, AnalysisFailure)
    assert result.stage == "ir_conversion"
    assert result.reason_code == "ir_conversion_failed"
    assert not output.exists()


def test_packaged_profile_audit_artifact_unblocks_architecture_stage(tmp_path):
    # The packaged RX_9060_XT profile ships a verified locked-clock audit
    # artifact (audits/rx9060xt_resource_peaks_v3.json), so the architecture
    # stage's require_verified_audit_evidence gate passes and analysis proceeds
    # further. The dummy reference fails downstream (graph_extraction), not at
    # the audit gate.
    result = api.analyze(_request(tmp_path / "result"))

    assert isinstance(result, AnalysisFailure)
    assert result.stage != "architecture"
    assert "audit evidence unavailable" not in result.message


@pytest.mark.parametrize(
    "changes",
    [
        {"analysis_id": ""},
        {"reference_name": ""},
        {"reference_sha256": "A" * 64},
        {"reference_sha256": "a" * 63},
        {"atol": -1},
        {"rtol": float("nan")},
        {"max_error_cap": float("inf")},
        {"required_matched_ratio": 1.1},
    ],
)
def test_analysis_request_rejects_invalid_contract_fields(tmp_path, changes):
    conversion_values: dict[str, Any] = {
        "analysis_id": "analysis",
        "reference": lambda value: value,
        "input_factory": lambda seed: (seed,),
        "reference_name": "reference",
        "reference_sha256": "a" * 64,
    }
    policy_values: dict[str, Any] = {"atol": 1e-2, "rtol": 1e-2}
    if set(changes) <= set(policy_values) | {
        "max_error_cap",
        "required_matched_ratio",
    }:
        policy_values.update(changes)
    else:
        conversion_values.update(changes)
    with pytest.raises(ValueError):
        AnalysisRequest(
            conversion=ConversionRequest(
                **conversion_values,
                verification=VerificationPolicy(**policy_values),
            ),
            architecture={},
            output_dir=tmp_path / "result",
        )


def test_analyze_refuses_to_overwrite_existing_output(tmp_path):
    output = tmp_path / "result"
    output.mkdir()
    result = api.analyze(_request(output))
    assert isinstance(result, AnalysisFailure)
    assert result.stage == "prepare"
    assert result.reason_code == "output_exists"


def test_bound_and_reason_code_helpers_fail_closed():
    valid = api._extract_bound(
        {
            "schema_version": 3,
            "total": {"lower_bound_seconds": 0, "compute_resource": None},
            "metadata": {"bound_kind": "capacity_constrained_tile_aware_v1"},
        },
    )
    assert valid.seconds == 0
    assert valid.limiting_resource is None
    for seconds in (None, -1, float("nan")):
        with pytest.raises(ValueError, match="finite lower bound"):
            api._extract_bound(
                {
                    "schema_version": 3,
                    "total": {"lower_bound_seconds": seconds},
                    "metadata": {
                        "bound_kind": "capacity_constrained_tile_aware_v1",
                    },
                },
            )
    with pytest.raises(ValueError, match="non-tile-aware"):
        api._extract_bound(
            {
                "schema_version": 3,
                "total": {"lower_bound_seconds": 1},
                "metadata": {"bound_kind": "roofline"},
            },
        )
    with pytest.raises(ValueError, match="unsupported schema"):
        api._extract_bound({"schema_version": 0})
    assert pipeline.pipeline_reason_code(
        SolarStage.FORMAL_ANALYSIS,
        OrojenesisError("missing"),
    ) == ("toolchain_unavailable")
    assert (
        pipeline.pipeline_reason_code(
            SolarStage.CONVERSION_VERIFICATION,
            VerificationError("bad"),
        )
        == "conversion_not_proven"
    )
    assert pipeline.pipeline_reason_code(
        SolarStage.GRAPH_EXTRACTION, RuntimeError()
    ) == ("graph_extraction_failed")


def test_analysis_request_defaults_to_eq1_roofline_bound():
    """Eq.1 roofline is the default bound policy (paper-aligned); Orojenesis is opt-in."""
    request = _request(Path("/tmp/solar-default"))
    assert request.require_orojenesis is False


def test_default_api_matmul_is_paper_roofline_without_orojenesis(tmp_path):
    result = api.analyze(_matmul_request(tmp_path / "result"))

    assert isinstance(result, AnalysisResult)
    assert result.bound.kind == "roofline_eq1_v1"
    assert result.sol_score_eligible is True
    assert result.publication_eligible is False
    analysis = yaml.safe_load(
        (result.output_dir / "solar-analysis.yaml").read_text(),
    )
    evidence = analysis["metadata"]["orojenesis"]
    assert evidence["status"] == "not_requested"
    assert evidence["formal_coverage"] == {
        "applicable_layers": 0,
        "total_layers": 1,
    }
    manifest = yaml.safe_load((result.output_dir / "manifest.yaml").read_text())
    assert manifest["sol_score_eligible"] is True
    assert manifest["publication_eligible"] is False


def test_formal_api_matmul_requires_and_records_orojenesis_evidence(
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []

    class FakeRunner:
        def __init__(self) -> None:
            self.toolchain_identity = {"verification_mode": "fake"}

        def run_layer(self, layer, output_dir, *, word_bits):
            calls.append(layer["semantic_op"]["equation"])
            output_dir.mkdir(parents=True, exist_ok=True)
            raw = output_dir / "raw.csv"
            raw.write_text("64,80\n")
            return {
                "word_bits": word_bits,
                "curve": [{"buffer_bytes": 64, "dram_bytes": 80.0}],
                "evidence_files": {
                    "raw": {
                        "path": raw.name,
                        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                    },
                },
            }

    runner = FakeRunner()
    monkeypatch.setattr(
        "solar.analysis.graph_analyzer.OrojenesisRunner",
        lambda home: runner,
    )

    result = api.analyze(
        _matmul_request(tmp_path / "result", require_orojenesis=True),
    )

    assert isinstance(result, AnalysisResult)
    assert result.bound.kind == "capacity_constrained_tile_aware_v1"
    assert result.sol_score_eligible is True
    assert result.publication_eligible is True
    assert calls == ["MK,KN->MN"]
    analysis = yaml.safe_load(
        (result.output_dir / "solar-analysis.yaml").read_text(),
    )
    evidence = analysis["metadata"]["orojenesis"]
    assert evidence["status"] == "complete"
    assert evidence["formal_coverage"] == {
        "applicable_layers": 1,
        "total_layers": 1,
    }
    assert set(evidence["layers"]) == {"mm"}
    artifact_paths = {artifact.path for artifact in result.artifacts}
    assert "orojenesis/mm/raw.csv" in artifact_paths
    manifest = yaml.safe_load(
        (result.output_dir / "manifest.yaml").read_text(),
    )
    assert "orojenesis/mm/raw.csv" in {
        artifact["path"] for artifact in manifest["artifacts"]
    }


def test_incomplete_optional_orojenesis_evidence_falls_back_to_eq1(
    tmp_path,
    monkeypatch,
):
    class FakeRunner:
        def __init__(self) -> None:
            self.toolchain_identity = {"verification_mode": "fake"}

        def run_layer(self, layer, output_dir, *, word_bits):
            del layer, output_dir
            return {
                "word_bits": word_bits,
                "curve": [{"buffer_bytes": 64, "dram_bytes": 8_000.0}],
                "evidence_files": {},
            }

    monkeypatch.setattr(
        "solar.analysis.graph_analyzer.OrojenesisRunner",
        lambda home: FakeRunner(),
    )

    result = api.analyze(
        _matmul_request(
            tmp_path / "result",
            orojenesis_home="configured-but-incomplete",
        ),
    )

    assert isinstance(result, AnalysisResult)
    assert result.bound.kind == "roofline_eq1_v1"
    analysis = yaml.safe_load(
        (result.output_dir / "solar-analysis.yaml").read_text(),
    )
    assert analysis["metadata"]["orojenesis"]["status"] == "incomplete"
    assert (
        analysis["total"]["prefetched_bytes"]
        == analysis["total"]["fused_bytes"]
    )


def test_default_api_keeps_unmodeled_contraction_on_paper_roofline(tmp_path):
    result = api.analyze(
        _conv_request(tmp_path / "result", require_orojenesis=False),
    )

    assert isinstance(result, AnalysisResult)
    assert result.bound.kind == "roofline_eq1_v1"
    assert result.sol_score_eligible is True
    analysis = yaml.safe_load(
        (result.output_dir / "solar-analysis.yaml").read_text(),
    )
    evidence = analysis["metadata"]["orojenesis"]
    assert evidence["status"] == "not_requested"
    assert evidence["formal_coverage"] == {
        "applicable_layers": 0,
        "total_layers": 1,
    }


def test_strict_api_rejects_contraction_without_exact_orojenesis_proof(
    tmp_path,
    monkeypatch,
):
    class FakeRunner:
        def __init__(self) -> None:
            self.toolchain_identity = {"verification_mode": "fake"}

    monkeypatch.setattr(
        "solar.analysis.graph_analyzer.OrojenesisRunner",
        lambda home: FakeRunner(),
    )
    output = tmp_path / "result"

    result = api.analyze(_conv_request(output))

    assert isinstance(result, AnalysisFailure)
    assert result.stage == SolarStage.FORMAL_ANALYSIS
    assert result.reason_code == "formal_analysis_failed"
    assert "exact Orojenesis proof representation" in result.message
    assert "convolution" in result.message
    assert not output.exists()


def test_diagnostic_analysis_does_not_construct_orojenesis_runner(
    tmp_path,
    monkeypatch,
):
    observed: dict[str, object] = {}

    class FakeAnalyzer:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def analyze_graph(self, *args, **kwargs):
            observed.update(kwargs)
            return {"schema_version": 3}

    monkeypatch.setattr(
        "solar.analysis.graph_analyzer.OrojenesisRunner",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("runner must not be constructed"),
        ),
    )
    monkeypatch.setattr(
        "solar.analysis.graph_analyzer.IRGraphAnalyzer",
        FakeAnalyzer,
    )

    result = ir_lifecycle(IRKind.ATEN).analyze(
        _request(tmp_path / "result"),
        cast(Any, _Profile()),
        tmp_path,
        IRGraphArtifact(tmp_path / "ir_graph.yaml", IRKind.ATEN),
    )

    assert result == {"schema_version": 3}
    assert observed["orojenesis_runner"] is None
    assert observed["require_orojenesis"] is False


def test_extract_bound_accepts_paper_roofline_when_orojenesis_not_required():
    bound = api._extract_bound(
        {
            "schema_version": 3,
            "total": {"lower_bound_seconds": 1.5, "compute_resource": "mfma"},
            "metadata": {"bound_kind": "roofline_eq1_v1"},
        },
        require_orojenesis=False,
    )
    assert bound.seconds == 1.5
    assert bound.kind == "roofline_eq1_v1"
    assert bound.limiting_resource == "mfma"


def test_extract_bound_rejects_roofline_when_tile_evidence_is_required():
    with pytest.raises(ValueError, match="non-tile-aware"):
        api._extract_bound(
            {
                "schema_version": 3,
                "total": {"lower_bound_seconds": 1.5},
                "metadata": {"bound_kind": "roofline_eq1_v1"},
            },
            require_orojenesis=True,
        )


def test_extract_bound_rejects_unknown_kind_even_when_orojenesis_not_required():
    with pytest.raises(ValueError, match="unsupported bound kind"):
        api._extract_bound(
            {
                "schema_version": 3,
                "total": {"lower_bound_seconds": 1.5},
                "metadata": {"bound_kind": "roofline"},
            },
            require_orojenesis=False,
        )
