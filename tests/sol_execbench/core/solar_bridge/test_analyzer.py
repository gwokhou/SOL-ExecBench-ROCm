from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import NumericCheck, Workload
from sol_execbench.core.solar_bridge import analyzer, workload_context
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarAnalysisStatus,
    SolarStage,
    SolarWorkerRequest,
    formal_precision_for_definition,
)
from sol_execbench.core.solar_bridge.workload_context import (
    SolarWorkloadContext,
)
from solar.api import AnalysisFailure, AnalysisResult, ArtifactRef, SOLBound
from solar.ir.contracts import IRPath

_FORMAL_ARTIFACTS = tuple(
    ArtifactRef(path, "b" * 64)
    for path in (
        "operator_graph.yaml",
        "einsum_graph.yaml",
        "conversion-attestation.yaml",
        "solar-analysis.yaml",
    )
)


def _definition(dtype: str = "torch.float16") -> Definition:
    tensor = SimpleNamespace(dtype=dtype)
    return cast(
        Definition,
        SimpleNamespace(
            name="problem",
            reference="def reference(x): return x",
            custom_inputs_entrypoint=None,
            inputs={"x": tensor},
            outputs={"y": tensor},
        ),
    )


def _workload() -> Workload:
    return cast(
        Workload,
        SimpleNamespace(
            uuid="workload-1",
            checks=[
                NumericCheck(
                    output="y",
                    max_atol=0.01,
                    max_rtol=0.02,
                    required_matched_ratio=1.0,
                ),
            ],
        ),
    )


def _context() -> SolarWorkloadContext:
    return SolarWorkloadContext(
        _definition(),
        _workload(),
        lambda value: value,
        lambda seed: (seed,),
    )


def test_analyze_workload_adapts_outer_models_to_solar(
    tmp_path,
    monkeypatch,
) -> None:
    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "definition.json").write_text("{}")
    (problem / "workload.jsonl").write_text("{}\n")
    definition = _definition()
    workload = _workload()

    def reference(value):
        return value

    def input_factory(seed):
        return (seed,)

    expected = SolarAnalysisOutcome(
        status=SolarAnalysisStatus.ANALYZED,
        analysis_id="problem:workload-1",
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(analyzer, "require_formal_device", lambda device: None)
    context = SolarWorkloadContext(
        definition,
        workload,
        reference,
        input_factory,
    )
    monkeypatch.setattr(
        analyzer,
        "load_solar_workload_context",
        lambda *_args: context,
    )

    def fake_invoke(**kwargs):
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(analyzer, "_invoke_solar", fake_invoke)

    outcome = analyzer.analyze_workload(
        problem_dir=problem,
        workload_uuid="workload-1",
        output_dir=tmp_path / "output",
        device="hip:0",
        orojenesis_home=None,
    )

    assert outcome is expected
    assert observed["context"] is context


def test_invoke_solar_maps_failure_without_claiming_bound(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        analyzer,
        "formal_precision_for_definition",
        lambda value: "fp16",
    )
    monkeypatch.setattr(
        "solar.api.analyze",
        lambda request: AnalysisFailure(
            status=SolarAnalysisStatus.FAILED,
            analysis_id=request.analysis_id,
            stage=SolarStage.FORMAL_ANALYSIS,
            reason_code="bound_failed",
            message="unsupported",
        ),
    )

    outcome = analyzer._invoke_solar(
        context=_context(),
        output_dir=tmp_path,
        device="hip:0",
        orojenesis_home=None,
    )

    assert outcome.status == "failed"
    assert outcome.reason_code == "bound_failed"
    assert outcome.lower_bound_seconds is None


def test_invoke_solar_maps_successful_bound_and_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    result_dir = tmp_path / "result"
    monkeypatch.setattr(
        analyzer,
        "formal_precision_for_definition",
        lambda value: "fp16",
    )

    def fake_analyze(request):
        assert request.require_orojenesis is True
        return AnalysisResult(
            status=SolarAnalysisStatus.ANALYZED,
            analysis_id=request.analysis_id,
            output_dir=result_dir,
            architecture_sha256="a" * 64,
            artifacts=_FORMAL_ARTIFACTS,
            bound=SOLBound(
                0.001,
                "capacity_constrained_tile_aware_v1",
                "memory",
            ),
        )

    monkeypatch.setattr("solar.api.analyze", fake_analyze)

    outcome = analyzer._invoke_solar(
        context=_context(),
        output_dir=result_dir,
        device="hip:0",
        orojenesis_home=None,
    )

    assert outcome.status == "analyzed"
    assert outcome.lower_bound_seconds == 0.001
    assert outcome.limiting_resource == "memory"
    assert outcome.publication_eligible is True
    assert {artifact["path"] for artifact in outcome.artifacts} == {
        artifact.path for artifact in _FORMAL_ARTIFACTS
    }


def test_invoke_solar_rejects_non_formal_result(tmp_path, monkeypatch) -> None:
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    monkeypatch.setattr(
        analyzer,
        "formal_precision_for_definition",
        lambda value: "fp16",
    )
    monkeypatch.setattr(
        "solar.api.analyze",
        lambda request: AnalysisResult(
            status=SolarAnalysisStatus.ANALYZED,
            analysis_id=request.analysis_id,
            output_dir=result_dir,
            architecture_sha256="a" * 64,
            artifacts=_FORMAL_ARTIFACTS,
            bound=SOLBound(0.001, "diagnostic", "memory"),
        ),
    )

    outcome = analyzer._invoke_solar(
        context=_context(),
        output_dir=result_dir,
        device="hip:0",
        orojenesis_home=None,
    )

    assert outcome.status == "failed"
    assert outcome.stage == "formal_acceptance"
    assert outcome.reason_code == "non_formal_bound"
    assert not result_dir.exists()


def test_select_workload_requires_exact_uuid_match() -> None:
    workload = _workload()

    assert workload_context._select_workload([workload], "workload-1") == (
        0,
        workload,
    )
    with pytest.raises(ValueError, match="match exactly once"):
        workload_context._select_workload([workload, workload], "workload-1")


def test_formal_device_requires_rocm_gfx1200(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "test-rocm")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(gcnArchName="gfx942:sramecc+"),
    )

    with pytest.raises(RuntimeError, match="requires gfx1200"):
        analyzer.require_formal_device("cuda")


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("torch.float8_e4m3fn", "fp8"),
        ("torch.bfloat16", "bf16"),
        ("torch.float16", "fp16"),
        ("torch.float32", "fp32"),
    ],
)
def test_formal_precision_follows_tensor_contract(dtype, expected) -> None:
    assert formal_precision_for_definition(_definition(dtype)) == expected


def test_worker_request_round_trips_optional_home() -> None:
    request = SolarWorkerRequest(
        problem_dir="problem",
        workload_uuid="workload-1",
        output_dir="output",
        device="hip:0",
        orojenesis_home="orojenesis",
        ir_path=IRPath.MAKE_FX_ATEN,
    )

    assert SolarWorkerRequest.from_dict(request.to_dict()) == request
    assert request.to_dict()["ir_path"] == "make_fx_aten"
