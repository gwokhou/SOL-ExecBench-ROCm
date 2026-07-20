from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.solar_bridge import analyzer
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarWorkerRequest,
    formal_precision_for_definition,
)
from solar.api import AnalysisFailure, AnalysisResult, ArtifactRef, SolBound


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
    tolerance = SimpleNamespace(
        max_atol=0.01,
        max_rtol=0.02,
        required_matched_ratio=1.0,
        max_error_cap=None,
        allow_negative_inf=False,
    )
    return cast(Workload, SimpleNamespace(uuid="workload-1", tolerance=tolerance))


def test_analyze_workload_adapts_outer_models_to_solar(tmp_path, monkeypatch) -> None:
    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "definition.json").write_text("{}")
    (problem / "workload.jsonl").write_text("{}\n")
    definition = _definition()
    workload = _workload()
    reference_module = object()

    def reference(value):
        return value

    def input_factory(seed):
        return (seed,)

    expected = SolarAnalysisOutcome(status="analyzed", analysis_id="problem:workload-1")
    observed: dict[str, object] = {}

    monkeypatch.setattr(analyzer, "_require_formal_device", lambda device: None)
    monkeypatch.setattr(
        analyzer.Definition, "model_validate_json", lambda payload: definition
    )
    monkeypatch.setattr(analyzer, "_load_workloads", lambda path: [workload])
    monkeypatch.setattr(
        analyzer,
        "load_reference_function",
        lambda source: (reference_module, reference),
    )
    monkeypatch.setattr(
        analyzer,
        "build_input_factory",
        lambda *args: input_factory,
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
    assert observed["definition"] is definition
    assert observed["workload"] is workload
    assert observed["reference"] is reference
    assert observed["input_factory"] is input_factory


def test_invoke_solar_maps_failure_without_claiming_bound(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        analyzer, "formal_precision_for_definition", lambda value: "fp16"
    )
    monkeypatch.setattr(
        "solar.api.analyze",
        lambda request: AnalysisFailure(
            status="failed",
            analysis_id=request.analysis_id,
            stage="formal_analysis",
            reason_code="bound_failed",
            message="unsupported",
        ),
    )

    outcome = analyzer._invoke_solar(
        definition=_definition(),
        workload=_workload(),
        reference=lambda value: value,
        input_factory=lambda seed: (seed,),
        output_dir=tmp_path,
        device="hip:0",
        orojenesis_home=None,
    )

    assert outcome.status == "failed"
    assert outcome.reason_code == "bound_failed"
    assert outcome.lower_bound_seconds is None


def test_invoke_solar_maps_successful_bound_and_artifacts(
    tmp_path, monkeypatch
) -> None:
    result_dir = tmp_path / "result"
    monkeypatch.setattr(
        analyzer, "formal_precision_for_definition", lambda value: "fp16"
    )
    monkeypatch.setattr(
        "solar.api.analyze",
        lambda request: AnalysisResult(
            status="analyzed",
            analysis_id=request.analysis_id,
            output_dir=result_dir,
            architecture_sha256="a" * 64,
            artifacts=(ArtifactRef("manifest.yaml", "b" * 64),),
            bound=SolBound(0.001, "roofline", "memory"),
        ),
    )

    outcome = analyzer._invoke_solar(
        definition=_definition(),
        workload=_workload(),
        reference=lambda value: value,
        input_factory=lambda seed: (seed,),
        output_dir=result_dir,
        device="hip:0",
        orojenesis_home=None,
    )

    assert outcome.status == "analyzed"
    assert outcome.lower_bound_seconds == 0.001
    assert outcome.limiting_resource == "memory"
    assert outcome.artifacts == ({"path": "manifest.yaml", "sha256": "b" * 64},)


def test_select_workload_requires_exact_uuid_match() -> None:
    workload = _workload()

    assert analyzer._select_workload([workload], "workload-1") == (0, workload)
    with pytest.raises(ValueError, match="match exactly once"):
        analyzer._select_workload([workload, workload], "workload-1")


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
        analyzer._require_formal_device("cuda")


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
    )

    assert SolarWorkerRequest.from_dict(request.to_dict()) == request
