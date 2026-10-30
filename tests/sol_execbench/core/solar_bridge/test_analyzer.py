from __future__ import annotations

import fcntl
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import NumericCheck, Workload
from sol_execbench.core.solar_bridge import (
    analyzer,
    formal_device,
    invocation,
    workload_context,
)
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarAnalysisStatus,
    SolarStage,
    SolarWorkerRequest,
    formal_precision_for_definition,
)
from sol_execbench.core.solar_bridge.semantic_metadata import (
    performance_analysis_metadata,
)
from sol_execbench.core.solar_bridge.workload_context import (
    SolarWorkloadContext,
)
from solar.api import AnalysisFailure, AnalysisResult, ArtifactRef, SOLBound
from solar.ir.access_artifact import ACCESS_FILENAME
from solar.ir.contracts import IRPath
from solar.rocm.architecture import ArchitectureProfile

_FORMAL_ARTIFACTS = tuple(
    ArtifactRef(path, "b" * 64)
    for path in (
        "operator_graph.yaml",
        "einsum_graph.yaml",
        "conversion-attestation.yaml",
        ACCESS_FILENAME,
        "solar-analysis.yaml",
    )
)


def _definition(dtype: str = "torch.float16") -> Definition:
    tensor = SimpleNamespace(dtype=dtype)
    return cast(
        Definition,
        SimpleNamespace(
            name="problem",
            op_type="elementwise",
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
            inputs={},
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

    monkeypatch.setattr(analyzer, "invoke_solar", fake_invoke)

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
        invocation,
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

    outcome = invocation.invoke_solar(
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
    cleanup_devices: list[str] = []
    monkeypatch.setattr(
        invocation,
        "formal_precision_for_definition",
        lambda value: "fp16",
    )
    monkeypatch.setattr(
        formal_device,
        "release_formal_device_memory",
        cleanup_devices.append,
    )

    def fake_analyze(request):
        assert request.require_orojenesis is True
        assert request.execution_policy.device_stage_cleanup is not None
        request.execution_policy.device_stage_cleanup()
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

    outcome = invocation.invoke_solar(
        context=_context(),
        output_dir=result_dir,
        device="hip:0",
        orojenesis_home=None,
        device_stage_lock_path=tmp_path / "device.lock",
    )

    assert outcome.status == "analyzed"
    assert outcome.lower_bound_seconds == 0.001
    assert outcome.limiting_resource == "memory"
    assert outcome.publication_eligible is True
    assert cleanup_devices == ["hip:0"]
    assert {artifact["path"] for artifact in outcome.artifacts} == {
        artifact.path for artifact in _FORMAL_ARTIFACTS
    }


def test_performance_metadata_marks_controlled_concurrent_graph() -> None:
    definition = cast(
        Definition,
        SimpleNamespace(op_type="concurrent_graph"),
    )
    context = SolarWorkloadContext(
        definition,
        _workload(),
        lambda value: value,
        lambda seed: (seed,),
    )

    assert performance_analysis_metadata(context) == {
        "performance_semantics": {
            "graph_class": "concurrent_graph",
        }
    }


def test_performance_metadata_requires_exact_minigpt_contract() -> None:
    definition = cast(
        Definition,
        SimpleNamespace(
            op_type="transformer_block",
            inputs={
                "input": SimpleNamespace(dtype=SimpleNamespace(value="float32"))
            },
            get_resolved_axes_values=lambda axes: {
                **axes,
                "N": 768,
                "M": axes["S"],
            },
        ),
    )
    workload = cast(
        Workload,
        SimpleNamespace(axes={"S": 128}),
    )
    context = SolarWorkloadContext(
        definition,
        workload,
        lambda value: value,
        lambda seed: (seed,),
    )

    assert performance_analysis_metadata(context) == {
        "performance_semantics": {
            "graph_class": "transformer_block",
            "hidden_size": 768,
            "num_heads": 8,
            "sequence_length": 128,
            "dtype": "float32",
        }
    }


def test_invoke_solar_rejects_non_formal_result(tmp_path, monkeypatch) -> None:
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    monkeypatch.setattr(
        invocation,
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

    outcome = invocation.invoke_solar(
        context=_context(),
        output_dir=result_dir,
        device="hip:0",
        orojenesis_home=None,
    )

    assert outcome.status == "failed"
    assert outcome.stage == "formal_acceptance"
    assert outcome.reason_code == "non_formal_bound"
    assert not result_dir.exists()


def test_invoke_solar_non_formal_path_keeps_non_formal_bound(
    tmp_path,
    monkeypatch,
) -> None:
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    monkeypatch.setattr(
        invocation,
        "formal_precision_for_definition",
        lambda value: "fp16",
    )

    observed: dict[str, object] = {}

    def fake_analyze(request):
        observed["require_orojenesis"] = request.require_orojenesis
        observed["require_verified_audit"] = request.require_verified_audit
        return AnalysisResult(
            status=SolarAnalysisStatus.ANALYZED,
            analysis_id=request.analysis_id,
            output_dir=result_dir,
            architecture_sha256="a" * 64,
            artifacts=_FORMAL_ARTIFACTS,
            bound=SOLBound(0.001, "roofline_eq1_v1", "memory"),
        )

    monkeypatch.setattr("solar.api.analyze", fake_analyze)

    outcome = invocation.invoke_solar(
        context=_context(),
        output_dir=result_dir,
        device="hip:0",
        orojenesis_home=None,
        architecture="MI300X",
        formal=False,
    )

    assert outcome.status == "analyzed"
    assert outcome.lower_bound_seconds == 0.001
    assert outcome.publication_eligible is False
    assert observed["require_orojenesis"] is False
    assert observed["require_verified_audit"] is False
    assert result_dir.exists()


def test_architecture_for_gfx_target_maps_cdna3_and_rdna4() -> None:
    assert analyzer._architecture_for_gfx_target("gfx1200") == "RX_9060_XT"
    assert analyzer._architecture_for_gfx_target("gfx942") == "MI300X"
    with pytest.raises(ValueError, match="unsupported_solar_architecture"):
        analyzer._architecture_for_gfx_target("gfx1100")


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


@pytest.mark.parametrize("custom_path", [False, True])
def test_default_solar_stage_shares_native_lock_without_nested_deadlock(
    tmp_path,
    monkeypatch,
    custom_path,
) -> None:
    from sol_execbench.core.bench.gpu_lock import acquire_evaluation_gpu_lock
    from solar.pipeline import analysis as pipeline

    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.delenv("GPU_LOCK_HELD_BY_HOST", raising=False)
    cleaned = []
    monkeypatch.setattr(
        formal_device, "release_formal_device_memory", cleaned.append
    )
    lock_path = (
        tmp_path / "gpu-11111111-2222-3333-4444-555555555555.L1-ownership.lock"
    )

    def extract(*_args):
        with lock_path.open("a+") as handle, pytest.raises(BlockingIOError):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with acquire_evaluation_gpu_lock(timeout_seconds=0.01):
            pass
        raise ValueError("test stage failure")

    monkeypatch.setattr(pipeline, "extract_request_graph", extract)

    def analyze(request):
        with pytest.raises(
            pipeline.PipelineStageError, match="test stage failure"
        ):
            pipeline.run_pipeline(
                request, cast(ArchitectureProfile, None), tmp_path
            )
        return AnalysisFailure(
            status=SolarAnalysisStatus.FAILED,
            analysis_id=request.analysis_id,
            stage=SolarStage.GRAPH_EXTRACTION,
            reason_code="test_failure",
            message="test stage failure",
        )

    monkeypatch.setattr("solar.api.analyze", analyze)
    invocation.invoke_solar(
        context=_context(),
        output_dir=tmp_path,
        device="cuda:0",
        orojenesis_home=None,
        device_stage_lock_path=lock_path if custom_path else None,
        device_stage_lock_timeout_seconds=0.02,
    )
    assert cleaned == ["cuda:0"]


def test_cpu_analysis_preserves_explicit_serialization(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))
    path = (
        tmp_path / "gpu-11111111-2222-3333-4444-555555555555.L1-ownership.lock"
    )
    policy = formal_device.device_execution_policy("cpu", path, 1)
    assert policy.device_stage_lock_path == path
    assert policy.device_stage_context is None
    assert policy.device_stage_cleanup is None


pytestmark = pytest.mark.usefixtures("native_gpu_cpu_observation")


@pytest.mark.parametrize(
    "field",
    ["peak_ops_per_second", "precision_support", "audit_evidence"],
)
def test_corpus_identity_includes_rates_support_and_audit(field):
    from sol_execbench.core.dataset.aka_corpus import FORMAL_ARCHITECTURE_SHA256
    from solar.api import architecture_profile_sha256
    from solar.rocm.architecture import ArchitectureProfile

    data = ArchitectureProfile.load("RX_9060_XT").to_dict()
    if field == "peak_ops_per_second":
        data[field]["fp32"] *= 2
    elif field == "precision_support":
        data[field]["fp8"]["limitation"] = "Changed runtime qualification"
    else:
        data[field]["sha256"] = "0" * 64
    assert architecture_profile_sha256(data) != FORMAL_ARCHITECTURE_SHA256


def test_corpus_profile_retains_verified_audit_evidence():
    ArchitectureProfile.load("RX_9060_XT").require_verified_audit_evidence()
