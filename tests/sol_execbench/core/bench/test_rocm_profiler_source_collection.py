from __future__ import annotations

from collections.abc import Sequence
import subprocess

import pytest

from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3CollectionResult,
    Rocprofv3ProfileRequest,
    Rocprofv3ProfileResult,
    Rocprofv3ProfileStatus,
    SourceTimingRequest,
    collect_rocprofv3_profile,
    collect_source_timing_evidence,
    discover_rocprofv3_artifacts,
    select_default_timing,
)
from sol_execbench.core.bench.timing_policy import (
    TimingBackend,
    TimingSourceType,
    select_timing_policy,
)


ROCPROFV3_CSV = """Domain,Name,Start_Timestamp,End_Timestamp,Duration(ns)
KERNEL_DISPATCH,rmsnorm_kernel,1000,5000,4000
HIP_RUNTIME_API,hipLaunchKernel,900,5100,4200
KERNEL_DISPATCH,post_kernel,6000,9000,3000
"""


def test_source_collection_selects_triton_rocprofv3_and_records_run_config(tmp_path):
    calls: list[list[str]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        (tmp_path / "timing.csv").write_text(ROCPROFV3_CSV)
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout="profiled",
            stderr="",
        )

    result = collect_source_timing_evidence(
        SourceTimingRequest(
            application_command=("uv", "run", "sol-execbench", "problem"),
            languages=("triton",),
            output_directory=tmp_path,
            output_file="timing",
            tool_version="rocprofv3 7.0.0",
            gpu_architecture="gfx942",
            warmup_runs=5,
            iterations=50,
            trial_count=2,
            clock_locked=False,
            timeout_seconds=41.0,
        ),
        runner=runner,
    )

    assert calls
    assert result.profiler_collected is True
    assert result.evidence is not None
    payload = result.evidence.to_dict()
    assert payload["backend"] == "rocprofv3"
    assert payload["gpu_architecture"] == "gfx942"
    assert payload["warmup_runs"] == 5
    assert payload["iterations"] == 50
    assert payload["trial_count"] == 2
    assert payload["clock_locked"] is False


def test_source_collection_selects_hip_native_rocprofv3(tmp_path):
    calls: list[list[str]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        (tmp_path / "timing.csv").write_text(ROCPROFV3_CSV)
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout="profiled",
            stderr="",
        )

    result = collect_source_timing_evidence(
        SourceTimingRequest(
            application_command=("uv", "run", "sol-execbench", "problem"),
            languages=("hip_cpp",),
            output_directory=tmp_path,
            output_file="timing",
            tool_version="rocprofv3 7.0.0",
            gpu_architecture="gfx1200",
        ),
        runner=runner,
    )

    assert calls
    assert result.profiler_collected is True
    assert result.selection.profiler_backed is True
    assert result.selection.policy.backend == TimingBackend.ROCPROFV3
    assert result.evidence is not None
    assert result.evidence.activity_domain == "kernel_activity"
    assert result.evidence.backend == "rocprofv3"


def test_source_collection_routes_pytorch_to_explicit_fallback(tmp_path):
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"runner should not be called: {command}")

    result = collect_source_timing_evidence(
        SourceTimingRequest(
            application_command=("uv", "run", "sol-execbench", "problem"),
            languages=("pytorch",),
            output_directory=tmp_path,
            output_file="timing",
            tool_version="rocprofv3 7.0.0",
            gpu_architecture="gfx942",
            warmup_runs=5,
            iterations=50,
            trial_count=1,
            clock_locked=True,
        ),
        runner=runner,
    )

    assert result.profiler_collected is False
    assert result.evidence is None
    assert result.selection.fallback_applied is True
    assert result.selection.policy.backend == TimingBackend.DEVICE_EVENTS


def test_discover_with_empty_output_file_does_not_register_every_file(tmp_path):
    (tmp_path / "unrelated.csv").write_text("a,b\n1,2\n")
    (tmp_path / "notes.txt").write_text("notes")
    (tmp_path / "metadata.json").write_text("{}")

    artifacts = discover_rocprofv3_artifacts(tmp_path, "")

    assert [artifact.path.name for artifact in artifacts] == []


def test_profile_collection_timeout_keeps_partial_artifacts_and_reasons(tmp_path):
    def runner(command, cwd, timeout):
        (tmp_path / "profile_counters.csv").write_text(
            "Metric,Value,Unit\nSQ_INSTS_VALU,1,count\n"
        )
        raise subprocess.TimeoutExpired(cmd=list(command), timeout=timeout or 0)

    request = Rocprofv3ProfileRequest(
        application_command=("python", "eval_driver.py"),
        output_directory=tmp_path,
        output_file="profile",
        timeout_seconds=5,
    )

    result = collect_rocprofv3_profile(request, runner=runner)
    payload = result.to_dict()

    assert result.status == "failed"
    assert "rocprof_command_timeout" in payload["reason_codes"]
    assert "rocprof_command_failed" not in payload["reason_codes"]
    assert payload["artifact_coverage_status"] == "partial"
    assert payload["profiler_data_artifacts"] is True
    assert any(artifact["kind"] == "counter_csv" for artifact in payload["artifacts"])


def test_profile_result_rejects_contradictory_success_reason(tmp_path) -> None:
    with pytest.raises(ValueError, match="successful profiling"):
        Rocprofv3ProfileResult(
            status=Rocprofv3ProfileStatus.SUCCESS,
            command=("rocprofv3",),
            output_directory=tmp_path,
            output_file="profile",
            skipped_reason="profiler unavailable",
        )


def test_collection_result_requires_evidence_for_profiler_selection() -> None:
    selection = select_default_timing(
        select_timing_policy(TimingSourceType.HIP_NATIVE),
        rocprofv3_available=True,
    )

    with pytest.raises(ValueError, match="requires timing evidence"):
        Rocprofv3CollectionResult(evidence=None, selection=selection)
