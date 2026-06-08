from __future__ import annotations

import subprocess
from collections.abc import Sequence

from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3CollectionRequest,
    Rocprofv3ProfileRequest,
    build_rocprofv3_profile_command,
    build_rocprofv3_command,
    build_timing_evidence,
    collect_rocprofv3_profile,
    collect_source_timing_evidence,
    discover_rocprofv3_artifacts,
    collect_rocprofv3_timing,
    parse_rocprofv3_csv,
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


def test_build_rocprofv3_command_places_application_after_separator():
    command = build_rocprofv3_command(
        ["uv", "run", "sol-execbench", "problem"],
        output_directory="out/prof",
        output_file="timing",
    )

    separator_index = command.index("--")
    assert command[:separator_index] == [
        "rocprofv3",
        "--kernel-trace",
        "--hip-runtime-trace",
        "--output-format",
        "csv",
        "--output-directory",
        "out/prof",
        "--output-file",
        "timing",
    ]
    assert command[separator_index + 1 :] == ["uv", "run", "sol-execbench", "problem"]


def test_build_rocprofv3_profile_command_prefers_rocpd_artifacts():
    command = build_rocprofv3_profile_command(
        ["python", "eval_driver.py"],
        output_directory="out/profile",
        output_file="profile",
    )

    separator_index = command.index("--")
    assert command[:separator_index] == [
        "rocprofv3",
        "--kernel-trace",
        "--hip-runtime-trace",
        "--output-format",
        "rocpd",
        "--output-directory",
        "out/profile",
        "--output-file",
        "profile",
    ]
    assert command[separator_index + 1 :] == ["python", "eval_driver.py"]


def test_profile_artifact_discovery_classifies_rocpd_and_csv_outputs(tmp_path):
    (tmp_path / "profile.rocpd").write_text("db")
    (tmp_path / "profile_kernel.csv").write_text("kernel")
    (tmp_path / "profile_agent_info.csv").write_text("agent")
    (tmp_path / "unrelated.csv").write_text("skip")

    artifacts = discover_rocprofv3_artifacts(tmp_path, "profile")

    assert sorted(artifact.kind for artifact in artifacts) == [
        "agent_info_csv",
        "rocpd",
        "trace_csv",
    ]
    assert all(artifact.size_bytes > 0 for artifact in artifacts)


def test_profile_collection_records_success_metadata(tmp_path):
    calls: list[list[str]] = []

    def runner(
        command: Sequence[str],
        cwd,
        timeout,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        (tmp_path / "profile.rocpd").write_text("profile db")
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout='{"definition": "demo"}\n',
            stderr="profiler note",
        )

    request = Rocprofv3ProfileRequest(
        application_command=("python", "eval_driver.py"),
        output_directory=tmp_path,
        output_file="profile",
        working_directory=tmp_path,
        timeout_seconds=30,
    )

    result = collect_rocprofv3_profile(request, runner=runner)
    payload = result.to_dict()

    assert result.succeeded is True
    assert calls[0][-3:] == ["--", "python", "eval_driver.py"]
    assert payload["schema_version"] == "sol_execbench.rocprofv3_profile.v1"
    assert payload["diagnostic_only"] is True
    assert payload["score_authority"] is False
    assert payload["status"] == "success"
    assert payload["working_directory"] == str(tmp_path)
    assert payload["timeout_seconds"] == 30
    assert payload["returncode"] == 0
    assert payload["artifacts"][0]["kind"] == "rocpd"
    assert payload["stderr_tail"] == "profiler note"


def test_profile_collection_unavailable_is_nonfatal_metadata(tmp_path):
    def runner(
        command: Sequence[str], cwd, timeout
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"runner should not be called: {command}")

    request = Rocprofv3ProfileRequest(
        application_command=("python", "eval_driver.py"),
        output_directory=tmp_path,
        output_file="profile",
    )

    result = collect_rocprofv3_profile(
        request,
        rocprofv3_available=False,
        runner=runner,
    )

    assert result.succeeded is False
    assert result.status == "unavailable"
    assert result.returncode is None
    assert result.skipped_reason == "rocprofv3 is not available on PATH"
    assert result.to_dict()["artifacts"] == []


def test_profile_collection_failure_records_artifact_and_stderr_tail(tmp_path):
    def runner(
        command: Sequence[str],
        cwd,
        timeout,
    ) -> subprocess.CompletedProcess[str]:
        (tmp_path / "profile_kernel.csv").write_text("partial")
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=22,
            stdout="",
            stderr="profiler failed",
        )

    request = Rocprofv3ProfileRequest(
        application_command=("python", "eval_driver.py"),
        output_directory=tmp_path,
        output_file="profile",
    )

    result = collect_rocprofv3_profile(request, runner=runner)
    payload = result.to_dict()

    assert result.status == "failed"
    assert payload["returncode"] == 22
    assert payload["failed_reason"] == "rocprofv3 command failed with exit code 22"
    assert payload["stderr_tail"] == "profiler failed"
    assert payload["artifacts"][0]["kind"] == "trace_csv"


def test_parse_rocprofv3_csv_keeps_domains_separate():
    rows = parse_rocprofv3_csv(ROCPROFV3_CSV)

    assert len(rows) == 3
    assert [row.name for row in rows if row.is_kernel_activity] == [
        "rmsnorm_kernel",
        "post_kernel",
    ]
    assert [row.name for row in rows if not row.is_kernel_activity] == [
        "hipLaunchKernel"
    ]
    assert sum(row.duration_ms for row in rows if row.is_kernel_activity) == 0.007


def test_timing_evidence_contains_auditable_profiler_fields():
    policy = select_timing_policy(TimingSourceType.HIP_NATIVE)
    evidence = build_timing_evidence(
        policy=policy,
        csv_content=ROCPROFV3_CSV,
        tool_version="rocprofv3 7.0.0",
        gpu_architecture="gfx1200",
        warmup_runs=3,
        iterations=11,
        trial_count=1,
        clock_locked=True,
    )
    payload = evidence.to_dict()

    assert payload["derived"] is True
    assert payload["canonical_output"] == "trace_jsonl"
    assert payload["tool_version"] == "rocprofv3 7.0.0"
    assert payload["gpu_architecture"] == "gfx1200"
    assert payload["activity_domain"] == "kernel_activity"
    assert payload["aggregation_rule"] == policy.aggregation_rule
    assert payload["backend"] == "rocprofv3"
    assert payload["interpretation"] == policy.interpretation
    assert payload["warmup_runs"] == 3
    assert payload["iterations"] == 11
    assert payload["trial_count"] == 1
    assert payload["clock_locked"] is True
    assert payload["kernel_duration_ms"] == 0.007
    assert len(payload["parsed_rows"]) == 3


def test_default_timing_selects_rocprofv3_for_kernel_activity_policies():
    for source_type in (TimingSourceType.HIP_NATIVE, TimingSourceType.TRITON):
        policy = select_timing_policy(source_type)
        selection = select_default_timing(policy, rocprofv3_available=True)

        assert selection.profiler_backed is True
        assert selection.fallback_applied is False
        assert selection.policy.backend == TimingBackend.ROCPROFV3


def test_default_timing_fallback_is_explicit_when_rocprofv3_unavailable():
    policy = select_timing_policy(TimingSourceType.TRITON)
    selection = select_default_timing(policy, rocprofv3_available=False)

    assert selection.profiler_backed is False
    assert selection.fallback_applied is True
    assert selection.policy.backend == TimingBackend.DEVICE_EVENTS
    assert "rocprofv3 is unavailable" in selection.reason
    assert selection.policy.reason == selection.reason


def test_pytorch_policy_does_not_masquerade_as_rocprofv3_kernel_activity():
    policy = select_timing_policy(TimingSourceType.PYTORCH)
    selection = select_default_timing(policy, rocprofv3_available=True)

    assert selection.profiler_backed is False
    assert selection.fallback_applied is True
    assert selection.policy.backend == TimingBackend.DEVICE_EVENTS
    assert "not rocprofv3 kernel activity timing" in selection.reason
    assert selection.policy.reason == selection.reason


def test_live_collection_invokes_runner_and_reads_generated_csv(tmp_path):
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

    request = Rocprofv3CollectionRequest(
        application_command=("uv", "run", "sol-execbench", "problem"),
        output_directory=tmp_path,
        output_file="timing",
        policy=select_timing_policy(TimingSourceType.TRITON),
        tool_version="rocprofv3 7.0.0",
        gpu_architecture="gfx1200",
    )

    result = collect_rocprofv3_timing(request, runner=runner)
    payload = result.to_dict()

    assert result.profiler_collected is True
    assert calls
    assert calls[0][:3] == ["rocprofv3", "--kernel-trace", "--hip-runtime-trace"]
    assert calls[0][-4:] == ["uv", "run", "sol-execbench", "problem"]
    assert result.evidence is not None
    assert result.evidence.kernel_duration_ms == 0.007
    assert payload["evidence"]["backend"] == "rocprofv3"
    assert payload["csv_path"].endswith("timing.csv")
    assert payload["stdout"] == "profiled"


def test_live_collection_returns_fallback_for_non_rocprofv3_policy(tmp_path):
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"runner should not be called: {command}")

    request = Rocprofv3CollectionRequest(
        application_command=("python", "-m", "benchmark"),
        output_directory=tmp_path,
        output_file="timing",
        policy=select_timing_policy(TimingSourceType.PYTORCH),
        tool_version="rocprofv3 7.0.0",
        gpu_architecture="gfx1200",
    )

    result = collect_rocprofv3_timing(request, runner=runner)

    assert result.profiler_collected is False
    assert result.evidence is None
    assert result.selection.fallback_applied is True
    assert result.selection.policy.backend == TimingBackend.DEVICE_EVENTS
    assert "not rocprofv3 kernel activity timing" in result.selection.reason
    assert result.selection.policy.reason == result.selection.reason


def test_live_collection_labels_failed_profiler_as_fallback(tmp_path):
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=17,
            stdout="",
            stderr="profiler failed",
        )

    request = Rocprofv3CollectionRequest(
        application_command=("uv", "run", "sol-execbench", "problem"),
        output_directory=tmp_path,
        output_file="timing",
        policy=select_timing_policy(TimingSourceType.HIP_NATIVE),
        tool_version="rocprofv3 7.0.0",
        gpu_architecture="gfx1200",
    )

    result = collect_rocprofv3_timing(request, runner=runner)

    assert result.profiler_collected is False
    assert result.evidence is None
    assert result.selection.fallback_applied is True
    assert "exit code 17" in result.selection.reason
    assert result.stderr == "profiler failed"


def test_live_collection_labels_missing_csv_as_fallback(tmp_path):
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=0,
            stdout="profiled",
            stderr="",
        )

    request = Rocprofv3CollectionRequest(
        application_command=("uv", "run", "sol-execbench", "problem"),
        output_directory=tmp_path,
        output_file="timing",
        policy=select_timing_policy(TimingSourceType.HIP_NATIVE),
        tool_version="rocprofv3 7.0.0",
        gpu_architecture="gfx1200",
    )

    result = collect_rocprofv3_timing(request, runner=runner)

    assert result.profiler_collected is False
    assert result.evidence is None
    assert result.selection.fallback_applied is True
    assert "did not produce a CSV" in result.selection.reason


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
        application_command=("uv", "run", "sol-execbench", "problem"),
        languages=("triton",),
        output_directory=tmp_path,
        output_file="timing",
        tool_version="rocprofv3 7.0.0",
        gpu_architecture="gfx942",
        runner=runner,
        warmup_runs=5,
        iterations=50,
        trial_count=2,
        clock_locked=False,
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
        application_command=("uv", "run", "sol-execbench", "problem"),
        languages=("hip_cpp",),
        output_directory=tmp_path,
        output_file="timing",
        tool_version="rocprofv3 7.0.0",
        gpu_architecture="gfx1200",
        runner=runner,
    )

    assert calls
    assert result.profiler_collected is True
    assert result.selection.profiler_backed is True
    assert result.selection.policy.backend == TimingBackend.ROCPROFV3
    assert result.evidence is not None
    assert result.evidence.activity_domain.value == "kernel_activity"
    assert result.evidence.backend.value == "rocprofv3"


def test_source_collection_routes_pytorch_to_explicit_fallback(tmp_path):
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"runner should not be called: {command}")

    result = collect_source_timing_evidence(
        application_command=("uv", "run", "sol-execbench", "problem"),
        languages=("pytorch",),
        output_directory=tmp_path,
        output_file="timing",
        tool_version="rocprofv3 7.0.0",
        gpu_architecture="gfx942",
        runner=runner,
        warmup_runs=5,
        iterations=50,
        trial_count=1,
        clock_locked=True,
    )

    assert result.profiler_collected is False
    assert result.evidence is None
    assert result.selection.fallback_applied is True
    assert result.selection.policy.backend == TimingBackend.DEVICE_EVENTS
