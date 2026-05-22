from __future__ import annotations

from sol_execbench.core.bench.rocm_profiler import (
    build_rocprofv3_command,
    build_timing_evidence,
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
    assert selection.policy.reason == "profiler-backed timing is unavailable"


def test_pytorch_policy_does_not_masquerade_as_rocprofv3_kernel_activity():
    policy = select_timing_policy(TimingSourceType.PYTORCH)
    selection = select_default_timing(policy, rocprofv3_available=True)

    assert selection.profiler_backed is False
    assert selection.fallback_applied is True
    assert selection.policy.backend == TimingBackend.DEVICE_EVENTS
    assert "not rocprofv3 kernel activity timing" in selection.reason
