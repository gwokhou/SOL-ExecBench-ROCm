from __future__ import annotations

from pathlib import Path

from sol_execbench.core.bench.profile_summary.artifacts import (
    structured_profile_evidence,
)
from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3ProfileArtifact,
    Rocprofv3ProfileResult,
    Rocprofv3ProfileStatus,
)


def _profile_result(
    tmp_path: Path,
    artifacts: tuple[Rocprofv3ProfileArtifact, ...],
) -> Rocprofv3ProfileResult:
    return Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.SUCCESS,
        command=("rocprofv3", "--kernel-trace", "--", "python", "eval_driver.py"),
        output_directory=tmp_path,
        output_file="profile",
        artifacts=artifacts,
        returncode=0,
        profiler_available=True,
        artifact_coverage_status="complete",
        reason_codes=("rocprof_artifacts_registered",),
    )


def test_structured_profile_evidence_parses_trace_counter_and_metadata(
    tmp_path: Path,
) -> None:
    trace_csv = tmp_path / "profile_kernel_trace.csv"
    trace_csv.write_text(
        "Domain,Name,Duration(ns)\n"
        "KERNEL_DISPATCH,matmul_kernel,2000000\n"
        "HIP_RUNTIME_API,hipLaunchKernel,50000\n"
    )
    counter_csv = tmp_path / "profile_counters.csv"
    counter_csv.write_text("Metric,Value,Unit\nSQ_INSTS_VALU,120000,count\n")
    metadata_json = tmp_path / "profile.json"
    metadata_json.write_text('{"workload_id":"w0","kernel_dispatches":2}\n')
    rocpd = tmp_path / "profile.rocpd"
    rocpd.write_text("opaque database")
    profile = _profile_result(
        tmp_path,
        (
            Rocprofv3ProfileArtifact(
                path=trace_csv,
                kind="trace_csv",
                size_bytes=trace_csv.stat().st_size,
            ),
            Rocprofv3ProfileArtifact(
                path=counter_csv,
                kind="counter_csv",
                size_bytes=counter_csv.stat().st_size,
            ),
            Rocprofv3ProfileArtifact(
                path=metadata_json,
                kind="metadata_json",
                size_bytes=metadata_json.stat().st_size,
            ),
            Rocprofv3ProfileArtifact(
                path=rocpd,
                kind="rocpd",
                size_bytes=rocpd.stat().st_size,
            ),
        ),
    )

    evidence = structured_profile_evidence(profile)

    assert [metric.model_dump(mode="json") for metric in evidence.workload_metrics] == [
        {
            "name": "artifact_coverage_status",
            "value": "complete",
            "unit": None,
            "source": "rocprofv3_profile_metadata",
            "workload_id": "w0",
            "artifact": None,
            "parse_status": "available",
        },
        {
            "name": "kernel_dispatch_count",
            "value": 2,
            "unit": "count",
            "source": "profile.json",
            "workload_id": "w0",
            "artifact": "profile.json",
            "parse_status": "available",
        },
    ]
    assert [
        (
            metric.kernel_name,
            metric.name,
            metric.value,
            metric.unit,
            metric.artifact,
        )
        for metric in evidence.kernel_metrics
    ] == [
        ("matmul_kernel", "kernel_duration_ms", 2.0, "ms", "profile_kernel_trace.csv"),
        ("profile_counters", "SQ_INSTS_VALU", 120000, "count", "profile_counters.csv"),
    ]
    assert [hint.category for hint in evidence.bottleneck_hints] == ["compute_bound"]
    assert evidence.parse_warnings == [
        "profile.rocpd: rocpd artifacts are citation-only in sol_execbench.profile_summary.v2"
    ]


def test_structured_profile_evidence_rejects_bool_and_nan_dispatch_count(
    tmp_path: Path,
) -> None:
    metadata_json = tmp_path / "profile.json"
    metadata_json.write_text('{"kernel_dispatches": true, "dispatch_count": "nan"}\n')
    profile = _profile_result(
        tmp_path,
        (
            Rocprofv3ProfileArtifact(
                path=metadata_json,
                kind="metadata_json",
                size_bytes=metadata_json.stat().st_size,
            ),
        ),
    )

    evidence = structured_profile_evidence(profile)

    assert [
        metric
        for metric in evidence.workload_metrics
        if metric.name == "kernel_dispatch_count"
    ] == []


def test_structured_profile_evidence_reports_missing_artifacts(
    tmp_path: Path,
) -> None:
    missing_trace_csv = tmp_path / "missing_kernel_trace.csv"
    missing_counter_csv = tmp_path / "missing_counters.csv"
    missing_metadata_json = tmp_path / "missing_profile.json"
    profile = _profile_result(
        tmp_path,
        (
            Rocprofv3ProfileArtifact(
                path=missing_trace_csv,
                kind="trace_csv",
                size_bytes=0,
            ),
            Rocprofv3ProfileArtifact(
                path=missing_counter_csv,
                kind="counter_csv",
                size_bytes=0,
            ),
            Rocprofv3ProfileArtifact(
                path=missing_metadata_json,
                kind="metadata_json",
                size_bytes=0,
            ),
        ),
    )

    evidence = structured_profile_evidence(profile)

    assert evidence.kernel_metrics == []
    assert [metric.model_dump(mode="json") for metric in evidence.workload_metrics] == [
        {
            "name": "artifact_coverage_status",
            "value": "complete",
            "unit": None,
            "source": "rocprofv3_profile_metadata",
            "workload_id": None,
            "artifact": None,
            "parse_status": "available",
        }
    ]
    assert evidence.parse_warnings == [
        "missing_kernel_trace.csv: profiler artifact is missing",
        "missing_counters.csv: profiler artifact is missing",
        "missing_profile.json: profiler artifact is missing",
    ]
