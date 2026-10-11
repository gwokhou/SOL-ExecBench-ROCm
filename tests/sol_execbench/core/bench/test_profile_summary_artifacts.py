from __future__ import annotations

from pathlib import Path

import pytest

from sol_execbench.core.bench.profile_summary.artifacts import (
    structured_profile_evidence,
)
from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3ArtifactCoverageStatus,
    Rocprofv3ArtifactKind,
    Rocprofv3ProfileArtifact,
    Rocprofv3ProfileResult,
    Rocprofv3ProfileStatus,
    Rocprofv3ReasonCode,
    classify_profile_artifact,
    parse_rocprofv3_csv,
    summarize_rocprofv3_csv,
)

_ARTIFACT_KIND_CASES = (
    ("profile.rocpd", Rocprofv3ArtifactKind.ROCPD),
    ("profile_agent_info.csv", Rocprofv3ArtifactKind.AGENT_INFO_CSV),
    ("profile_counters.csv", Rocprofv3ArtifactKind.COUNTER_CSV),
    ("profile_kernel_trace.csv", Rocprofv3ArtifactKind.TRACE_CSV),
    ("profile_diagnostic.json", Rocprofv3ArtifactKind.DIAGNOSTIC_JSON),
    ("profile_metadata.json", Rocprofv3ArtifactKind.METADATA_JSON),
    ("profile.pftrace", Rocprofv3ArtifactKind.PERFETTO_TRACE),
    ("profile.otf2", Rocprofv3ArtifactKind.OTF2_TRACE),
    ("profile.bin", Rocprofv3ArtifactKind.OTHER),
)


@pytest.mark.parametrize(("filename", "expected"), _ARTIFACT_KIND_CASES)
def test_profile_artifact_classifier_covers_kind(
    filename: str,
    expected: Rocprofv3ArtifactKind,
) -> None:
    assert classify_profile_artifact(Path(filename)) is expected


def test_profile_artifact_classifier_cases_cover_every_kind() -> None:
    assert {expected for _, expected in _ARTIFACT_KIND_CASES} == set(
        Rocprofv3ArtifactKind,
    )


def _profile_result(
    tmp_path: Path,
    artifacts: tuple[Rocprofv3ProfileArtifact, ...],
) -> Rocprofv3ProfileResult:
    return Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.SUCCESS,
        command=(
            "rocprofv3",
            "--kernel-trace",
            "--",
            "python",
            "eval_driver.py",
        ),
        output_directory=tmp_path,
        output_file="profile",
        artifacts=artifacts,
        returncode=0,
        profiler_available=True,
        artifact_coverage_status=Rocprofv3ArtifactCoverageStatus.COMPLETE,
        reason_codes=(Rocprofv3ReasonCode.ARTIFACTS_REGISTERED,),
    )


@pytest.mark.parametrize(
    ("start", "end", "expected_ns"),
    [
        pytest.param("1000", "4000", 3000, id="timestamp-duration"),
        pytest.param("1000", "1000", 0, id="zero-duration"),
        pytest.param("4000", "1000", None, id="reversed-timestamps"),
        pytest.param("1000", "", None, id="missing-end"),
        pytest.param("invalid", "4000", None, id="invalid-start"),
        pytest.param("nan", "4000", None, id="nan-start"),
        pytest.param("1000", "inf", None, id="infinite-end"),
        pytest.param("-1e308", "1e308", None, id="duration-overflow"),
    ],
)
def test_timestamp_csv_has_consistent_summary_and_timing_metrics(
    tmp_path: Path, start: str, end: str, expected_ns: int | None
) -> None:
    """ROCm 10.1 kernel CSVs omit the standalone duration column."""
    content = (
        '"Kind","Kernel_Name","Start_Timestamp","End_Timestamp"\n'
        f'"KERNEL_DISPATCH","_vecadd","{start}","{end}"\n'
        '"HIP_RUNTIME_API","hipLaunchKernel","1000","5000"\n'
    )
    path = tmp_path / "profile_kernel_trace.csv"
    path.write_text(content, encoding="utf-8")
    profile = _profile_result(
        tmp_path,
        (
            Rocprofv3ProfileArtifact(
                path=path,
                kind=Rocprofv3ArtifactKind.TRACE_CSV,
                size_bytes=path.stat().st_size,
            ),
        ),
    )

    evidence = structured_profile_evidence(profile)
    timing_rows = [
        row for row in parse_rocprofv3_csv(content) if row.is_kernel_activity
    ]
    count, duration_ns = summarize_rocprofv3_csv(path)

    assert evidence.parse_warnings == []
    assert count == len(timing_rows) == len(evidence.kernel_metrics)
    if expected_ns is None:
        assert count == 0
        assert duration_ns == 0
    else:
        assert count == 1
        assert duration_ns == expected_ns == timing_rows[0].duration_ns
        metric = evidence.kernel_metrics[0]
        assert metric.kernel_name == "_vecadd"
        assert metric.unit == "ms"
        assert metric.value == pytest.approx(expected_ns / 1_000_000)


def test_structured_profile_evidence_parses_trace_counter_and_metadata(
    tmp_path: Path,
) -> None:
    trace_csv = tmp_path / "profile_kernel_trace.csv"
    trace_csv.write_text(
        "Domain,Name,Duration(ns)\n"
        "KERNEL_DISPATCH,matmul_kernel,2000000\n"
        "HIP_RUNTIME_API,hipLaunchKernel,50000\n",
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
                kind=Rocprofv3ArtifactKind.TRACE_CSV,
                size_bytes=trace_csv.stat().st_size,
            ),
            Rocprofv3ProfileArtifact(
                path=counter_csv,
                kind=Rocprofv3ArtifactKind.COUNTER_CSV,
                size_bytes=counter_csv.stat().st_size,
            ),
            Rocprofv3ProfileArtifact(
                path=metadata_json,
                kind=Rocprofv3ArtifactKind.METADATA_JSON,
                size_bytes=metadata_json.stat().st_size,
            ),
            Rocprofv3ProfileArtifact(
                path=rocpd,
                kind=Rocprofv3ArtifactKind.ROCPD,
                size_bytes=rocpd.stat().st_size,
            ),
        ),
    )

    evidence = structured_profile_evidence(profile)

    assert [
        metric.model_dump(mode="json") for metric in evidence.workload_metrics
    ] == [
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
        (
            "matmul_kernel",
            "kernel_duration_ms",
            2.0,
            "ms",
            "profile_kernel_trace.csv",
        ),
        (
            "profile_counters",
            "SQ_INSTS_VALU",
            120000,
            "count",
            "profile_counters.csv",
        ),
    ]
    assert [hint.category for hint in evidence.bottleneck_hints] == [
        "compute_bound",
    ]
    assert evidence.parse_warnings == [
        "profile.rocpd: hip_event_artifact_invalid_or_unsupported",
    ]


def test_structured_profile_evidence_rejects_bool_and_nan_dispatch_count(
    tmp_path: Path,
) -> None:
    metadata_json = tmp_path / "profile.json"
    metadata_json.write_text(
        '{"kernel_dispatches": true, "dispatch_count": "nan"}\n',
    )
    profile = _profile_result(
        tmp_path,
        (
            Rocprofv3ProfileArtifact(
                path=metadata_json,
                kind=Rocprofv3ArtifactKind.METADATA_JSON,
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
                kind=Rocprofv3ArtifactKind.TRACE_CSV,
                size_bytes=0,
            ),
            Rocprofv3ProfileArtifact(
                path=missing_counter_csv,
                kind=Rocprofv3ArtifactKind.COUNTER_CSV,
                size_bytes=0,
            ),
            Rocprofv3ProfileArtifact(
                path=missing_metadata_json,
                kind=Rocprofv3ArtifactKind.METADATA_JSON,
                size_bytes=0,
            ),
        ),
    )

    evidence = structured_profile_evidence(profile)

    assert evidence.kernel_metrics == []
    assert [
        metric.model_dump(mode="json") for metric in evidence.workload_metrics
    ] == [
        {
            "name": "artifact_coverage_status",
            "value": "complete",
            "unit": None,
            "source": "rocprofv3_profile_metadata",
            "workload_id": None,
            "artifact": None,
            "parse_status": "available",
        },
    ]
    assert evidence.parse_warnings == [
        "missing_kernel_trace.csv: profiler artifact is missing",
        "missing_counters.csv: profiler artifact is missing",
        "missing_profile.json: hip_event_artifact_invalid_or_unsupported",
        "missing_profile.json: profiler artifact is missing",
    ]
