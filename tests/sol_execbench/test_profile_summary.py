from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.profile_summary import (
    ProfileSummaryGovernanceGuardrail,
    ProfileSummarySidecar,
    build_profile_summary_sidecar,
    evaluate_profile_summary_governance,
    profile_summary_artifact_citation_from_path,
    validate_profile_summary_freshness,
)
from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3ProfileArtifact,
    Rocprofv3ProfileResult,
)
from sol_execbench.core.claim_upgrade import build_claim_upgrade_report


def _profile_result(tmp_path: Path, status: str = "success") -> Rocprofv3ProfileResult:
    artifact = tmp_path / "profile.rocpd"
    artifact.write_text("profile artifact\n")
    artifacts = (
        Rocprofv3ProfileArtifact(
            path=artifact,
            kind="rocpd",
            size_bytes=artifact.stat().st_size,
        ),
    )
    return Rocprofv3ProfileResult(
        status=status,
        command=("rocprofv3", "--kernel-trace", "--", "python", "eval_driver.py"),
        output_directory=tmp_path,
        output_file="profile",
        artifacts=artifacts if status == "success" else (),
        returncode=0 if status == "success" else 1,
        failed_reason=None if status == "success" else "rocprofv3 failed",
        profiler_available=True,
        artifact_coverage_status="complete" if status == "success" else "none",
        reason_codes=(
            ("rocprof_artifacts_registered",)
            if status == "success"
            else ("rocprof_command_failed",)
        ),
        timeout_seconds=60,
    )


def test_profile_summary_sidecar_is_diagnostic_only(tmp_path: Path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text('{"definition":"toy"}\n')
    profile = _profile_result(tmp_path)
    citation = profile_summary_artifact_citation_from_path(
        kind="trace",
        label="canonical_trace_jsonl",
        path=trace_path,
    )

    sidecar = build_profile_summary_sidecar(
        profile_result=profile,
        trace_path=str(trace_path),
        run_id="run-0",
        generated_at="2026-06-16T00:00:00Z",
        artifact_citations=[citation],
    )
    payload = sidecar.model_dump(mode="json")

    assert payload["schema_version"] == "sol_execbench.profile_summary.v1"
    assert payload["status"] == "available"
    assert payload["reason_code"] == "profile_summary_generated"
    assert payload["identity"]["trace_path"] == "trace.jsonl"
    assert payload["identity"]["run_id"] == "run-0"
    assert payload["authority"]["diagnostic_only"] is True
    for key, value in payload["authority"].items():
        if key != "diagnostic_only":
            assert value is False
    assert payload["summary"]["profiler_status"] == "success"
    assert payload["summary"]["artifact_coverage_status"] == "complete"
    assert payload["summary"]["reason_codes"] == ["rocprof_artifacts_registered"]
    assert payload["summary"]["warnings"] == []
    assert payload["summary"]["artifact_count"] == 1
    assert payload["summary"]["artifact_kinds"] == {"rocpd": 1}
    assert payload["summary"]["metrics"][0]["name"] == "artifact_count"
    assert payload["artifact_citations"][0]["sha256"] == citation.sha256


def test_profile_summary_builds_structured_metrics_and_hints_from_text_artifacts(
    tmp_path: Path,
):
    trace_csv = tmp_path / "profile_kernel_trace.csv"
    trace_csv.write_text(
        "Domain,Name,Duration(ns)\n"
        "KERNEL_DISPATCH,matmul_kernel,2000000\n"
        "KERNEL_DISPATCH,matmul_epilogue,1000000\n"
        "HIP_RUNTIME_API,hipLaunchKernel,50000\n"
    )
    counter_csv = tmp_path / "profile_counters.csv"
    counter_csv.write_text(
        "Metric,Value,Unit\n"
        "SQ_INSTS_VALU,120000,count\n"
        "L2_CACHE_HIT_RATE,42,percent\n"
        "LDS_BANK_CONFLICT,8,count\n"
    )
    metadata_json = tmp_path / "profile.json"
    metadata_json.write_text(
        '{"workload_id":"w0","kernel_dispatches":2,"device":"gfx1200"}\n'
    )
    rocpd = tmp_path / "profile.rocpd"
    rocpd.write_text("opaque database")
    profile = Rocprofv3ProfileResult(
        status="success",
        command=("rocprofv3", "--kernel-trace", "--", "python", "eval_driver.py"),
        output_directory=tmp_path,
        output_file="profile",
        artifacts=(
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
        returncode=0,
        profiler_available=True,
        artifact_coverage_status="complete",
        reason_codes=("rocprof_artifacts_registered",),
    )

    sidecar = build_profile_summary_sidecar(
        profile_result=profile,
        trace_path=str(tmp_path / "trace.jsonl"),
        run_id="run-0",
        generated_at="2026-06-21T00:00:00Z",
    )
    payload = sidecar.model_dump(mode="json")
    summary = payload["summary"]

    assert summary["workload_metrics"] == [
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
    assert {
        (metric["kernel_name"], metric["name"], metric["value"], metric["unit"])
        for metric in summary["kernel_metrics"]
    } == {
        ("matmul_kernel", "kernel_duration_ms", 2.0, "ms"),
        ("matmul_epilogue", "kernel_duration_ms", 1.0, "ms"),
        ("profile_counters", "SQ_INSTS_VALU", 120000, "count"),
        ("profile_counters", "L2_CACHE_HIT_RATE", 42, "percent"),
        ("profile_counters", "LDS_BANK_CONFLICT", 8, "count"),
    }
    assert summary["bottleneck_hints"] == [
        {
            "category": "memory_l2_bound",
            "severity": "medium",
            "confidence": "low",
            "message": "Low L2 hit-rate counter suggests memory/L2 pressure.",
            "source_metrics": ["L2_CACHE_HIT_RATE"],
            "evidence_artifacts": ["profile_counters.csv"],
        },
        {
            "category": "lds_bound",
            "severity": "low",
            "confidence": "low",
            "message": "LDS bank conflict counter is present and non-zero.",
            "source_metrics": ["LDS_BANK_CONFLICT"],
            "evidence_artifacts": ["profile_counters.csv"],
        },
        {
            "category": "compute_bound",
            "severity": "low",
            "confidence": "low",
            "message": "VALU instruction counter is present without stronger memory or launch evidence.",
            "source_metrics": ["SQ_INSTS_VALU"],
            "evidence_artifacts": ["profile_counters.csv"],
        },
    ]
    assert summary["parse_warnings"] == [
        "profile.rocpd: rocpd artifacts are citation-only in profile_summary.sidecar.v1"
    ]
    assert all(
        "rocpd" not in metric.get("artifact", "")
        for metric in [*summary["workload_metrics"], *summary["kernel_metrics"]]
    )
    assert payload["authority"]["diagnostic_only"] is True
    assert payload["authority"]["score_authority"] is False
    assert payload["authority"]["cutover_authority"] is False


def test_profile_summary_reports_insufficient_counters_without_speculation(
    tmp_path: Path,
):
    trace_csv = tmp_path / "profile_kernel_trace.csv"
    trace_csv.write_text("Domain,Name,Duration(ns)\nKERNEL_DISPATCH,kernel,100000\n")
    profile = Rocprofv3ProfileResult(
        status="success",
        command=("rocprofv3", "--kernel-trace", "--", "python", "eval_driver.py"),
        output_directory=tmp_path,
        output_file="profile",
        artifacts=(
            Rocprofv3ProfileArtifact(
                path=trace_csv,
                kind="trace_csv",
                size_bytes=trace_csv.stat().st_size,
            ),
        ),
        returncode=0,
        profiler_available=True,
        artifact_coverage_status="complete",
        reason_codes=("rocprof_artifacts_registered",),
    )

    sidecar = build_profile_summary_sidecar(profile_result=profile)
    hints = sidecar.model_dump(mode="json")["summary"]["bottleneck_hints"]

    assert hints == [
        {
            "category": "insufficient_counters",
            "severity": "low",
            "confidence": "high",
            "message": "No bounded counter artifact was available for bottleneck classification.",
            "source_metrics": [],
            "evidence_artifacts": [],
        }
    ]


def test_profile_summary_sidecar_handles_unavailable_inputs(tmp_path: Path):
    unavailable = build_profile_summary_sidecar(
        profile_result=None,
        trace_path="trace.jsonl",
        generated_at="2026-06-16T00:00:00Z",
    )
    failed = build_profile_summary_sidecar(
        profile_result=_profile_result(tmp_path, status="failed"),
        trace_path="trace.jsonl",
        generated_at="2026-06-16T00:00:00Z",
    )

    assert unavailable.status == "unavailable"
    assert unavailable.reason_code == "no_profile_result"
    assert unavailable.summary.artifact_count == 0
    assert "No rocprofv3 profile result" in unavailable.limitations[-1]
    assert failed.status == "partial"
    assert failed.reason_code == "profile_partial"
    assert failed.summary.profiler_status == "failed"
    assert failed.summary.failed_reason == "rocprofv3 failed"
    assert failed.summary.reason_codes == ["rocprof_command_failed"]


def test_profile_summary_success_without_artifacts_is_partial(tmp_path: Path):
    success_empty = Rocprofv3ProfileResult(
        status="success",
        command=("rocprofv3", "--kernel-trace", "--", "python", "eval_driver.py"),
        output_directory=tmp_path,
        output_file="profile",
        artifacts=(),
        returncode=0,
        profiler_available=True,
        timeout_seconds=60,
        artifact_coverage_status="none",
        reason_codes=("rocprof_no_registered_artifacts",),
    )

    sidecar = build_profile_summary_sidecar(
        profile_result=success_empty,
        trace_path="trace.jsonl",
        generated_at="2026-06-16T00:00:00Z",
    )
    payload = sidecar.model_dump(mode="json")

    assert payload["status"] == "partial"
    assert payload["reason_code"] == "profile_partial"
    assert payload["summary"]["profiler_status"] == "success"
    assert payload["summary"]["artifact_coverage_status"] == "none"
    assert payload["summary"]["reason_codes"] == ["rocprof_no_registered_artifacts"]
    assert payload["summary"]["artifact_count"] == 0
    assert any(
        "without registered artifacts" in limitation
        for limitation in payload["limitations"]
    )


def test_profile_summary_freshness_and_governance(tmp_path: Path):
    sidecar = build_profile_summary_sidecar(
        profile_result=_profile_result(tmp_path),
        trace_path=str(tmp_path / "trace.jsonl"),
        run_id="run-0",
    )

    current = validate_profile_summary_freshness(
        sidecar,
        trace_path=str(tmp_path / "trace.jsonl"),
        run_id="run-0",
    )
    stale = validate_profile_summary_freshness(
        sidecar,
        trace_path=str(tmp_path / "other.jsonl"),
        run_id="run-1",
    )
    unknown = validate_profile_summary_freshness(sidecar)

    assert current.status == "current"
    assert stale.status == "stale"
    assert stale.reason_codes == ["trace_path_mismatch", "run_id_mismatch"]
    assert unknown.status == "unknown"
    assert (
        evaluate_profile_summary_governance(
            sidecar=sidecar,
            freshness=current,
        ).status
        == "usable_diagnostic"
    )
    assert (
        evaluate_profile_summary_governance(
            sidecar=sidecar,
            freshness=stale,
        ).status
        == "stale_diagnostic"
    )
    assert evaluate_profile_summary_governance(sidecar=None).status == "unavailable"
    assert (
        evaluate_profile_summary_governance(
            sidecar=None,
            parse_error="invalid json",
        ).status
        == "invalid_diagnostic"
    )


def test_profile_summary_rejects_authority_override(tmp_path: Path):
    sidecar = build_profile_summary_sidecar(profile_result=_profile_result(tmp_path))
    payload = sidecar.model_dump(mode="json")
    payload["authority"]["timing_authority"] = True

    with pytest.raises(ValidationError):
        ProfileSummarySidecar.model_validate(payload)

    guardrail = evaluate_profile_summary_governance(sidecar=sidecar)
    guardrail_payload = guardrail.model_dump(mode="json")
    guardrail_payload["score_authority"] = True

    with pytest.raises(ValidationError):
        ProfileSummaryGovernanceGuardrail.model_validate(guardrail_payload)


def test_profile_summary_does_not_promote_claim_upgrade(tmp_path: Path):
    sidecar = build_profile_summary_sidecar(profile_result=_profile_result(tmp_path))
    guardrail = evaluate_profile_summary_governance(sidecar=sidecar)
    report = build_claim_upgrade_report(created_at="2026-06-16T00:00:00Z")

    assert report.highest_eligible_claim == "diagnostic_only"
    assert report.claim_boundary.score_authority is False
    assert report.claim_boundary.leaderboard_authority is False
    assert guardrail.score_authority is False
    assert guardrail.timing_authority is False
    assert guardrail.release_gate_authority is False
