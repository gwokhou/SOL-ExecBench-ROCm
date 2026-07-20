from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.profile_summary import (
    ProfileSummaryGovernanceGuardrail,
    ProfileSummaryIdentity,
    ProfileSummarySidecar,
    build_profile_summary_sidecar,
    evaluate_profile_summary_governance,
    profile_summary_artifact_citation_from_path,
    validate_profile_summary_freshness,
)
from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3ProfileArtifact,
    Rocprofv3ProfileResult,
    Rocprofv3ProfileStatus,
)


def _profile_result(
    tmp_path: Path,
    status: Rocprofv3ProfileStatus = Rocprofv3ProfileStatus.SUCCESS,
) -> Rocprofv3ProfileResult:
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
        artifacts=artifacts if status is Rocprofv3ProfileStatus.SUCCESS else (),
        returncode=0 if status is Rocprofv3ProfileStatus.SUCCESS else 1,
        failed_reason=(
            None if status is Rocprofv3ProfileStatus.SUCCESS else "rocprofv3 failed"
        ),
        profiler_available=True,
        artifact_coverage_status=(
            "complete" if status is Rocprofv3ProfileStatus.SUCCESS else "none"
        ),
        reason_codes=(
            ("rocprof_artifacts_registered",)
            if status is Rocprofv3ProfileStatus.SUCCESS
            else ("rocprof_command_failed",)
        ),
        timeout_seconds=60,
    )


def test_profile_summary_identity_uses_sol_version_only() -> None:
    identity = ProfileSummaryIdentity(
        generated_at="2026-01-01T00:00:00Z",
        sol_version="v1.43",
        trace_path="trace.jsonl",
        run_id="run-1",
    )

    payload = identity.model_dump(mode="json", exclude_none=True)

    assert payload["sol_version"] == "v1.43"
    assert "sol_contract_version" not in payload


def test_profile_summary_identity_rejects_sol_contract_version_alias() -> None:
    with pytest.raises(ValidationError, match="sol_contract_version"):
        ProfileSummaryIdentity.model_validate(
            {
                "generated_at": "2026-01-01T00:00:00Z",
                "sol_version": "v1.43",
                "sol_contract_version": "v1.43",
                "trace_path": "trace.jsonl",
                "run_id": "run-1",
            }
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

    assert payload["schema_version"] == "sol_execbench.profile_summary.v2"
    assert payload["status"] == "available"
    assert payload["reason_code"] == "profile_summary_generated"
    assert payload["identity"]["trace_path"] == "trace.jsonl"
    assert payload["identity"]["run_id"] == "run-0"
    assert payload["identity"]["sol_version"] == "v3.0.0"
    assert "sol_contract_version" not in payload["identity"]
    assert payload["authority"] == "diagnostic"
    assert payload["summary"]["profiler_status"] == "success"
    assert payload["summary"]["artifact_coverage_status"] == "complete"
    assert payload["summary"]["reason_codes"] == ["rocprof_artifacts_registered"]
    assert payload["summary"]["warnings"] == []
    assert payload["summary"]["artifact_count"] == 1
    assert payload["summary"]["artifact_kinds"] == {"rocpd": 1}
    assert payload["summary"]["metrics"][0]["name"] == "artifact_count"
    assert payload["artifact_citations"][0]["sha256"] == citation.sha256


def test_profile_summary_sidecar_accepts_consumer_sol_version(
    tmp_path: Path,
) -> None:
    sidecar = build_profile_summary_sidecar(
        profile_result=_profile_result(tmp_path),
        trace_path=str(tmp_path / "trace.jsonl"),
        run_id="consumer-run-001",
        sol_version="consumer-sol-version",
        generated_at="2026-06-16T00:00:00Z",
    )

    payload = sidecar.model_dump(mode="json")

    assert payload["identity"]["run_id"] == "consumer-run-001"
    assert payload["identity"]["sol_version"] == "consumer-sol-version"


def test_profile_summary_freshness_uses_canonical_sol_version(tmp_path: Path):
    sidecar = build_profile_summary_sidecar(
        profile_result=_profile_result(tmp_path),
        trace_path=str(tmp_path / "trace.jsonl"),
        run_id="run-0",
    )

    current = validate_profile_summary_freshness(
        sidecar,
        trace_path=str(tmp_path / "trace.jsonl"),
        run_id="run-0",
        sol_version="v3.0.0",
    )
    stale = validate_profile_summary_freshness(sidecar, sol_version="9.9")

    assert current.status == "current"
    assert stale.status == "stale"
    assert stale.reason_codes == ["sol_version_mismatch"]


def test_profile_summary_freshness_sol_version_only_is_current(tmp_path: Path):
    sidecar = build_profile_summary_sidecar(
        profile_result=_profile_result(tmp_path),
        trace_path=str(tmp_path / "trace.jsonl"),
        run_id="run-0",
    )

    current = validate_profile_summary_freshness(sidecar, sol_version="v3.0.0")

    assert current.status == "current"
    assert current.reason_codes == []


def test_profile_summary_sidecar_includes_structured_artifact_evidence(
    tmp_path: Path,
) -> None:
    trace_csv = tmp_path / "profile_kernel_trace.csv"
    trace_csv.write_text("Domain,Name,Duration(ns)\nKERNEL_DISPATCH,kernel,1000000\n")
    profile = Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.SUCCESS,
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

    payload = build_profile_summary_sidecar(profile_result=profile).model_dump(
        mode="json"
    )

    assert payload["summary"]["kernel_metrics"] == [
        {
            "kernel_name": "kernel",
            "name": "kernel_duration_ms",
            "value": 1.0,
            "unit": "ms",
            "source": "profile_kernel_trace.csv",
            "artifact": "profile_kernel_trace.csv",
            "parse_status": "available",
        }
    ]
    assert payload["summary"]["bottleneck_hints"] == [
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
        profile_result=_profile_result(tmp_path, status=Rocprofv3ProfileStatus.FAILED),
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


def test_profile_summary_diagnostic_log_only_profile_is_partial(tmp_path: Path):
    diagnostic = tmp_path / "profile.diagnostics.json"
    diagnostic.write_text('{"status":"no_profiler_data_artifacts"}\n')
    success_empty = Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.PARTIAL,
        command=("rocprofv3", "--kernel-trace", "--", "python", "eval_driver.py"),
        output_directory=tmp_path,
        output_file="profile",
        artifacts=(
            Rocprofv3ProfileArtifact(
                path=diagnostic,
                kind="diagnostic_json",
                size_bytes=diagnostic.stat().st_size,
            ),
        ),
        returncode=0,
        profiler_available=True,
        timeout_seconds=60,
        artifact_coverage_status="diagnostic_logs_only",
        reason_codes=(
            "rocprof_no_registered_artifacts",
            "rocprof_diagnostic_log_registered",
        ),
        warnings=(
            "rocprofv3 returned success but produced no profiler data artifacts",
        ),
    )

    sidecar = build_profile_summary_sidecar(
        profile_result=success_empty,
        trace_path="trace.jsonl",
        generated_at="2026-06-16T00:00:00Z",
    )
    payload = sidecar.model_dump(mode="json")

    assert payload["status"] == "partial"
    assert payload["reason_code"] == "profile_partial"
    assert payload["summary"]["profiler_status"] == "partial"
    assert payload["summary"]["artifact_coverage_status"] == "diagnostic_logs_only"
    assert payload["summary"]["reason_codes"] == [
        "rocprof_no_registered_artifacts",
        "rocprof_diagnostic_log_registered",
    ]
    assert payload["summary"]["artifact_count"] == 1
    assert payload["summary"]["artifact_kinds"] == {"diagnostic_json": 1}
    assert payload["summary"]["parse_warnings"] == [
        "profile.diagnostics.json: diagnostic_json artifacts are citation-only in sol_execbench.profile_summary.v2"
    ]
    assert any(
        "diagnostic logs but no profiler data artifacts" in limitation
        for limitation in payload["limitations"]
    )


def test_profile_summary_success_with_only_diagnostic_json_is_partial(
    tmp_path: Path,
):
    diagnostic = tmp_path / "profile.diagnostics.json"
    diagnostic.write_text('{"status":"no_profiler_data_artifacts"}\n')
    profile = Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.SUCCESS,
        command=("rocprofv3", "--kernel-trace", "--", "python", "eval_driver.py"),
        output_directory=tmp_path,
        output_file="profile",
        artifacts=(
            Rocprofv3ProfileArtifact(
                path=diagnostic,
                kind="diagnostic_json",
                size_bytes=diagnostic.stat().st_size,
            ),
        ),
        returncode=0,
        profiler_available=True,
        artifact_coverage_status="diagnostic_logs_only",
        reason_codes=(
            "rocprof_no_registered_artifacts",
            "rocprof_diagnostic_log_registered",
        ),
    )

    sidecar = build_profile_summary_sidecar(profile_result=profile)
    payload = sidecar.model_dump(mode="json")

    assert payload["status"] == "partial"
    assert payload["reason_code"] == "profile_partial"
    assert payload["summary"]["profiler_status"] == "success"
    assert payload["summary"]["artifact_coverage_status"] == "diagnostic_logs_only"
    assert payload["summary"]["artifact_kinds"] == {"diagnostic_json": 1}
    assert any(
        "diagnostic logs but no profiler data artifacts" in limitation
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
    payload["authority"] = "score"

    with pytest.raises(ValidationError):
        ProfileSummarySidecar.model_validate(payload)

    guardrail = evaluate_profile_summary_governance(sidecar=sidecar)
    guardrail_payload = guardrail.model_dump(mode="json")
    guardrail_payload["score_authority"] = True

    with pytest.raises(ValidationError):
        ProfileSummaryGovernanceGuardrail.model_validate(guardrail_payload)
