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
