from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackGovernanceGuardrail,
    artifact_citation_from_path,
    build_agent_feedback_sidecar,
    evaluate_agent_feedback_governance,
    validate_agent_feedback_freshness,
)
from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult
from sol_execbench.core.claim_upgrade import build_claim_upgrade_report
from sol_execbench.core.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from sol_execbench.core.data.workload import ScalarInput, Workload


def _trace(status: EvaluationStatus = EvaluationStatus.PASSED) -> Trace:
    correctness = None
    performance = None
    if status in {EvaluationStatus.PASSED, EvaluationStatus.INCORRECT_NUMERICAL}:
        correctness = Correctness(max_relative_error=0.0, max_absolute_error=0.0)
    if status == EvaluationStatus.PASSED:
        performance = Performance(
            latency_ms=1.0,
            reference_latency_ms=2.0,
            speedup_factor=2.0,
        )
    return Trace(
        definition="toy",
        solution="candidate",
        workload=Workload(
            uuid="w0",
            axes={"n": 1},
            inputs={"n": ScalarInput(value=1)},
        ),
        evaluation=Evaluation(
            status=status,
            environment=Environment(hardware="AMD gfx1200", libs={"hip": "7.0"}),
            timestamp="2026-06-16T00:00:00Z",
            correctness=correctness,
            performance=performance,
        ),
    )


def test_agent_feedback_sidecar_is_diagnostic_only_for_passing_trace():
    sidecar = build_agent_feedback_sidecar(traces=[_trace()])
    payload = sidecar.model_dump(mode="json")

    assert payload["schema_version"] == "sol_execbench.agent_feedback.v1"
    assert payload["status"] == "available"
    assert payload["reason_code"] == "feedback_generated"
    assert payload["summary"]["status_counts"] == {"PASSED": 1}
    assert payload["authority"]["diagnostic_only"] is True
    for key, value in payload["authority"].items():
        if key != "diagnostic_only":
            assert value is False
    assert payload["items"][0]["code"] == "all_evaluated_traces_passed"
    assert "Canonical Trace JSONL remains" in payload["limitations"][1]


def test_agent_feedback_sidecar_records_identity_and_artifact_citations(
    tmp_path: Path,
):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text('{"status":"ok"}\n')
    citation = artifact_citation_from_path(
        kind="trace",
        label="canonical_trace_jsonl",
        path=trace_path,
    )

    sidecar = build_agent_feedback_sidecar(
        traces=[_trace()],
        trace_path=str(trace_path),
        target_id="problem-0",
        run_id="run-0",
        candidate_hash="candidate-sha",
        source_hash="source-sha",
        generated_at="2026-06-16T00:00:00Z",
        artifact_citations=[citation],
    )
    payload = sidecar.model_dump(mode="json")

    assert payload["identity"]["generated_at"] == "2026-06-16T00:00:00Z"
    assert payload["identity"]["trace_path"] == "trace.jsonl"
    assert payload["identity"]["target_id"] == "problem-0"
    assert payload["identity"]["run_id"] == "run-0"
    assert payload["identity"]["candidate_hash"] == "candidate-sha"
    assert payload["identity"]["source_hash"] == "source-sha"
    assert payload["artifact_citations"] == [
        {
            "kind": "trace",
            "label": "canonical_trace_jsonl",
            "path": "trace.jsonl",
            "sha256": citation.sha256,
            "status": None,
        }
    ]
    assert citation.sha256 is not None
    assert len(citation.sha256) == 64


def test_agent_feedback_freshness_validation_classifies_identity(tmp_path: Path):
    trace_path = tmp_path / "trace.jsonl"
    sidecar = build_agent_feedback_sidecar(
        traces=[_trace()],
        trace_path=str(trace_path),
        target_id="problem-0",
        run_id="run-0",
    )

    current = validate_agent_feedback_freshness(
        sidecar,
        trace_path=str(trace_path),
        target_id="problem-0",
        run_id="run-0",
    )
    stale = validate_agent_feedback_freshness(
        sidecar,
        trace_path=str(tmp_path / "other.jsonl"),
        target_id="problem-0",
        run_id="run-1",
    )
    unknown = validate_agent_feedback_freshness(sidecar)

    assert current.status == "current"
    assert current.reason_codes == []
    assert stale.status == "stale"
    assert stale.reason_codes == ["trace_path_mismatch", "run_id_mismatch"]
    assert unknown.status == "unknown"
    assert unknown.reason_codes == ["insufficient_expected_identity"]


def test_agent_feedback_sidecar_summarizes_failures_and_optional_profile():
    profile = Rocprofv3ProfileResult(
        status="unavailable",
        command=("rocprofv3", "--", "python", "eval_driver.py"),
        output_directory=Path("profile"),
        output_file="profile",
        skipped_reason="rocprofv3 missing",
        profiler_available=False,
    )

    sidecar = build_agent_feedback_sidecar(
        traces=[_trace(EvaluationStatus.COMPILE_ERROR)],
        profile_result=profile,
    )
    payload = sidecar.model_dump(mode="json")

    assert payload["status"] == "partial"
    assert payload["reason_code"] == "partial_diagnostics"
    assert payload["summary"]["profile_status"] == "unavailable"
    assert payload["items"][0]["code"] == "compile_error"
    assert payload["items"][0]["bottleneck"] == "compile_failure"
    assert any(ref["kind"] == "profile" for ref in payload["source_refs"])


def test_agent_feedback_sidecar_rejects_authority_override():
    sidecar = build_agent_feedback_sidecar(traces=[_trace()])
    payload = sidecar.model_dump(mode="json")
    payload["authority"]["score_authority"] = True

    with pytest.raises(ValidationError):
        type(sidecar).model_validate(payload)


def test_agent_feedback_sidecar_rejects_unknown_bottleneck():
    sidecar = build_agent_feedback_sidecar(
        traces=[_trace(EvaluationStatus.COMPILE_ERROR)]
    )
    payload = sidecar.model_dump(mode="json")
    payload["items"][0]["bottleneck"] = "ad_hoc_bottleneck"

    with pytest.raises(ValidationError):
        type(sidecar).model_validate(payload)


def test_agent_feedback_governance_guardrail_states_remain_diagnostic_only(
    tmp_path: Path,
):
    sidecar = build_agent_feedback_sidecar(
        traces=[_trace()],
        trace_path=str(tmp_path / "trace.jsonl"),
    )
    stale = validate_agent_feedback_freshness(
        sidecar,
        trace_path=str(tmp_path / "other.jsonl"),
    )

    guardrails = [
        evaluate_agent_feedback_governance(sidecar=sidecar),
        evaluate_agent_feedback_governance(sidecar=sidecar, freshness=stale),
        evaluate_agent_feedback_governance(sidecar=None),
        evaluate_agent_feedback_governance(
            sidecar=None,
            parse_error="invalid json",
        ),
    ]

    assert [guardrail.status for guardrail in guardrails] == [
        "usable_diagnostic",
        "stale_diagnostic",
        "unavailable",
        "invalid_diagnostic",
    ]
    for guardrail in guardrails:
        payload = guardrail.model_dump(mode="json")
        assert payload["diagnostic_only"] is True
        for key, value in payload.items():
            if key.endswith("_authority"):
                assert value is False


def test_agent_feedback_governance_rejects_authority_override():
    guardrail = evaluate_agent_feedback_governance(
        sidecar=build_agent_feedback_sidecar(traces=[_trace()])
    )
    payload = guardrail.model_dump(mode="json")
    payload["claim_upgrade_authority"] = True

    with pytest.raises(ValidationError):
        AgentFeedbackGovernanceGuardrail.model_validate(payload)


def test_agent_feedback_sidecar_states_do_not_promote_claim_upgrade():
    sidecar = build_agent_feedback_sidecar(traces=[_trace()])
    stale = validate_agent_feedback_freshness(sidecar, trace_path="other.jsonl")

    for guardrail in (
        evaluate_agent_feedback_governance(sidecar=sidecar),
        evaluate_agent_feedback_governance(sidecar=sidecar, freshness=stale),
        evaluate_agent_feedback_governance(sidecar=None),
        evaluate_agent_feedback_governance(sidecar=None, parse_error="invalid json"),
    ):
        report = build_claim_upgrade_report(created_at="2026-06-16T00:00:00Z")

        assert report.highest_eligible_claim == "diagnostic_only"
        assert report.claim_boundary.score_authority is False
        assert report.claim_boundary.leaderboard_authority is False
        assert guardrail.score_authority is False
        assert guardrail.evidence_tier_authority is False
        assert guardrail.release_gate_authority is False
        assert guardrail.cutover_authority is False
