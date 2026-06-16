from __future__ import annotations

import pytest
from pathlib import Path
from pydantic import ValidationError

from sol_execbench.core.bench.agent_feedback import build_agent_feedback_sidecar
from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult
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
