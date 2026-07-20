from __future__ import annotations

from sol_execbench.core.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
)
from sol_execbench.core.platform.diagnostics import (
    DiagnosticStage,
    StageDiagnostic,
)
from sol_execbench.core.reports.reporting import (
    build_evidence_report,
    format_trace_summary,
    summarize_traces,
)
from sol_execbench_type_helpers import make_trace, make_workload


def _evaluation(
    status: EvaluationStatus, latency_ms: float | None = None
) -> Evaluation:
    passed = status is EvaluationStatus.PASSED
    return Evaluation(
        status=status,
        environment=Environment(hardware="test GPU"),
        timestamp="2026-07-19T00:00:00Z",
        correctness=Correctness() if passed else None,
        performance=(
            Performance(latency_ms=latency_ms or 0.0, trials=1) if passed else None
        ),
    )


def _trace(name: str, evaluation: Evaluation | None):
    return make_trace(
        definition="demo",
        solution="candidate" if evaluation else None,
        workload=make_workload(uuid=name, axes={}, inputs={}),
        evaluation=evaluation,
    )


def test_summarize_traces_counts_statuses_and_latency() -> None:
    summary = summarize_traces(
        [
            _trace("passed-a", _evaluation(EvaluationStatus.PASSED, 1.0)),
            _trace("passed-b", _evaluation(EvaluationStatus.PASSED, 3.0)),
            _trace("failed", _evaluation(EvaluationStatus.RUNTIME_ERROR)),
            _trace("pending", None),
        ]
    )

    assert summary.total == 4
    assert summary.passed == 2
    assert summary.pass_rate == 0.5
    assert summary.statuses == {"PASSED": 2, "RUNTIME_ERROR": 1, "NO_EVALUATION": 1}
    assert summary.median_latency_ms == 2.0
    assert summary.mean_latency_ms == 2.0
    assert "median_latency_ms=2.000" in format_trace_summary(summary)


def test_empty_summary_has_zero_pass_rate_and_no_latency() -> None:
    summary = summarize_traces([])

    assert summary.pass_rate == 0.0
    assert summary.median_latency_ms is None
    assert "statuses=[]" in format_trace_summary(summary)


def test_evidence_report_serializes_diagnostic_without_changing_traces() -> None:
    trace = _trace("passed", _evaluation(EvaluationStatus.PASSED, 1.5))
    before = trace.model_dump(mode="json")
    diagnostic = StageDiagnostic(
        stage=DiagnosticStage.RUNTIME,
        status="warning",
        message="clock lock unavailable",
        hint="run diagnostic timing only",
    )

    payload = build_evidence_report([trace], [diagnostic]).to_dict()

    assert payload["derived"] is True
    assert payload["summary"]["passed"] == 1
    assert payload["diagnostics"] == [
        {
            "stage": "runtime",
            "status": "warning",
            "message": "clock lock unavailable",
            "hint": "run diagnostic timing only",
        }
    ]
    assert trace.model_dump(mode="json") == before
