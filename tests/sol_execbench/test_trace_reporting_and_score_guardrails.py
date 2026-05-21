from __future__ import annotations

from sol_execbench.core.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.reporting import format_trace_summary, summarize_traces
from sol_execbench.core.scoring_guardrails import (
    AMD_PERFORMANCE_CLAIM_WARNING,
    interpret_sol_score,
)
from sol_execbench.sol_score import sol_score


def _workload(uuid: str) -> Workload:
    return Workload(axes={}, inputs={}, uuid=uuid)


def _trace(status: EvaluationStatus, latency_ms: float | None = None) -> Trace:
    performance = None
    correctness = None
    if status == EvaluationStatus.PASSED:
        performance = Performance(
            latency_ms=latency_ms or 1.0,
            reference_latency_ms=2.0,
            speedup_factor=2.0,
        )
        correctness = Correctness()
    elif status == EvaluationStatus.INCORRECT_NUMERICAL:
        correctness = Correctness(max_absolute_error=1.0, max_relative_error=1.0)

    return Trace(
        definition="demo",
        workload=_workload(status.value),
        solution="solution",
        evaluation=Evaluation(
            status=status,
            environment=Environment(hardware="AMD gfx1200", libs={}),
            timestamp="2026-05-22T00:00:00+08:00",
            correctness=correctness,
            performance=performance,
        ),
    )


def test_trace_summary_does_not_mutate_public_trace_schema():
    traces = [
        _trace(EvaluationStatus.PASSED, 1.5),
        _trace(EvaluationStatus.PASSED, 2.5),
        _trace(EvaluationStatus.RUNTIME_ERROR),
    ]

    before = [trace.model_dump(mode="json") for trace in traces]
    summary = summarize_traces(traces)
    after = [trace.model_dump(mode="json") for trace in traces]

    assert before == after
    assert summary.total == 3
    assert summary.passed == 2
    assert summary.statuses == {"PASSED": 2, "RUNTIME_ERROR": 1}
    assert summary.median_latency_ms == 2.0
    assert "pass_rate=66.67%" in format_trace_summary(summary)


def test_sol_score_formula_stays_unchanged_for_existing_contract():
    assert sol_score(t_k=2.0, t_b=2.0, t_sol=1.0) == 0.5
    assert sol_score(t_k=1.0, t_b=2.0, t_sol=1.0) == 1.0


def test_amd_native_claims_receive_guardrail_warning():
    interpretation = interpret_sol_score(0.75, amd_native_claim=True)
    assert interpretation.score == 0.75
    assert interpretation.claim_level == "amd-native-performance"
    assert interpretation.warning == AMD_PERFORMANCE_CLAIM_WARNING


def test_benchmark_relative_scores_do_not_warn():
    interpretation = interpret_sol_score(0.75)
    assert interpretation.claim_level == "benchmark-relative"
    assert interpretation.warning is None
