from __future__ import annotations

from sol_execbench.core.bench.static_kernel.evidence import (
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceStatus,
    build_static_kernel_evidence_sidecar,
)
from sol_execbench.core.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.platform.diagnostics import DiagnosticStage, StageDiagnostic
from sol_execbench.core.reports.reporting import (
    CANONICAL_BENCHMARK_OUTPUT,
    DERIVED_EVIDENCE_SCHEMA_VERSION,
    build_evidence_report,
    format_trace_summary,
    summarize_traces,
)
from sol_execbench.core.evidence.scoring_guardrails import (
    AMD_PERFORMANCE_CLAIM_WARNING,
    interpret_sol_score,
)
from sol_execbench.core.scoring.amd_score.sidecar_parsing import (
    amd_sol_bound_from_payload,
    minimal_solar_aggregate_from_payload,
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


def test_derived_evidence_report_labels_noncanonical_output():
    traces = [_trace(EvaluationStatus.PASSED, 1.5)]
    diagnostics = [
        StageDiagnostic(
            stage=DiagnosticStage.ENVIRONMENT,
            status="available",
            message="hipcc found",
        )
    ]

    before = [trace.model_dump(mode="json") for trace in traces]
    report = build_evidence_report(traces, diagnostics)
    after = [trace.model_dump(mode="json") for trace in traces]
    payload = report.to_dict()

    assert before == after
    assert report.schema_version == DERIVED_EVIDENCE_SCHEMA_VERSION
    assert report.derived is True
    assert report.canonical_output == CANONICAL_BENCHMARK_OUTPUT
    assert payload["canonical_output"] == "trace_jsonl"
    assert payload["summary"]["total"] == 1
    assert payload["diagnostics"] == [
        {
            "stage": "environment",
            "status": "available",
            "message": "hipcc found",
            "hint": None,
        }
    ]


def test_sol_score_formula_orders_slow_equal_fast_candidates():
    slow = sol_score(t_k=3.0, t_b=2.0, t_sol=1.0)
    equal = sol_score(t_k=2.0, t_b=2.0, t_sol=1.0)
    fast = sol_score(t_k=1.5, t_b=2.0, t_sol=1.0)
    at_sol = sol_score(t_k=1.0, t_b=2.0, t_sol=1.0)

    assert slow == 1.0 / 3.0
    assert equal == 0.5
    assert fast == 2.0 / 3.0
    assert at_sol == 1.0
    assert slow < equal < fast < at_sol


def test_amd_native_claims_receive_guardrail_warning():
    interpretation = interpret_sol_score(0.75, amd_native_claim=True)
    assert interpretation.score == 0.75
    assert interpretation.claim_level == "amd-native-performance"
    assert interpretation.warning == AMD_PERFORMANCE_CLAIM_WARNING


def test_benchmark_relative_scores_do_not_warn():
    interpretation = interpret_sol_score(0.75)
    assert interpretation.claim_level == "benchmark-relative"
    assert interpretation.warning is None


def test_static_evidence_sidecar_construction_does_not_mutate_trace_or_scoring():
    trace = _trace(EvaluationStatus.PASSED, 1.5)
    before = trace.model_dump(mode="json")

    sidecar = build_static_kernel_evidence_sidecar(
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
    )
    summary = summarize_traces([trace])
    score = sol_score(t_k=1.0, t_b=2.0, t_sol=1.0)
    interpretation = interpret_sol_score(score)
    after = trace.model_dump(mode="json")

    assert sidecar.diagnostic_only is True
    assert sidecar.score_authority is False
    assert before == after
    assert "static_kernel_evidence" not in before
    assert summary.total == 1
    assert score == 1.0
    assert interpretation.claim_level == "benchmark-relative"


def test_amd_sol_bound_rejects_malformed_nested_payloads() -> None:
    assert amd_sol_bound_from_payload({"schema_version": "wrong"}) is None
    assert (
        amd_sol_bound_from_payload(
            {
                "schema_version": "sol_execbench.amd_sol_bound.v3",
                "aggregate_bound": [],
                "hardware_model": {},
                "coverage_summary": {},
            }
        )
        is None
    )


def test_minimal_solar_aggregate_rejects_non_dict_aggregate_status() -> None:
    assert minimal_solar_aggregate_from_payload({"aggregate_status": []}) is None
