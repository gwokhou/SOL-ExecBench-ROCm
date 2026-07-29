from __future__ import annotations

from sol_execbench.core.bench.performance_model.acceptance import (
    DiagnosticAcceptanceCase,
    evaluate_diagnostic_acceptance,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind


def _case(
    uuid: str,
    kind: WorkloadKind,
    attribution: str,
    measured: float,
    predicted: float,
) -> DiagnosticAcceptanceCase:
    return DiagnosticAcceptanceCase(
        workload_uuid=uuid,
        workload_kind=kind,
        candidate_sha256=uuid * 64,
        predicted_ms=predicted,
        measured_ms=measured,
        primary_attribution_code=attribution,
    )


def test_frozen_acceptance_requires_accuracy_coverage_and_attribution() -> None:
    records = [
        ("a", WorkloadKind.ELEMENTWISE, "launch_bound", 1.0, 1.05),
        ("b", WorkloadKind.TRANSPOSE, "memory_access_efficiency", 2.0, 2.1),
        ("c", WorkloadKind.TRANSPOSE, "memory_access_efficiency", 2.0, 1.9),
        ("d", WorkloadKind.REDUCTION, "lds_barrier_pressure", 3.0, 3.1),
        ("e", WorkloadKind.REDUCTION, "lds_barrier_pressure", 3.0, 2.9),
        ("f", WorkloadKind.MATMUL, "matrix_path_missing", 4.0, 4.2),
    ]

    result = evaluate_diagnostic_acceptance(
        [_case(*record) for record in records]
    )

    assert result.accepted is True
    assert result.median_absolute_percentage_error <= 15.0
    assert result.p90_absolute_percentage_error <= 30.0


def test_tuning_samples_and_wrong_attribution_fail_acceptance() -> None:
    case = _case("a", WorkloadKind.ELEMENTWISE, "wrong", 1.0, 2.0)
    case = case.model_copy(update={"tuning_sample": True})

    result = evaluate_diagnostic_acceptance([case])

    assert result.accepted is False
    assert "tuning_sample_in_acceptance" in result.reason_codes
    assert "primary_attribution_mismatch" in result.reason_codes
