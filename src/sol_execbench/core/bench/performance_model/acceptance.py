# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Frozen held-out acceptance contract for the gfx1200 diagnostic model."""

from __future__ import annotations

import statistics

import numpy as np
from pydantic import ConfigDict, Field

from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.data.base_model import BaseModelWithDocstrings
from sol_execbench.core.integrity import SHA256Digest

_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    allow_inf_nan=False,
)
_EXPECTED_PRIMARY_CODES = {
    WorkloadKind.ELEMENTWISE: {"launch_bound"},
    WorkloadKind.TRANSPOSE: {"memory_access_efficiency"},
    WorkloadKind.REDUCTION: {"lds_barrier_pressure"},
    WorkloadKind.MATMUL: {"matrix_path_missing"},
}
_MINIMUM_CASES = {
    WorkloadKind.ELEMENTWISE: 1,
    WorkloadKind.TRANSPOSE: 2,
    WorkloadKind.REDUCTION: 2,
    WorkloadKind.MATMUL: 1,
}


class DiagnosticAcceptanceCase(BaseModelWithDocstrings):
    """One independent held-out workload result."""

    model_config = _MODEL_CONFIG

    workload_uuid: str
    workload_kind: WorkloadKind
    candidate_sha256: SHA256Digest
    tuning_sample: bool = False
    predicted_ms: float = Field(gt=0.0)
    measured_ms: float = Field(gt=0.0)
    primary_attribution_code: str


class DiagnosticAcceptanceResult(BaseModelWithDocstrings):
    """Aggregate frozen acceptance verdict."""

    model_config = _MODEL_CONFIG

    accepted: bool
    case_count: int = Field(ge=0)
    median_absolute_percentage_error: float = Field(ge=0.0)
    p90_absolute_percentage_error: float = Field(ge=0.0)
    attribution_matches: dict[WorkloadKind, bool]
    reason_codes: list[str] = Field(default_factory=list)


def evaluate_diagnostic_acceptance(
    cases: list[DiagnosticAcceptanceCase],
) -> DiagnosticAcceptanceResult:
    """Apply the plan's held-out accuracy and attribution thresholds."""
    reasons: list[str] = []
    if any(case.tuning_sample for case in cases):
        reasons.append("tuning_sample_in_acceptance")
    counts = {
        kind: sum(case.workload_kind is kind for case in cases)
        for kind in _MINIMUM_CASES
    }
    if any(counts[kind] < minimum for kind, minimum in _MINIMUM_CASES.items()):
        reasons.append("acceptance_workload_coverage_incomplete")
    errors = [
        abs(case.predicted_ms - case.measured_ms) / case.measured_ms * 100.0
        for case in cases
        if not case.tuning_sample
    ]
    median_error = statistics.median(errors) if errors else 0.0
    p90_error = float(np.percentile(errors, 90.0)) if errors else 0.0
    if not errors:
        reasons.append("held_out_samples_missing")
    if median_error > 15.0:
        reasons.append("median_absolute_percentage_error_exceeded")
    if p90_error > 30.0:
        reasons.append("p90_absolute_percentage_error_exceeded")
    attribution_matches = _attribution_matches(cases)
    if not all(attribution_matches.values()):
        reasons.append("primary_attribution_mismatch")
    return DiagnosticAcceptanceResult(
        accepted=not reasons,
        case_count=len(errors),
        median_absolute_percentage_error=median_error,
        p90_absolute_percentage_error=p90_error,
        attribution_matches=attribution_matches,
        reason_codes=reasons,
    )


def _attribution_matches(
    cases: list[DiagnosticAcceptanceCase],
) -> dict[WorkloadKind, bool]:
    result: dict[WorkloadKind, bool] = {}
    for kind, expected_codes in _EXPECTED_PRIMARY_CODES.items():
        kind_cases = [
            case
            for case in cases
            if case.workload_kind is kind and not case.tuning_sample
        ]
        result[kind] = bool(kind_cases) and all(
            case.primary_attribution_code in expected_codes
            for case in kind_cases
        )
    return result


__all__ = [
    "DiagnosticAcceptanceCase",
    "DiagnosticAcceptanceResult",
    "evaluate_diagnostic_acceptance",
]
