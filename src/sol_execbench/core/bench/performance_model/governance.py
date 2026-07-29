# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Freshness and governance checks for performance diagnostics."""

from __future__ import annotations

from sol_execbench.core.bench.diagnostic_sidecar import (
    DiagnosticFreshnessStatus,
    DiagnosticFreshnessValidation,
    DiagnosticGovernanceGuardrail,
    classify_diagnostic_governance,
)
from sol_execbench.core.bench.performance_model.models import (
    PerformanceDiagnosticSidecar,
)


def validate_performance_diagnostic_freshness(
    sidecar: PerformanceDiagnosticSidecar,
    *,
    run_id: str,
    candidate_sha256: str,
    gpu_architecture: str,
    trace_sha256: str,
) -> DiagnosticFreshnessValidation:
    """Require exact run, candidate, GPU, and canonical trace identity."""
    reasons: list[str] = []
    if sidecar.run_id != run_id:
        reasons.append("run_id_mismatch")
    if sidecar.candidate_sha256 != candidate_sha256:
        reasons.append("candidate_sha256_mismatch")
    if sidecar.gpu_architecture != gpu_architecture:
        reasons.append("gpu_architecture_mismatch")
    trace_refs = [
        reference for reference in sidecar.evidence if reference.kind == "trace"
    ]
    if not trace_refs:
        reasons.append("trace_evidence_missing")
    elif all(reference.sha256 != trace_sha256 for reference in trace_refs):
        reasons.append("trace_sha256_mismatch")
    return DiagnosticFreshnessValidation(
        status=(
            DiagnosticFreshnessStatus.STALE
            if reasons
            else DiagnosticFreshnessStatus.CURRENT
        ),
        reason_codes=reasons,
    )


def evaluate_performance_diagnostic_governance(
    *,
    sidecar: PerformanceDiagnosticSidecar | None,
    freshness: DiagnosticFreshnessValidation | None = None,
    parse_error: str | None = None,
) -> DiagnosticGovernanceGuardrail:
    """Return the shared diagnostic-only governance classification."""
    status, reasons = classify_diagnostic_governance(
        sidecar_present=sidecar is not None,
        freshness_status=freshness.status if freshness else None,
        freshness_reason_codes=freshness.reason_codes if freshness else None,
        parse_error=parse_error,
    )
    return DiagnosticGovernanceGuardrail(
        status=status,
        reason_codes=reasons,
    )


__all__ = [
    "evaluate_performance_diagnostic_governance",
    "validate_performance_diagnostic_freshness",
]
