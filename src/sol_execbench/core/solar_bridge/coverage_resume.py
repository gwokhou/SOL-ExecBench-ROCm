"""Revalidate current worker evidence when continuing a coverage campaign."""

from pathlib import Path

from sol_execbench.core.dataset.target_view import MaterializedTargetView
from sol_execbench.core.solar_bridge.coverage_evidence import (
    classify_failure,
    verify_formal_outcome,
    verify_stage_outcome,
)
from sol_execbench.core.solar_bridge.coverage_models import (
    CoverageOptions,
    CoverageStatus,
    TargetCoverageReport,
)
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarStageAuditOutcome,
)


def authenticate_resumed_attempts(
    target: MaterializedTargetView,
    report: TargetCoverageReport,
    output: Path,
    options: CoverageOptions,
) -> None:
    """Hashes alone do not authorize reuse: replay the existing receipt validators."""
    for index, row in enumerate(report.rows):
        for number, attempt in enumerate(row.attempts):
            directory = output / f"row-{index:03d}" / f"attempt-{number:02d}"
            if attempt.readiness is None:
                if attempt.status is not CoverageStatus.INFRASTRUCTURE_ERROR:
                    raise ValueError(
                        "resume attempt lacks original worker response"
                    )
                continue
            readiness = SolarStageAuditOutcome.from_dict(attempt.readiness)
            verify_stage_outcome(
                target, row, readiness, directory / "readiness", options.device
            )
            if readiness.ready and attempt.formal is not None:
                formal = SolarAnalysisOutcome.from_dict(attempt.formal)
                verify_formal_outcome(target, row, formal, directory / "formal")
                status = (
                    CoverageStatus.PASSED
                    if formal.is_formal_publication
                    else classify_failure(formal.reason_code, formal.stage)
                )
            elif not readiness.ready and attempt.formal is None:
                status = classify_failure(
                    readiness.reason_code, readiness.failure_stage
                )
            elif (
                readiness.ready
                and attempt.status is CoverageStatus.BUDGET_EXHAUSTED
            ):
                status = CoverageStatus.BUDGET_EXHAUSTED
            else:
                raise ValueError(
                    "resume formal response lacks a successful upstream gate"
                )
            if attempt.status is not status:
                raise ValueError(
                    "resume classification differs from worker evidence"
                )
        if row.attempts and row.status is not row.attempts[-1].status:
            raise ValueError(
                "resume row differs from its last retained attempt"
            )
