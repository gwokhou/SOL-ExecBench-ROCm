"""Serial two-path audit of an existing LLM_CORE target view."""

import os
import time
from pathlib import Path

from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
)
from sol_execbench.core.dataset.target_view import (
    MaterializedTargetView,
    require_llm_core_scope,
    scoped_artifact,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.process import exclusive_file_lock
from sol_execbench.core.process.resource_guard import WorkerResourceBudget
from sol_execbench.core.solar_bridge.coverage_evidence import (
    classify_failure,
    verification_dimensions,
    verify_formal_outcome,
    verify_stage_outcome,
)
from sol_execbench.core.solar_bridge.coverage_models import (
    COVERAGE_SOURCE_SHA256,
    COVERAGE_TARGET_IDENTITY_SHA256,
    CoverageAttempt,
    CoverageOptions,
    CoverageRow,
    CoverageStatus,
    TargetCoverageReport,
)
from sol_execbench.core.solar_bridge.coverage_plan import new_coverage_report
from sol_execbench.core.solar_bridge.coverage_resume import (
    authenticate_resumed_attempts,
)
from sol_execbench.core.solar_bridge.models import (
    SolarStageAuditRequest,
    SolarWorkerRequest,
)
from sol_execbench.core.solar_bridge.observation import StageObservation
from sol_execbench.core.solar_bridge.runner import (
    run_solar_stage_worker,
    run_solar_worker,
)
from solar.analysis.orojenesis.runner import OrojenesisRunner

_REPORT_FILENAME = "target-coverage.json"
_STOP_STATUSES = frozenset(
    {CoverageStatus.INFRASTRUCTURE_ERROR, CoverageStatus.BUDGET_EXHAUSTED}
)


def audit_target_coverage(
    target: MaterializedTargetView,
    output: Path,
    options: CoverageOptions,
    *,
    resume: bool = False,
) -> TargetCoverageReport:
    """Admit only the frozen C0 source and complete denominator to execution."""
    require_llm_core_scope(target)
    if (
        target.view.source_manifest_sha256 != COVERAGE_SOURCE_SHA256
        or target.view.target_descriptor_sha256
        != COVERAGE_TARGET_IDENTITY_SHA256
    ):
        raise ValueError("M0-C0 source identity is not frozen")
    return _audit_validated_target(target, output, options, resume=resume)


def _audit_validated_target(
    target: MaterializedTargetView,
    output: Path,
    options: CoverageOptions,
    *,
    resume: bool = False,
) -> TargetCoverageReport:
    """Invoke production workers only after validated materialized-view admission."""
    options = options.model_copy(
        update={
            "memory_bytes": min(
                options.memory_bytes,
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2,
            )
        }
    )
    mapper = OrojenesisRunner(options.orojenesis_home)
    expected = new_coverage_report(target, options, mapper.toolchain_identity)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(output / ".coverage.lock"):
        report = _resume_or_create(output, expected, resume)
        if resume:
            authenticate_resumed_attempts(target, report, output, options)
        for index, row in enumerate(report.rows):
            if row.attempts:
                continue
            stop = _stop_reason(report, output, options, row)
            if stop:
                row.reason = stop
                continue
            _execute_row(target, output, options, report, row, index)
        report.matrix_complete = all(
            row.attempts and row.status not in _STOP_STATUSES
            for row in report.rows
        )
        report.costs_observed = _costs_observed(report)
        report.complete = report.matrix_complete and report.costs_observed
        _save(output, report)
    return report


def _resume_or_create(
    output: Path, expected: TargetCoverageReport, resume: bool
) -> TargetCoverageReport:
    path = output / _REPORT_FILENAME
    if not resume:
        if any(item.name != ".coverage.lock" for item in output.iterdir()):
            raise FileExistsError("coverage output is not empty")
        _save(output, expected)
        return expected
    report = load_json_file(TargetCoverageReport, path)
    if (
        report.identity != expected.identity
        or report.identity_sha256 != expected.identity_sha256
    ):
        raise ValueError(
            "coverage resume identity differs from current sources, runtime, view or budgets"
        )
    if [row.key for row in report.rows] != [row.key for row in expected.rows]:
        raise ValueError("coverage resume row inventory mismatch")
    for row, planned in zip(report.rows, expected.rows, strict=True):
        if row.model_dump(
            mode="json", exclude={"status", "reason", "attempts"}
        ) != planned.model_dump(
            mode="json", exclude={"status", "reason", "attempts"}
        ):
            raise ValueError("coverage resume row metadata mismatch")
        for attempt in row.attempts:
            if attempt.reason == "interrupted_in_progress":
                raise ValueError(
                    "interrupted worker requires an explicit new campaign; elapsed budget is unknown"
                )
            for relative, digest in attempt.artifacts.items():
                if sha256_file(scoped_artifact(output, relative)) != digest:
                    raise ValueError("coverage resume artifact drift")
    return report


def _stop_reason(
    report: TargetCoverageReport,
    output: Path,
    options: CoverageOptions,
    row: CoverageRow,
) -> str | None:
    if report.elapsed_seconds >= options.activity_seconds:
        return "activity budget exhausted"
    if (
        row.slot == "smoke"
        and report.smoke_elapsed_seconds >= options.smoke_seconds
    ):
        return "smoke budget exhausted"
    smoke = [item for item in report.rows if item.slot == "smoke"]
    if any(item.status in _STOP_STATUSES for item in smoke):
        return "smoke infrastructure or budget failure blocks expansion"
    if row.slot != "smoke" and not all(item.attempts for item in smoke):
        return "smoke incomplete; expansion not reached"
    if (
        sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        >= options.evidence_bytes
    ):
        return "evidence byte budget exhausted"
    return None


def _execute_row(
    target: MaterializedTargetView,
    output: Path,
    options: CoverageOptions,
    report: TargetCoverageReport,
    row: CoverageRow,
    index: int,
) -> None:
    directory = output / f"row-{index:03d}" / f"attempt-{len(row.attempts):02d}"
    directory.mkdir(parents=True, exist_ok=False)
    row.attempts.append(
        CoverageAttempt(
            status=CoverageStatus.INFRASTRUCTURE_ERROR,
            wall_seconds=0,
            reason="interrupted_in_progress",
        )
    )
    row.status = CoverageStatus.INFRASTRUCTURE_ERROR
    _save(output, report)
    start = time.monotonic()
    remaining = options.activity_seconds - report.elapsed_seconds
    if row.slot == "smoke":
        remaining = min(
            remaining, options.smoke_seconds - report.smoke_elapsed_seconds
        )
    try:
        attempt = _audit_row(target, row, directory, options, remaining)
    except (ValueError, OSError) as exc:
        attempt = CoverageAttempt(
            status=CoverageStatus.INFRASTRUCTURE_ERROR,
            wall_seconds=time.monotonic() - start,
            reason=str(exc),
        )
    attempt.wall_seconds = time.monotonic() - start
    report.elapsed_seconds += attempt.wall_seconds
    if row.slot == "smoke":
        report.smoke_elapsed_seconds += attempt.wall_seconds
    attempt.artifacts = {
        str(p.relative_to(output)): sha256_file(p)
        for p in directory.rglob("*")
        if p.is_file() and not p.is_symlink()
    }
    row.attempts[-1] = attempt
    row.status = attempt.status
    row.reason = attempt.reason or "completed; see original worker evidence"
    _save(output, report)


def _audit_row(
    target: MaterializedTargetView,
    row: CoverageRow,
    directory: Path,
    options: CoverageOptions,
    remaining: float,
) -> CoverageAttempt:
    started = time.monotonic()
    budget = WorkerResourceBudget(
        memory_bytes=options.memory_bytes,
        evidence_root=directory.parents[1],
        evidence_bytes=options.evidence_bytes,
    )
    readiness = run_solar_stage_worker(
        SolarStageAuditRequest(
            problem_dir=str(target.problem_dirs[row.semantic_id]),
            workload_uuid=row.workload_uuid,
            output_dir=str(directory / "readiness"),
            device=options.device,
            ir_path=row.ir_path,
        ),
        timeout_seconds=min(options.readiness_seconds, remaining),
        observation_path=directory / "readiness-cost.json",
        resource_budget=budget,
    )
    attempt = CoverageAttempt(
        status=CoverageStatus.NOT_REACHED,
        readiness=readiness.to_dict(),
        wall_seconds=0,
    )
    verify_stage_outcome(
        target, row, readiness, directory / "readiness", options.device
    )
    attempt.observations["readiness"] = _observation(
        directory / "readiness-cost.json"
    )
    if not readiness.ready:
        attempt.status = classify_failure(
            readiness.reason_code, readiness.failure_stage
        )
        attempt.reason = (
            readiness.reason_code
            or "unknown readiness failure; attribution required"
        )
        return attempt
    attempt.observations["verification"] = verification_dimensions(
        directory / "readiness"
    )
    remaining -= time.monotonic() - started
    if remaining <= 0:
        attempt.status, attempt.reason = (
            CoverageStatus.BUDGET_EXHAUSTED,
            "activity budget before formal",
        )
        return attempt
    return _complete_formal_attempt(
        target, row, directory, options, remaining, budget, attempt
    )


def _complete_formal_attempt(
    target: MaterializedTargetView,
    row: CoverageRow,
    directory: Path,
    options: CoverageOptions,
    remaining: float,
    budget: WorkerResourceBudget,
    attempt: CoverageAttempt,
) -> CoverageAttempt:
    formal = run_solar_worker(
        SolarWorkerRequest(
            problem_dir=str(target.problem_dirs[row.semantic_id]),
            workload_uuid=row.workload_uuid,
            output_dir=str(directory / "formal"),
            device=options.device,
            ir_path=row.ir_path,
            orojenesis_home=str(options.orojenesis_home),
        ),
        timeout_seconds=min(options.formal_seconds, remaining),
        observation_path=directory / "formal-cost.json",
        resource_budget=budget,
    )
    attempt.formal = formal.to_dict()
    _, projection = verify_formal_outcome(
        target, row, formal, directory / "formal"
    )
    attempt.observations.update(projection)
    attempt.observations["formal"] = _observation(
        directory / "formal-cost.json"
    )
    attempt.status = (
        CoverageStatus.PASSED
        if formal.is_formal_publication
        else classify_failure(formal.reason_code, formal.stage)
    )
    attempt.reason = formal.reason_code
    return attempt


def _observation(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "status": "not_observable",
            "reason": "worker did not finish cost sidecar",
        }
    return load_json_file(StageObservation, path).model_dump(mode="json")


def _costs_observed(report: TargetCoverageReport) -> bool:
    for row in report.rows:
        if not row.attempts:
            return False
        attempt = row.attempts[-1]
        names = (
            ("readiness", "formal")
            if attempt.formal is not None
            else ("readiness",)
        )
        for name in names:
            observation = attempt.observations.get(name)
            if not isinstance(observation, dict) or not observation.get(
                "records"
            ):
                return False
    return True


def _save(output: Path, report: TargetCoverageReport) -> None:
    atomic_write_json_value(
        output / _REPORT_FILENAME, report.model_dump(mode="json")
    )
