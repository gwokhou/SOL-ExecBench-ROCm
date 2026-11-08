"""Process-bounded CPU diagnostic execution, separate from official evaluation."""

import os
import subprocess
import sys
import time
from pathlib import Path

from sol_execbench.core.bench.judge.candidate_catalog import candidate_source
from sol_execbench.core.bench.judge.contracts import (
    JudgeCase,
    JudgeStrategy,
    development_cases,
)
from sol_execbench.core.bench.judge.report import (
    DiagnosticStatus,
    JudgeDevelopmentReport,
    JudgeResult,
)
from sol_execbench.core.bench.judge.storage import source_identity
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
)
from sol_execbench.core.integrity import sha256_bytes, sha256_file
from sol_execbench.core.process import (
    redacted_file_tail,
    run_in_process_group_to_files,
)

CASE_TIMEOUT_SECONDS = 120
CPU_ACTIVITY_SECONDS = 12 * 3600
MAX_EVIDENCE_BYTES = 20 * 1024**3
MAX_MEMORY_BYTES = 16 * 1024**3
CPU_WORKER_MODULE = "sol_execbench.core.bench.judge.worker"


def execute_development(
    repository: Path,
    output: Path,
    *,
    run_nonce: str,
    cases: tuple[JudgeCase, ...] | None = None,
) -> JudgeDevelopmentReport:
    """Execute each selected case in a new bounded CPU process, with no retries."""
    selected = development_cases() if cases is None else cases
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("development selection must be nonempty and unique")
    for case in selected:
        case.require_development()
    if len(run_nonce) != 64 or any(
        c not in "0123456789abcdef" for c in run_nonce
    ):
        raise ValueError("invalid private development nonce")
    output.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    started = time.monotonic()
    reports: list[JudgeDevelopmentReport] = []
    failure = None
    for case in selected:
        remaining = CPU_ACTIVITY_SECONDS - (time.monotonic() - started)
        try:
            if (
                remaining <= 0
                or sum(
                    p.stat().st_size for p in output.rglob("*") if p.is_file()
                )
                >= MAX_EVIDENCE_BYTES
            ):
                raise TimeoutError("CPU diagnostic activity budget exhausted")
            index = development_cases().index(case)
            case_output = output / f"case-{index:03d}"
            _run_worker(
                repository,
                case_output,
                index,
                run_nonce,
                min(CASE_TIMEOUT_SECONDS, remaining),
            )
            reports.append(
                load_json_file(
                    JudgeDevelopmentReport, case_output / "development.json"
                )
            )
            if not reports[-1].execution_complete:
                raise RuntimeError("oracle uncertainty prevented completion")
        except (OSError, RuntimeError, ValueError) as exc:
            failure = str(exc)
            break
    if source_identity() != source or any(
        report.source_sha256 != source for report in reports
    ):
        failure = "CPU diagnostic source changed during execution"
    return _finish_report(selected, reports, source, output, failure)


def _finish_report(
    selected: tuple[JudgeCase, ...],
    reports: list[JudgeDevelopmentReport],
    source: dict[str, str],
    output: Path,
    failure: str | None,
) -> JudgeDevelopmentReport:
    results = [row for item in reports for row in item.results]
    completed = {row.case_id for row in results}
    for case in selected:
        if case.logical_id in completed:
            continue
        for strategy in JudgeStrategy:
            results.append(
                JudgeResult(
                    case_id=case.logical_id,
                    strategy=strategy,
                    input_sha256=None,
                    candidate_sha256=sha256_bytes(
                        candidate_source(case.task, strategy).encode()
                    ),
                    static_blocked=None,
                    static_review={},
                    diagnostic_status=DiagnosticStatus.NOT_REACHED,
                    message=failure or "upstream CPU case did not complete",
                )
            )
    report = JudgeDevelopmentReport(
        complete_development_denominator=set(selected)
        == set(development_cases()),
        execution_complete=failure is None,
        failure_reason=failure,
        case_count=len(selected),
        runtime=reports[0].runtime if reports else {},
        costs={
            key: value for item in reports for key, value in item.costs.items()
        },
        independent_oracle_points={
            key: value
            for item in reports
            for key, value in item.independent_oracle_points.items()
        },
        source_sha256=source,
        artifacts={
            str(p.relative_to(output)): sha256_file(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
        results=results,
    )
    atomic_write_json_value(
        output / "development.json", report.model_dump(mode="json")
    )
    return report


def _run_worker(
    repository: Path,
    output: Path,
    index: int,
    nonce: str,
    timeout: float,
) -> None:
    request = output.with_suffix(".request.json")
    stdout, stderr = (
        output.with_suffix(".stdout.log"),
        output.with_suffix(".stderr.log"),
    )
    atomic_write_json_value(
        request,
        {
            "repository": str(repository.resolve()),
            "output": str(output.resolve()),
            "index": index,
            "run_nonce": nonce,
        },
    )
    request.chmod(0o600)
    try:
        completed = run_in_process_group_to_files(
            [sys.executable, "-m", CPU_WORKER_MODULE, str(request)],
            stdout,
            stderr,
            timeout=timeout,
            env={
                **os.environ,
                "HIP_VISIBLE_DEVICES": "",
                "ROCR_VISIBLE_DEVICES": "",
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        if completed.returncode:
            raise RuntimeError(
                f"CPU judge worker failed: {redacted_file_tail(stderr)}"
            )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"CPU judge case {index} exceeded {timeout:g}s"
        ) from exc
    finally:
        request.unlink(missing_ok=True)
