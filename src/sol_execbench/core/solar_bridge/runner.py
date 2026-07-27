# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded parent-side runner for the isolated SOLAR worker."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)
from sol_execbench.core.process import (
    redacted_file_tail,
    run_in_process_group_to_files,
)
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarAnalysisStatus,
    SolarReadinessStatus,
    SolarStage,
    SolarStageAuditOutcome,
    SolarStageAuditRequest,
    SolarWorkerRequest,
)


def run_solar_worker(
    request: SolarWorkerRequest,
    *,
    timeout_seconds: float = 14_400,
) -> SolarAnalysisOutcome:
    """Run one analysis with process-group cleanup and file-backed logs."""
    payload, failure = _run_worker_payload(
        request.to_dict(),
        module="sol_execbench.core.solar_bridge.worker",
        timeout_seconds=timeout_seconds,
    )
    if failure is not None:
        return _failed_outcome(request, *failure)
    try:
        outcome = SolarAnalysisOutcome.from_dict(payload or {})
        if (
            outcome.status is SolarAnalysisStatus.ANALYZED
            and not outcome.is_formal_publication
        ):
            raise ValueError(
                "SOLAR worker returned a non-formal analyzed response",
            )
        return outcome
    except Exception as exc:  # noqa: BLE001 -- worker response boundary
        return _failed_outcome(request, "worker_response_invalid", str(exc))


def run_solar_stage_worker(
    request: SolarStageAuditRequest,
    *,
    timeout_seconds: float = 14_400,
) -> SolarStageAuditOutcome:
    """Run one readiness audit with bounded isolated-process cleanup."""
    payload, failure = _run_worker_payload(
        request.to_dict(),
        module="sol_execbench.core.solar_bridge.stage_worker",
        timeout_seconds=timeout_seconds,
    )
    if failure is not None:
        return _failed_stage_outcome(request, *failure)
    try:
        outcome = SolarStageAuditOutcome.from_dict(payload or {})
        if outcome.status is SolarReadinessStatus.READY and not outcome.ready:
            raise ValueError(
                "SOLAR stage worker returned invalid ready evidence",
            )
        return outcome
    except Exception as exc:  # noqa: BLE001 -- worker response boundary
        return _failed_stage_outcome(
            request,
            "worker_response_invalid",
            str(exc),
        )


def _run_worker_payload(
    request: Mapping[str, Any],
    *,
    module: str,
    timeout_seconds: float,
) -> tuple[Mapping[str, Any] | None, tuple[str, str] | None]:
    with tempfile.TemporaryDirectory(
        prefix="sol-execbench-solar-worker-",
    ) as temp:
        root = Path(temp)
        request_path = root / "request.json"
        response_path = root / "response.json"
        stdout_path = root / "stdout.log"
        stderr_path = root / "stderr.log"
        atomic_write_json_value(request_path, dict(request))
        command = [
            sys.executable,
            "-m",
            module,
            str(request_path),
            str(response_path),
        ]
        try:
            completed = run_in_process_group_to_files(
                command,
                stdout_path,
                stderr_path,
                env=dict(os.environ),
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return None, (
                "worker_timeout",
                f"SOLAR worker exceeded {timeout_seconds:g} seconds",
            )
        except Exception as exc:  # noqa: BLE001 -- isolated process boundary
            return None, ("worker_execution_failed", str(exc))
        if response_path.is_file():
            try:
                payload = load_json_value(response_path)
                if not isinstance(payload, dict):
                    raise ValueError("worker response must be a JSON object")
                return payload, None
            except Exception as exc:  # noqa: BLE001 -- worker response boundary
                return None, ("worker_response_invalid", str(exc))
        stderr = redacted_file_tail(stderr_path)
        stdout = redacted_file_tail(stdout_path)
        detail = stderr or stdout or f"worker exited {completed.returncode}"
        return None, ("worker_no_response", detail)


def _failed_outcome(
    request: SolarWorkerRequest,
    reason_code: str,
    message: str,
) -> SolarAnalysisOutcome:
    return SolarAnalysisOutcome(
        status=SolarAnalysisStatus.FAILED,
        analysis_id=request.workload_uuid,
        stage=SolarStage.OUTER_BRIDGE,
        reason_code=reason_code,
        message=message[:4096],
    )


def _failed_stage_outcome(
    request: SolarStageAuditRequest,
    reason_code: str,
    message: str,
) -> SolarStageAuditOutcome:
    return SolarStageAuditOutcome(
        status=SolarReadinessStatus.FAILED,
        analysis_id=request.workload_uuid,
        failure_stage=SolarStage.OUTER_BRIDGE,
        reason_code=reason_code,
        message=message[:4096],
    )


__all__ = ["run_solar_stage_worker", "run_solar_worker"]
