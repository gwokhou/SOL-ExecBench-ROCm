# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded parent-side runner for the isolated SOLAR worker."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from sol_execbench.core.data.json_utils import atomic_write_json_value, load_json_value
from sol_execbench.core.process import (
    redacted_file_tail,
    run_in_process_group_to_files,
)
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarWorkerRequest,
)


def run_solar_worker(
    request: SolarWorkerRequest, *, timeout_seconds: float = 14_400
) -> SolarAnalysisOutcome:
    """Run one analysis with process-group cleanup and file-backed logs."""
    with tempfile.TemporaryDirectory(prefix="sol-execbench-solar-worker-") as temp:
        root = Path(temp)
        request_path = root / "request.json"
        response_path = root / "response.json"
        stdout_path = root / "stdout.log"
        stderr_path = root / "stderr.log"
        atomic_write_json_value(request_path, request.to_dict())
        command = [
            sys.executable,
            "-m",
            "sol_execbench.core.solar_bridge.worker",
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
            return _failed_outcome(
                request,
                "worker_timeout",
                f"SOLAR worker exceeded {timeout_seconds:g} seconds",
            )
        except Exception as exc:
            return _failed_outcome(request, "worker_execution_failed", str(exc))
        if response_path.is_file():
            try:
                return SolarAnalysisOutcome.from_dict(load_json_value(response_path))
            except Exception as exc:
                return _failed_outcome(request, "worker_response_invalid", str(exc))
        stderr = redacted_file_tail(stderr_path)
        stdout = redacted_file_tail(stdout_path)
        detail = stderr or stdout or f"worker exited {completed.returncode}"
        return _failed_outcome(request, "worker_no_response", detail)


def _failed_outcome(
    request: SolarWorkerRequest, reason_code: str, message: str
) -> SolarAnalysisOutcome:
    return SolarAnalysisOutcome(
        status="failed",
        analysis_id=request.workload_uuid,
        stage="outer_bridge",
        reason_code=reason_code,
        message=message[:4096],
    )


__all__ = ["run_solar_worker"]
