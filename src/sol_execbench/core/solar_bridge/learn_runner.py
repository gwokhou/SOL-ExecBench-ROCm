# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Parent-side runner for offline handler learning."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from sol_execbench.core.data.json_utils import atomic_write_json_value, load_json_value
from sol_execbench.core.process import (
    redacted_file_tail,
    run_in_process_group_to_files,
)


def run_handler_learning(
    *,
    node_type: str,
    sample_path: Path,
    output_dir: Path,
    model: str,
    timeout_seconds: float = 600,
) -> dict[str, Any]:
    """Generate one candidate in a process with bounded, redacted logs."""
    with tempfile.TemporaryDirectory(prefix="sol-execbench-solar-learn-") as temp:
        root = Path(temp)
        request_path = root / "request.json"
        response_path = root / "response.json"
        stdout_path = root / "stdout.log"
        stderr_path = root / "stderr.log"
        atomic_write_json_value(
            request_path,
            {
                "node_type": node_type,
                "sample_path": str(sample_path.resolve()),
                "output_dir": str(output_dir.resolve()),
                "model": model,
            },
        )
        command = [
            sys.executable,
            "-m",
            "sol_execbench.core.solar_bridge.learn_worker",
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
            return _learning_failure(
                "worker_timeout",
                f"handler-learning worker exceeded {timeout_seconds:g} seconds",
            )
        except Exception as exc:
            return _learning_failure("worker_execution_failed", str(exc))
        if response_path.is_file():
            try:
                response = load_json_value(response_path)
                if not isinstance(response, dict):
                    raise ValueError("worker response must be a JSON object")
                return response
            except Exception as exc:
                return _learning_failure("worker_response_invalid", str(exc))
        detail = redacted_file_tail(stderr_path) or redacted_file_tail(stdout_path)
        return _learning_failure(
            "worker_no_response",
            detail or f"handler-learning worker exited {completed.returncode}",
        )


def _learning_failure(reason_code: str, message: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "reason_code": reason_code,
        "message": message[:4096],
    }


__all__ = ["run_handler_learning"]
