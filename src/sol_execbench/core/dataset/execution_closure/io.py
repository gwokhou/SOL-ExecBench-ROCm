# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Execution closure sidecar contract helpers."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.dataset.execution_closure.models import ExecutionClosureReport


def write_execution_closure_report(report: ExecutionClosureReport, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
