# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""I/O helpers for consistency reports."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.consistency_models import ConsistencyReport
from sol_execbench.core.consistency_rendering import render_consistency_markdown


def write_consistency_reports(
    report: ConsistencyReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Write consistency JSON and Markdown reports."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(render_consistency_markdown(report), encoding="utf-8")
