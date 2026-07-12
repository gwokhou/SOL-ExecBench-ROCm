# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""File IO helpers for AMD bound sanity reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sol_execbench.core.data.json_utils import load_json_value

from .models import AmdBoundSanityReport
from .rendering import render_amd_bound_sanity_markdown


def load_json(path: Path) -> Any:
    return load_json_value(path)


def write_amd_bound_sanity_reports(
    report: AmdBoundSanityReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(render_amd_bound_sanity_markdown(report), encoding="utf-8")
