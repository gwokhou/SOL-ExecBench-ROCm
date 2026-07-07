# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""File IO helpers for AMD bound sanity reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .amd_bound_sanity_models import AmdBoundSanityReport
from .amd_bound_sanity_rendering import render_amd_bound_sanity_markdown

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

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
