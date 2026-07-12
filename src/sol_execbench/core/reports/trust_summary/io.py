# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""I/O helpers for trust summary sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sol_execbench.core.data.json_utils import load_json_dict
from sol_execbench.core.reports.trust_summary.models import TrustSummaryReport
from sol_execbench.core.reports.trust_summary.rendering import (
    render_trust_summary_markdown,
)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from path."""

    return load_json_dict(path)


def write_trust_summary_reports(
    report: TrustSummaryReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Write trust summary JSON and Markdown reports."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(render_trust_summary_markdown(report), encoding="utf-8")
