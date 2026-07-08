# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""I/O helpers for trust summary sidecars."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sol_execbench.core.reports.trust_summary_models import TrustSummaryReport
from sol_execbench.core.reports.trust_summary_rendering import render_trust_summary_markdown


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from path."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path}")
    return dict(payload)


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
