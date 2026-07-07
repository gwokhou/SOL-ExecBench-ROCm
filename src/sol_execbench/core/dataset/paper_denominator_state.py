# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Internal state for paper denominator report building."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sol_execbench.core.dataset.paper_denominator_models import (
    PaperDenominatorProblem,
    PaperDenominatorRollup,
    PaperDenominatorSourceRef,
    PaperDenominatorWorkload,
)


@dataclass
class PaperDenominatorBuildState:
    inventory: dict[str, Any]
    readiness: dict[str, Any]
    execution_closure: dict[str, Any]
    manifest: dict[str, Any] | None = None
    ready_subset: dict[str, Any] | None = None
    amd_score_report: dict[str, Any] | None = None
    amd_sol_artifacts: list[PaperDenominatorSourceRef | dict[str, Any] | str | Path] = field(default_factory=list)
    solar_artifacts: list[PaperDenominatorSourceRef | dict[str, Any] | str | Path] = field(default_factory=list)
    source_paths: dict[str, Path | None] = field(default_factory=dict)
    categories: dict[str, PaperDenominatorRollup] = field(default_factory=dict)
    problems: dict[tuple[str, str], PaperDenominatorProblem] = field(default_factory=dict)
    workloads: dict[tuple[str, int | None, str | None], PaperDenominatorWorkload] = field(default_factory=dict)
    reason_groups: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence_groups: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
