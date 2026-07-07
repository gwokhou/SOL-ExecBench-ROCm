# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Report builder for paper denominator accounting sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sol_execbench.core.dataset.paper_denominator_assembly import (
    assemble_paper_denominator_report,
)
from sol_execbench.core.dataset.paper_denominator_models import (
    PaperDenominatorReport,
    PaperDenominatorSourceRef,
)
from sol_execbench.core.dataset.paper_denominator_stages import (
    add_missing_artifact_evidence,
    finalize_rollups,
    merge_execution_closure,
    merge_readiness,
    seed_inventory,
)
from sol_execbench.core.dataset.paper_denominator_state import PaperDenominatorBuildState


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_paper_denominator_report(
    *,
    inventory: dict[str, Any],
    readiness: dict[str, Any],
    execution_closure: dict[str, Any],
    manifest: dict[str, Any] | None = None,
    ready_subset: dict[str, Any] | None = None,
    amd_score_report: dict[str, Any] | None = None,
    amd_sol_artifacts: list[PaperDenominatorSourceRef | dict[str, Any] | str | Path]
    | None = None,
    solar_artifacts: list[PaperDenominatorSourceRef | dict[str, Any] | str | Path]
    | None = None,
    source_paths: dict[str, Path | None] | None = None,
    created_at: str | None = None,
) -> PaperDenominatorReport:
    state = PaperDenominatorBuildState(
        inventory=inventory,
        readiness=readiness,
        execution_closure=execution_closure,
        manifest=manifest,
        ready_subset=ready_subset,
        amd_score_report=amd_score_report,
        amd_sol_artifacts=amd_sol_artifacts or [],
        solar_artifacts=solar_artifacts or [],
        source_paths=source_paths or {},
    )
    seed_inventory(state)
    merge_readiness(state)
    merge_execution_closure(state)
    add_missing_artifact_evidence(state)
    finalize_rollups(state)
    return assemble_paper_denominator_report(state, created_at=created_at)
