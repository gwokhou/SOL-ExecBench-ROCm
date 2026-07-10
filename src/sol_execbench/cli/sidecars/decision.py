# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Decision sidecar helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from ...core.bench.decision.builder import build_decision_sidecar
from ...core.bench.static_kernel.evidence import StaticKernelEvidenceSidecar
from ...core.evidence.runtime_evidence import write_json_payload
from ...core.platform.arch_capabilities import (
    ArchIsaBudget,
    arch_capability_budget_from_dict,
)

console = Console(stderr=True)

DECISION_NONE = "none"
DECISION_AUTO = "auto"


def _load_budget_from_environment(
    environment_sidecar_path: Path | None,
) -> ArchIsaBudget | None:
    """Read the first available arch capability budget from an environment sidecar."""

    if environment_sidecar_path is None or not environment_sidecar_path.is_file():
        return None
    try:
        payload = json.loads(environment_sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for entry in payload.get("capability_budgets") or []:
        if entry.get("status") == "available" and entry.get("budget"):
            try:
                return arch_capability_budget_from_dict(entry["budget"])
            except Exception:
                return None
    return None


def _write_decision_sidecar(
    output_file: Path | None,
    enabled: str,
    static_evidence_result: StaticKernelEvidenceSidecar | None,
    environment_sidecar_path: Path | None,
    *,
    run_id: str | None = None,
    target_id: str | None = None,
    candidate_id: str | None = None,
    source_sha256: str | None = None,
    sol_version: str | None = None,
) -> Path | None:
    """Write an optional Layer R decision sidecar derived from static footprints."""

    if enabled == DECISION_NONE or output_file is None:
        return None
    footprints = (
        list(static_evidence_result.footprints)
        if static_evidence_result is not None
        else []
    )
    if not footprints:
        return None
    budget = _load_budget_from_environment(environment_sidecar_path)
    sidecar = build_decision_sidecar(
        footprints=footprints,
        budget=budget,
        trace_path=str(output_file),
        run_id=run_id,
        target_id=target_id,
        candidate_id=candidate_id,
        source_sha256=source_sha256,
        sol_version=sol_version,
    )
    sidecar_path = output_file.with_name(f"{output_file.name}.decision.json")
    try:
        write_json_payload(sidecar_path, sidecar.to_dict())
        console.print(
            "[green]Decision sidecar "
            f"{sidecar.status.value}; saved hints to {sidecar_path}[/green]"
        )
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Decision sidecar skipped: {exc}[/yellow]")
        return None
