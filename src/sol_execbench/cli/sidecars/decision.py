# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Decision sidecar helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from sol_execbench.cli.sidecars.mode import SidecarMode
from sol_execbench.core.bench.decision.builder import build_decision_sidecar
from sol_execbench.core.bench.decision.precedence import (
    apply_runtime_precedence,
)
from sol_execbench.core.bench.decision.runtime import (
    runtime_decision_precedence,
)
from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult
from sol_execbench.core.bench.static_kernel.evidence import (
    StaticKernelEvidenceSidecar,
)
from sol_execbench.core.evidence.runtime_evidence import write_json_payload
from sol_execbench.core.platform.arch_capabilities import (
    ArchCapabilityBudgetStatus,
    ArchISABudget,
    arch_capability_budget_from_dict,
)

console = Console(stderr=True)


def _load_budget_from_environment(
    environment_sidecar_path: Path | None,
    *,
    target_architecture: str | None = None,
) -> ArchISABudget | None:
    """Read the arch capability budget from an environment sidecar.

    When ``target_architecture`` is set (the static-evidence detected gfx), the
    matching budget is preferred over other available budgets so a multi-GPU
    environment does not yield the wrong arch's limits.
    """
    if (
        environment_sidecar_path is None
        or not environment_sidecar_path.is_file()
    ):
        return None
    try:
        payload = json.loads(
            environment_sidecar_path.read_text(encoding="utf-8"),
        )
    except (json.JSONDecodeError, OSError):
        return None
    candidates = [
        entry
        for entry in (payload.get("capability_budgets") or [])
        if entry.get("status") == ArchCapabilityBudgetStatus.AVAILABLE
        and entry.get("budget")
    ]
    if target_architecture:
        norm = target_architecture.split(":")[0].strip().lower()
        candidates.sort(
            key=lambda entry: (entry.get("architecture") or "").lower() != norm,
        )
    for entry in candidates:
        try:
            return arch_capability_budget_from_dict(entry["budget"])
        except (ValueError, TypeError, KeyError):
            # Skip a malformed budget and try the next candidate rather than
            # aborting the whole loop (a later valid budget may be usable).
            continue
    return None


def _write_decision_sidecar(
    output_file: Path | None,
    enabled: SidecarMode,
    static_evidence_result: StaticKernelEvidenceSidecar | None,
    environment_sidecar_path: Path | None,
    *,
    profile_result: Rocprofv3ProfileResult | None = None,
    run_id: str | None = None,
    target_id: str | None = None,
    candidate_id: str | None = None,
    source_sha256: str | None = None,
    sol_version: str | None = None,
) -> Path | None:
    """Write an optional Layer R decision sidecar derived from static footprints."""
    if enabled is SidecarMode.NONE or output_file is None:
        return None
    footprints = (
        list(static_evidence_result.footprints)
        if static_evidence_result is not None
        else []
    )
    if not footprints:
        return None
    target_architecture = None
    if static_evidence_result is not None:
        detected = static_evidence_result.classification.detected_architectures
        if detected:
            target_architecture = detected[0]
    budget = _load_budget_from_environment(
        environment_sidecar_path,
        target_architecture=target_architecture,
    )
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
    # Runtime profiling takes precedence only after a successful profiler result
    # yields a known classification. File presence, partial diagnostics, and
    # unknown/insufficient-counter summaries carry no precedence.
    runtime_precedence = runtime_decision_precedence(profile_result)
    if runtime_precedence.available:
        sidecar = apply_runtime_precedence(
            sidecar,
            runtime_profile_available=True,
            demoted_classes=set(runtime_precedence.demoted_classes),
        )
    sidecar_path = output_file.with_name(f"{output_file.name}.decision.json")
    try:
        write_json_payload(sidecar_path, sidecar.to_dict())
        console.print(
            "[green]Decision sidecar "
            f"{sidecar.status}; saved hints to {sidecar_path}[/green]",
        )
        return sidecar_path
    except Exception as exc:  # noqa: BLE001 -- optional decision sidecar
        console.print(f"[yellow]Decision sidecar skipped: {exc}[/yellow]")
        return None
