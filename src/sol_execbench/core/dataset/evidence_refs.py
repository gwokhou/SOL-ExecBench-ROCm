# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Derived evidence reference helpers for dataset reports."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.evidence_refs import (
    relative_ref,
    safe_sidecar_stem,
    sidecar_stem_for_workload,
)

__all__ = [
    "build_derived_evidence_refs",
    "relative_ref",
    "safe_sidecar_stem",
    "sidecar_stem_for_workload",
]


def build_derived_evidence_refs(
    *,
    definition_name: str,
    workload_uuid: str | None,
    problem_output_dir: Path,
    output_dir: Path,
    amd_score_report: Path | None,
    sol_bound_artifact_dir: Path | None,
    solar_derivation_dir: Path | None,
    timing_evidence_dir: Path | None,
    category: str,
) -> tuple[dict[str, str], list[str]]:
    """Build closure evidence references and missing-evidence gap labels."""
    refs: dict[str, str] = {}
    gaps: list[str] = []
    if amd_score_report is not None:
        refs["amd_score"] = relative_ref(amd_score_report.resolve(), output_dir)
    if workload_uuid:
        sidecar_stem = sidecar_stem_for_workload(
            definition_name,
            workload_uuid,
            problem_namespace=f"{category}/{problem_output_dir.name}",
        )
        if sol_bound_artifact_dir is not None:
            path = sol_bound_artifact_dir / f"{sidecar_stem}.amd-sol-v2.json"
            if path.exists():
                refs["amd_sol_bound"] = relative_ref(path, output_dir)
            else:
                gaps.append("amd_sol_evidence_missing")
        if solar_derivation_dir is not None:
            path = solar_derivation_dir / f"{sidecar_stem}.solar-derivation.json"
            if path.exists():
                refs["solar_derivation"] = relative_ref(path, output_dir)
            else:
                gaps.append("solar_derivation_missing")
    if timing_evidence_dir is not None:
        path = (
            timing_evidence_dir.resolve()
            / category
            / f"{problem_output_dir.name}.timing.json"
        )
        if path.exists():
            refs["timing_evidence"] = relative_ref(path, output_dir)
        else:
            gaps.append("timing_evidence_missing")
    return refs, gaps
