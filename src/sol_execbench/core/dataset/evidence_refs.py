# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Derived evidence reference helpers for dataset reports."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_SAFE_SIDECAR_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_sidecar_stem(*parts: str) -> str:
    """Return a deterministic filename stem for untrusted benchmark identifiers."""
    safe_parts: list[str] = []
    changed = False
    for part in parts:
        raw = str(part)
        safe = _SAFE_SIDECAR_COMPONENT.sub("_", raw)
        safe = re.sub(r"_+", "_", safe).strip("._")
        if not safe:
            raise ValueError(f"unsafe sidecar identifier: {raw!r}")
        changed = changed or safe != raw
        safe_parts.append(safe)

    safe_stem = ".".join(safe_parts)
    if changed:
        digest = hashlib.sha256(
            "\0".join(str(part) for part in parts).encode()
        ).hexdigest()[:12]
        safe_stem = f"{safe_stem}.{digest}"
    return safe_stem


def sidecar_stem_for_workload(
    definition_name: str,
    workload_uuid: str,
    *,
    problem_namespace: str | None = None,
) -> str:
    """Return a sidecar stem scoped to a problem when a namespace is available."""
    if problem_namespace:
        return safe_sidecar_stem(problem_namespace, definition_name, workload_uuid)
    return safe_sidecar_stem(definition_name, workload_uuid)


def relative_ref(path: Path, base: Path) -> str:
    """Return a stable path reference relative to *base* when possible."""
    path = path.resolve()
    base = base.resolve()
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.name


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
                gaps.append("amd_sol_bound_missing")
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
