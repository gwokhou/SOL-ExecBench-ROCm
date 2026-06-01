# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset runner selection and run-state helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sol_execbench.core.dataset.categories import DEFAULT_CATEGORIES
from sol_execbench.core.dataset.execution_closure import (
    ExecutionClosureStatus,
    closure_status_for_trace_status,
)


def discover_problems(
    benchmark_dir: Path,
    categories: list[str] | None = None,
    *,
    known_categories: set[str] | None = None,
) -> list[Path]:
    """Return sorted problem directories containing definition and workload files."""
    allowed = known_categories or set(DEFAULT_CATEGORIES)
    if categories:
        roots = [benchmark_dir / category for category in categories]
    else:
        roots = sorted(
            path
            for path in benchmark_dir.iterdir()
            if path.is_dir() and path.name in allowed
        )

    problems: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for problem_dir in sorted(root.iterdir()):
            if (
                (problem_dir / "definition.json").exists()
                and (problem_dir / "workload.jsonl").exists()
            ):
                problems.append(problem_dir)
    return problems


def workload_key(uuid: str | None, row_index: int | None) -> tuple[str, str | int]:
    """Return the stable identity used for workload refs and trace matching."""
    if uuid:
        return ("uuid", uuid)
    return ("row_index", int(row_index or 0))


def read_workload_rows(workload_path: Path) -> list[tuple[int, dict[str, Any], str]]:
    """Read non-empty workload JSONL rows as row-index, payload, raw-line tuples."""
    rows: list[tuple[int, dict[str, Any], str]] = []
    for row_index, line in enumerate(workload_path.read_text().splitlines()):
        if not line.strip():
            continue
        import json

        rows.append((row_index, json.loads(line), line))
    return rows


def ready_problem_map(ready_subset: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Index ready-subset problems by problem id."""
    if ready_subset is None:
        return {}
    return {
        str(problem["problem_id"]): problem
        for problem in ready_subset.get("problems", [])
    }


def readiness_workload_map(
    readiness: dict[str, Any] | None,
) -> dict[tuple[str, tuple[str, str | int]], dict[str, Any]]:
    """Index readiness workload records by problem id plus workload key."""
    if readiness is None:
        return {}
    indexed: dict[tuple[str, tuple[str, str | int]], dict[str, Any]] = {}
    for workload in readiness.get("workloads", []):
        key = workload_key(workload.get("workload_uuid"), workload.get("row_index"))
        indexed[(str(workload.get("problem_id")), key)] = workload
    return indexed


def selected_workload_rows(
    workload_path: Path,
    workload_refs: list[dict[str, Any]],
    *,
    max_workloads: int | None,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Select ready-subset workload rows, returning selected, capped, and missing refs."""
    rows = read_workload_rows(workload_path)
    by_uuid = {
        str(payload.get("uuid")): (row_index, payload, line)
        for row_index, payload, line in rows
        if payload.get("uuid")
    }
    by_row = {row_index: (row_index, payload, line) for row_index, payload, line in rows}

    selected: list[tuple[int, dict[str, Any], str, dict[str, Any]]] = []
    missing: list[dict[str, Any]] = []
    seen: set[tuple[str, str | int]] = set()
    for ref in workload_refs:
        uuid = ref.get("uuid")
        row_index = int(ref.get("row_index", 0))
        key = workload_key(uuid, row_index)
        if key in seen:
            continue
        seen.add(key)
        match = by_uuid.get(str(uuid)) if uuid else None
        if match is None:
            match = by_row.get(row_index)
        if match is None:
            missing.append(ref)
            continue
        selected.append((*match, ref))

    selected.sort(key=lambda item: item[0])
    capped = selected
    cap_filtered: list[dict[str, Any]] = []
    if max_workloads is not None and len(selected) > max_workloads:
        capped = selected[:max_workloads]
        cap_filtered = [item[3] for item in selected[max_workloads:]]

    return [item[2] for item in capped], [item[3] for item in capped], cap_filtered, missing


def trace_map(traces: list[dict[str, Any]]) -> dict[tuple[str, str | int], dict[str, Any]]:
    """Index traces by workload UUID when present, otherwise row index."""
    indexed: dict[tuple[str, str | int], dict[str, Any]] = {}
    for row_index, trace in enumerate(traces):
        workload = trace.get("workload") or {}
        key = workload_key(workload.get("uuid"), row_index)
        indexed[key] = trace
    return indexed


def trace_status(trace: dict[str, Any] | None) -> str | None:
    """Return the evaluation status from a trace payload."""
    if not trace:
        return None
    evaluation = trace.get("evaluation") or {}
    return evaluation.get("status", "UNKNOWN")


def closure_status_for_trace(
    trace: dict[str, Any] | None,
    *,
    skipped: bool = False,
) -> str:
    """Map a trace status to the execution-closure serialized status."""
    return closure_status_for_trace_status(trace_status(trace), skipped=skipped).value


def closure_status_with_evidence(status: str, evidence_gaps: list[str]) -> str:
    """Apply derived-evidence gaps to an execution-closure status."""
    from sol_execbench.core.dataset.execution_closure import closure_status_with_evidence

    return closure_status_with_evidence(
        ExecutionClosureStatus(status),
        evidence_gaps,
    ).value


def requested_evidence_requirements(
    *,
    amd_score_report: Path | None = None,
    amd_sol_bound_dir: Path | None = None,
    solar_derivation: Path | None = None,
    timing_evidence_dir: Path | None = None,
) -> list[str]:
    """Return requested derived evidence keys from runner option values."""
    requirements: list[str] = []
    if amd_score_report is not None:
        requirements.append("amd_score")
    if amd_sol_bound_dir is not None:
        requirements.append("amd_sol_bound")
    if solar_derivation is not None:
        requirements.append("solar_derivation")
    if timing_evidence_dir is not None:
        requirements.append("timing_evidence")
    return requirements
