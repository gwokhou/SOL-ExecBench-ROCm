# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Status and boundary helpers for SOLAR derivation evidence."""

from __future__ import annotations

from typing import Any

from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence


SOLAR_DERIVATION_STATUSES = frozenset({"scored", "degraded", "unscored"})
SOLAR_DERIVATION_SOURCE_BOUNDARY_FIELDS = frozenset(
    {
        "canonical_trace_jsonl",
        "public_schema",
        "candidate_solution_execution",
    }
)


def status_for_confidence(confidence: EstimateConfidence) -> str:
    """Map estimate confidence to SOLAR derivation status."""
    if confidence == EstimateConfidence.SUPPORTED:
        return "scored"
    if confidence == EstimateConfidence.INEXACT:
        return "degraded"
    return "unscored"


def unique_sorted(items: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return deterministic unique string values."""
    return tuple(sorted(dict.fromkeys(items)))


def empty_status_counts() -> dict[str, int]:
    """Return zero counts for every SOLAR derivation status."""
    return {status: 0 for status in sorted(SOLAR_DERIVATION_STATUSES)}


def ordered_status_counts(status_counts: dict[str, int]) -> dict[str, int]:
    """Return status counts ordered by locked SOLAR status vocabulary."""
    return {
        status: int(status_counts.get(status, 0))
        for status in sorted(SOLAR_DERIVATION_STATUSES)
    }


def default_source_boundary() -> dict[str, bool]:
    """Return the default sidecar-only SOLAR source boundary."""
    return {
        "canonical_trace_jsonl": False,
        "public_schema": False,
        "candidate_solution_execution": False,
    }


def derivation_warnings(
    graph_warnings: tuple[str, ...],
    estimates: tuple[Any, ...],
) -> tuple[str, ...]:
    """Build deterministic aggregate warnings for SOLAR derivation evidence."""
    warnings = [f"graph_warning:{warning}" for warning in graph_warnings]
    for estimate in estimates:
        warnings.extend(
            f"estimate_warning:{estimate.node_id}:{warning}"
            for warning in estimate.warnings
        )
        if estimate.confidence == EstimateConfidence.INEXACT:
            warnings.append(
                f"inexact_operator:{estimate.node_id}:{estimate.op_family.value}"
            )
        elif estimate.confidence == EstimateConfidence.UNSUPPORTED:
            warnings.append(
                f"unsupported_operator:{estimate.node_id}:{estimate.op_family.value}"
            )
    if any(estimate.confidence == EstimateConfidence.UNSUPPORTED for estimate in estimates):
        warnings.append("aggregate_unscored:unsupported semantic evidence")
    elif any(estimate.confidence == EstimateConfidence.INEXACT for estimate in estimates):
        warnings.append("aggregate_degraded:incomplete semantic evidence")
    return unique_sorted(warnings)
