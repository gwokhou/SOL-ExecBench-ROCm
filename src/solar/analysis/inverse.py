# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Necessary resource inequalities for fixed-capacity whole-graph views."""

from __future__ import annotations

from math import fsum, isclose, isfinite

from solar.analysis.exploration_contracts import (
    MEMORY_RESOURCE,
    FixedModel,
    InverseReason,
    InverseResult,
    InverseStatus,
    ModeWork,
    ResourceConstraint,
)
from solar.analysis.views import AnalysisView, ViewAvailability

COMPUTE_RATE_UNIT = "work_units/s"
MEMORY_RATE_UNIT = "B/s"
MODEL_ROUNDING_RELATIVE_TOLERANCE = 1e-12


def solve_constraints(
    model: FixedModel,
    *,
    target_seconds: float,
    scale_resources: tuple[str, ...] = (),
) -> InverseResult:
    """Solve shared-mode sums; never substitute this for serial operator inversion."""
    if (
        isinstance(target_seconds, bool)
        or not isfinite(target_seconds)
        or target_seconds <= 0
    ):
        raise ValueError("target time must be finite and positive")
    if len(set(scale_resources)) != len(scale_resources):
        raise ValueError("scaled resource group contains duplicates")
    unknown = set(scale_resources) - {*model.resources, MEMORY_RESOURCE}
    if unknown:
        raise ValueError(f"scaled resource group is unknown: {sorted(unknown)}")
    common: dict[str, object] = {
        "projection_sha256": model.projection_sha256,
        "view": model.view,
        "computation": model.computation,
        "target_seconds": target_seconds,
        "view_reason": model.estimate.reason.value,
        "scale_resources": scale_resources,
    }
    refused = _refusal(model, common)
    if refused is not None:
        return refused
    constraints = _constraints(model, target_seconds)
    group = max(
        (constraints[name].required_common_scale for name in scale_resources),
        default=None,
    )
    return InverseResult.model_validate(
        common
        | {
            "status": InverseStatus.AVAILABLE,
            "reason": InverseReason.READY,
            "constraints": constraints,
            "required_group_scale": group,
            "fixed_constraints_satisfied": all(
                item.currently_satisfied
                for name, item in constraints.items()
                if name not in scale_resources
            ),
            "forward_seconds_at_group_scale": (
                _forward_scaled(constraints, scale_resources, group)
                if group is not None
                else None
            ),
        }
    )


def _refusal(
    model: FixedModel, common: dict[str, object]
) -> InverseResult | None:
    if model.view is AnalysisView.UNFUSED:
        return InverseResult.model_validate(
            common
            | {
                "status": InverseStatus.UNSUPPORTED,
                "reason": InverseReason.UNFUSED,
            }
        )
    if model.estimate.status is ViewAvailability.UNAVAILABLE:
        return InverseResult.model_validate(
            common
            | {
                "status": InverseStatus.UNAVAILABLE,
                "reason": InverseReason.VIEW_UNAVAILABLE,
            }
        )
    return None


def _constraint(
    modes: dict[str, ModeWork], target: float, unit: str
) -> ResourceConstraint:
    seconds = fsum(item.work / item.rate for item in modes.values())
    return ResourceConstraint(
        work_by_mode={name: item.work for name, item in modes.items()},
        rates_by_mode={name: item.rate for name, item in modes.items()},
        rate_unit=unit,
        baseline_seconds=seconds,
        required_common_scale=seconds / target,
        minimum_single_mode_rate=next(iter(modes.values())).work / target
        if len(modes) == 1
        else None,
        currently_satisfied=seconds <= target,
    )


def _constraints(
    model: FixedModel, target: float
) -> dict[str, ResourceConstraint]:
    result = {
        name: _constraint(modes, target, COMPUTE_RATE_UNIT)
        for name, modes in model.resources.items()
    }
    if model.estimate.io_bytes is None:
        raise ValueError("available fixed model lacks traffic")
    result[MEMORY_RESOURCE] = _constraint(
        {"bytes": ModeWork(work=model.estimate.io_bytes, rate=model.bandwidth)},
        target,
        MEMORY_RATE_UNIT,
    )
    expected = {
        name: item.baseline_seconds
        for name, item in result.items()
        if name != MEMORY_RESOURCE
    }
    actual = model.estimate.resource_seconds
    if set(actual) != set(expected) or any(
        not isclose(
            actual[name],
            value,
            rel_tol=MODEL_ROUNDING_RELATIVE_TOLERANCE,
            abs_tol=0.0,
        )
        for name, value in expected.items()
    ):
        raise ValueError(
            "fixed model work/rates disagree with view resource times"
        )
    forward = max(item.baseline_seconds for item in result.values())
    if model.estimate.seconds is None or not isclose(
        forward,
        model.estimate.seconds,
        rel_tol=MODEL_ROUNDING_RELATIVE_TOLERANCE,
        abs_tol=0.0,
    ):
        raise ValueError("fixed model work/traffic disagree with view time")
    return result


def _forward_scaled(
    constraints: dict[str, ResourceConstraint],
    selected: tuple[str, ...],
    scale: float,
) -> float:
    return max(
        item.baseline_seconds
        if name not in selected
        else item.baseline_seconds / scale
        if scale
        else 0.0
        for name, item in constraints.items()
    )
