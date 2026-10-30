# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Stable model sensitivity and explicitly dimensioned Pareto summaries."""

from __future__ import annotations

from math import isfinite, log

from solar.analysis.scan_contracts import (
    AxisKind,
    ObjectiveKind,
    PointStatus,
    ScanDefinition,
    ScanPoint,
    ScanResult,
)
from solar.analysis.views import AnalysisView, ViewAvailability


def report_scan(
    request: ScanDefinition, result: ScanResult
) -> dict[str, object]:
    """Keep every point; only accepted estimates participate in comparisons."""
    available = [
        point for point in result.points if point.status is PointStatus.ANALYZED
    ]
    baseline = result.points[0]
    minimum = min((_seconds(point) for point in available), default=None)
    vectors = {
        point.point_id: tuple(
            _objective_value(item.kind, item.axis, point)
            for item in request.objectives
        )
        for point in available
    }
    pareto = (
        [
            point.point_id
            for point in available
            if not any(
                all(
                    left <= right
                    for left, right in zip(
                        other, vectors[point.point_id], strict=True
                    )
                )
                and any(
                    left < right
                    for left, right in zip(
                        other, vectors[point.point_id], strict=True
                    )
                )
                for other in vectors.values()
            )
        ]
        if request.objectives
        else []
    )
    return {
        "scan_sha256": result.scan_sha256,
        "view": request.view.value,
        "computation": request.computation.value,
        "planned_points": len(result.points),
        "attempted_points": sum(
            point.attempted and not point.resumed for point in result.points
        ),
        "resumed_points": sum(point.resumed for point in result.points),
        "available_points": len(available),
        "cache_aware_available_points": sum(
            point.views is not None
            and point.views[AnalysisView.CACHE_AWARE].status
            is ViewAvailability.AVAILABLE
            for point in result.points
        ),
        "baseline_point_id": result.baseline_point_id,
        "stop_reason": result.stop_reason,
        "best_point_ids": [
            point.point_id for point in available if _seconds(point) == minimum
        ],
        "pareto_point_ids": pareto,
        "objectives": _objective_descriptions(request),
        "points": [
            _point_report(point, baseline, request.target_seconds)
            for point in result.points
        ],
        "performance_authority": False,
        "publication_eligible": False,
        "comparison_scope": "same_source_path_view_and_declared_grid",
    }


def _seconds(point: ScanPoint) -> float:
    if point.estimate is None or point.estimate.seconds is None:
        raise ValueError("unavailable point cannot enter numerical comparisons")
    return point.estimate.seconds


def _objective_value(
    kind: ObjectiveKind, axis: str | None, point: ScanPoint
) -> float | int:
    if kind is ObjectiveKind.SECONDS:
        return _seconds(point)
    if axis is None:
        raise ValueError("axis objective has no axis")
    return point.coordinates[axis]


def _objective_descriptions(request: ScanDefinition) -> list[dict[str, object]]:
    axes = {axis.key: axis for axis in request.axes}
    return [
        {
            "kind": item.kind.value,
            "axis": item.axis,
            "direction": "minimize",
            "unit": "s"
            if item.kind is ObjectiveKind.SECONDS
            else "B"
            if item.axis is not None
            and axes[item.axis].kind is AxisKind.CAPACITY_BYTES
            else "ratio",
        }
        for item in request.objectives
    ]


def _point_report(
    point: ScanPoint, baseline: ScanPoint, target: float | None
) -> dict[str, object]:
    row = point.model_dump(mode="json")
    row.update(
        {
            "model_speedup": None,
            "single_axis_elasticity": None,
            "bottlenecks": [],
        }
    )
    if point.status is not PointStatus.ANALYZED:
        return row
    seconds = _seconds(point)
    row["target_necessary_condition"] = (
        {"target_seconds": target, "not_ruled_out_by_model": seconds <= target}
        if target is not None
        else None
    )
    if point.estimate is None:
        raise ValueError("analyzed point lacks its estimate")
    times = point.estimate.resource_seconds | {
        "dram": point.memory_seconds or 0.0
    }
    row["bottlenecks"] = sorted(
        name for name, value in times.items() if value == seconds
    )
    if baseline.status is not PointStatus.ANALYZED:
        return row
    initial = _seconds(baseline)
    ratio = initial / seconds if seconds else 1.0 if initial == 0 else None
    row["model_speedup"] = (
        ratio if ratio is not None and isfinite(ratio) else None
    )
    changed = [
        name
        for name, value in point.coordinates.items()
        if value != baseline.coordinates[name]
    ]
    if len(changed) == 1 and initial > 0 and seconds > 0:
        name = changed[0]
        elasticity = -(log(seconds) - log(initial)) / (
            log(point.coordinates[name]) - log(baseline.coordinates[name])
        )
        row["single_axis_elasticity"] = (
            {"axis": name, "value": elasticity, "unit": "dimensionless"}
            if isfinite(elasticity)
            else None
        )
    return row
