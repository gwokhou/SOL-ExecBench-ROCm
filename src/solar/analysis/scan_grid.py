# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded grid construction and explicit hypothetical profile transformations."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

from solar.analysis.scan_contracts import (
    AxisKind,
    AxisValue,
    PointStatus,
    ScanDefinition,
    ScanPoint,
)
from solar.artifacts import stable_json_checksum
from solar.rocm.architecture import ArchitectureProfile


def plan_grid(
    request: ScanDefinition, profile: ArchitectureProfile
) -> tuple[ScanPoint, ...]:
    """Check actual cardinality and baseline coordinates before allocating points."""
    cardinality = 1
    for axis in request.axes:
        cardinality *= len(axis.values)
        if cardinality > request.budget.max_points:
            raise ValueError("scan grid exceeds point budget")
        if axis.kind is AxisKind.RESOURCE_SCALE:
            if set(axis.resources) - set(profile.resource_limits):
                raise ValueError("scan axis selects an unknown resource")
        elif axis.kind is AxisKind.CAPACITY_BYTES:
            level = next(
                (
                    item
                    for item in profile.memory_hierarchy
                    if item.name == axis.memory_level
                ),
                None,
            )
            if level is None or level.capacity_bytes != axis.values[0]:
                raise ValueError(
                    "capacity axis must begin at the known baseline cache capacity"
                )
    points = []
    for index, values in enumerate(
        product(*(axis.values for axis in request.axes))
    ):
        coordinates = {
            axis.key: value
            for axis, value in zip(request.axes, values, strict=True)
        }
        points.append(
            ScanPoint(
                index=index,
                point_id=f"p{index:04d}-{stable_json_checksum(coordinates)[:12]}",
                coordinates=coordinates,
                status=PointStatus.NOT_RUN,
                reason="not_started",
            )
        )
    return tuple(points)


def profile_at_point(
    baseline: ArchitectureProfile,
    request: ScanDefinition,
    coordinates: dict[str, AxisValue],
) -> ArchitectureProfile:
    """Transform only the selected resource rates, DRAM bandwidth or cache bytes."""
    profile = baseline
    for axis in request.axes:
        value = coordinates[axis.key]
        if axis.kind is AxisKind.RESOURCE_SCALE:
            limits = {
                name: {
                    mode: rate * value if name in axis.resources else rate
                    for mode, rate in modes.items()
                }
                for name, modes in profile.resource_limits.items()
            }
            profile = replace(profile, resource_limits=limits)
        elif axis.kind is AxisKind.BANDWIDTH_SCALE:
            bandwidth = profile.memory_bandwidth_bytes_per_second * value
            profile = replace(
                profile,
                memory_bandwidth_bytes_per_second=bandwidth,
                memory_hierarchy=tuple(
                    replace(level, bandwidth_bytes_per_second=bandwidth)
                    if level.name == "vram"
                    else level
                    for level in profile.memory_hierarchy
                ),
            )
        else:
            hierarchy = tuple(
                replace(level, capacity_bytes=int(value))
                if level.name == axis.memory_level
                else level
                for level in profile.memory_hierarchy
            )
            profile = replace(
                profile,
                memory_hierarchy=hierarchy,
                l2_bytes=next(
                    (
                        level.capacity_bytes or 0
                        for level in hierarchy
                        if level.name == "l2"
                    ),
                    profile.l2_bytes,
                ),
                last_level_cache_bytes=max(
                    (
                        level.capacity_bytes or 0
                        for level in hierarchy
                        if level.name != "vram"
                    ),
                    default=0,
                ),
            )
    profile.validate()
    return profile
