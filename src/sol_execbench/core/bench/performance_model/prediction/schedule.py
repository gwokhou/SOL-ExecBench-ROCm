# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Dispatch scheduling and overlap prediction."""

from __future__ import annotations

from collections.abc import Sequence

from sol_execbench.core.bench.performance_model.models import (
    DiagnosticCalibrationProfile,
    DispatchEvidence,
    PredictionComponent,
)
from sol_execbench.core.bench.performance_model.prediction.calibration import (
    _interpolated_overlap_cell,
)
from sol_execbench.core.bench.performance_model.prediction.contracts import (
    _PredictionUnavailableError,
)
from sol_execbench.core.bench.performance_model.schedule_evidence import (
    dispatch_identity_reason,
    dispatch_intervals_overlap,
    dispatch_lane,
    schedule_predecessor_indices,
)

_F16_WMMA_FLOPS = 2.0 * 16.0 * 16.0 * 16.0
_DEFAULT_WAVE_SIZE = 32.0
_FP32_BYTES = 4.0


def _combine_components(
    components: Sequence[PredictionComponent],
) -> tuple[float, float, float]:
    dispatch = [item for item in components if item.name == "dispatch"]
    penalties = [item for item in components if item.name.endswith("_penalty")]
    resources = [
        item
        for item in components
        if item.name != "dispatch" and not item.name.endswith("_penalty")
    ]
    return tuple(
        sum(getattr(item, attribute) for item in dispatch)
        + max(
            (getattr(item, attribute) for item in resources),
            default=0.0,
        )
        + sum(getattr(item, attribute) for item in penalties)
        for attribute in ("time_ms", "lower_ms", "upper_ms")
    )


def _has_overlap(dispatches: Sequence[DispatchEvidence]) -> bool:
    return any(
        dispatch_intervals_overlap(left, right)
        for index, left in enumerate(dispatches)
        for right in dispatches[index + 1 :]
    )


def _concurrency_reason(
    dispatches: Sequence[DispatchEvidence],
    *,
    cross_lane_verified: bool = False,
) -> str | None:
    if identity_reason := dispatch_identity_reason(dispatches):
        return identity_reason
    if any(
        dispatch.queue_id is None and dispatch.stream_id is None
        for dispatch in dispatches
    ):
        return "dispatch_queue_identity_unverified"
    identities = {dispatch_lane(dispatch) for dispatch in dispatches}
    if any(
        dispatch_lane(left) == dispatch_lane(right)
        and dispatch_intervals_overlap(left, right)
        for index, left in enumerate(dispatches)
        for right in dispatches[index + 1 :]
    ):
        return "same_lane_dispatch_overlap"
    concurrent = len(identities) != 1 or _has_overlap(dispatches)
    if concurrent and any(
        dispatch.start_timestamp_ns is None or dispatch.end_timestamp_ns is None
        for dispatch in dispatches
    ):
        return "dispatch_schedule_topology_unverified"
    if len(identities) != 1 and not cross_lane_verified:
        return "cross_lane_event_dependencies_unverified"
    return None


def _schedule_estimate(
    dispatches: Sequence[DispatchEvidence],
    groups: Sequence[Sequence[PredictionComponent]],
    calibration: DiagnosticCalibrationProfile,
    event_edges: Sequence[tuple[str, str]] = (),
) -> tuple[float, float, float]:
    estimates = [_combine_components(group) for group in groups]
    if (
        not _has_overlap(dispatches)
        and len({dispatch_lane(dispatch) for dispatch in dispatches}) == 1
        and not event_edges
    ):
        return tuple(sum(values) for values in zip(*estimates, strict=True))
    ordered = sorted(
        range(len(dispatches)),
        key=lambda index: (
            dispatches[index].start_timestamp_ns
            if dispatches[index].start_timestamp_ns is not None
            else -1,
            dispatches[index].dispatch_id,
        ),
    )
    predecessors = schedule_predecessor_indices(dispatches)
    if event_edges:
        predecessors = _with_event_predecessors(
            dispatches, predecessors, event_edges
        )
    adjusted = [
        _overlap_adjusted_estimate(
            index,
            dispatches,
            groups[index],
            estimates[index],
            calibration,
        )
        for index in range(len(dispatches))
    ]
    finishes: dict[int, tuple[float, float, float]] = {}
    for index in ordered:
        prefix = tuple(
            max(
                (finishes[parent][position] for parent in predecessors[index]),
                default=0.0,
            )
            for position in range(3)
        )
        finishes[index] = (
            prefix[0] + adjusted[index][0],
            prefix[1] + adjusted[index][1],
            prefix[2] + adjusted[index][2],
        )
    return (
        max(finish[0] for finish in finishes.values()),
        max(finish[1] for finish in finishes.values()),
        max(finish[2] for finish in finishes.values()),
    )


def _overlap_adjusted_estimate(
    index: int,
    dispatches: Sequence[DispatchEvidence],
    components: Sequence[PredictionComponent],
    estimate: tuple[float, float, float],
    calibration: DiagnosticCalibrationProfile,
) -> tuple[float, float, float]:
    concurrent = sum(
        1
        for other, _dispatch in enumerate(dispatches)
        if dispatch_intervals_overlap(dispatches[index], dispatches[other])
    )
    if concurrent <= 1:
        return estimate
    resource_time = sum(
        component.time_ms
        for component in components
        if component.name != "dispatch"
    )
    compute_time = sum(
        component.time_ms
        for component in components
        if component.name
        in {"compute", "wmma", "reduction", "softmax_reduction"}
    )
    mix = compute_time / resource_time if resource_time > 0.0 else 0.0
    cell = _interpolated_overlap_cell(calibration, mix, concurrent)
    lower_efficiency, upper_efficiency = cell.confidence_interval
    return (
        estimate[0] / cell.value,
        estimate[1] / upper_efficiency,
        estimate[2] / lower_efficiency,
    )


def _overlaps(left: DispatchEvidence, right: DispatchEvidence) -> bool:
    values = (
        left.start_timestamp_ns,
        left.end_timestamp_ns,
        right.start_timestamp_ns,
        right.end_timestamp_ns,
    )
    if any(value is None for value in values):
        return False
    left_start = left.start_timestamp_ns
    left_end = left.end_timestamp_ns
    right_start = right.start_timestamp_ns
    right_end = right.end_timestamp_ns
    if (
        left_start is None
        or left_end is None
        or right_start is None
        or right_end is None
    ):
        return False
    return left_start < right_end and right_start < left_end


def _with_event_predecessors(
    dispatches: Sequence[DispatchEvidence],
    predecessors: dict[int, set[int]],
    event_edges: Sequence[tuple[str, str]],
) -> dict[int, set[int]]:
    """Verified event edges serialize lanes; timestamps stay observations."""
    index = {
        dispatch.dispatch_id: position
        for position, dispatch in enumerate(dispatches)
    }
    if any(
        predecessor not in index or successor not in index
        for predecessor, successor in event_edges
    ):
        raise _PredictionUnavailableError(
            "cross_lane_event_dependencies_unverified"
        )
    merged = {
        position: set(parents) for position, parents in predecessors.items()
    }
    for predecessor, successor in event_edges:
        merged[index[successor]].add(index[predecessor])
    return merged
