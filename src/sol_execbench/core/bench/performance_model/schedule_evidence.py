# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Duration-free dispatch topology for controlled concurrent replay."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.models import (
    CrossLaneEventDependencies,
    DispatchEvidence,
    DispatchScheduleEdge,
    PerformanceScheduleEvidence,
)


def build_schedule_evidence(
    dispatches: Sequence[DispatchEvidence],
    *,
    scope_verified: bool,
    event_links: Sequence[tuple[str, str]] = (),
    cross_lane_verified: bool = False,
) -> PerformanceScheduleEvidence | None:
    """Build lane dependencies; cross-lane timestamps are observations only.

    Verified cross-lane edges come only from an independently qualified scope
    schedule; temporal adjacency never becomes a dependency.
    """
    if not dispatches:
        return None
    reasons: list[str] = []
    if not scope_verified:
        reasons.append("controlled_replay_scope_unverified")
    if identity_reason := dispatch_identity_reason(dispatches):
        reasons.append(identity_reason)
        scope_verified = False
    if any(
        dispatch.queue_id is None and dispatch.stream_id is None
        for dispatch in dispatches
    ):
        reasons.append("dispatch_lane_identity_unverified")
    concurrent = len({dispatch_lane(item) for item in dispatches}) > 1
    if concurrent and not cross_lane_verified:
        reasons.append("cross_lane_event_dependencies_unverified")
    if order_reason := _lane_order_reason(dispatches):
        reasons.append(order_reason)
    predecessors = schedule_predecessor_indices(dispatches)
    edges = [
        DispatchScheduleEdge(
            predecessor_dispatch_id=dispatches[predecessor].dispatch_id,
            successor_dispatch_id=dispatches[successor].dispatch_id,
            reason="same_lane",
        )
        for successor, parents in predecessors.items()
        for predecessor in sorted(parents)
    ]
    if scope_verified and concurrent and cross_lane_verified:
        known = {item.dispatch_id for item in dispatches}
        if any(
            predecessor not in known or successor not in known
            for predecessor, successor in event_links
        ):
            raise ValueError("verified event link references unknown dispatch")
        edges.extend(
            DispatchScheduleEdge(
                predecessor_dispatch_id=predecessor,
                successor_dispatch_id=successor,
                reason="event_wait",
            )
            for predecessor, successor in event_links
        )
    reasons = list(dict.fromkeys(reasons))
    cross_lane = (
        (
            CrossLaneEventDependencies.VERIFIED
            if cross_lane_verified and scope_verified
            else CrossLaneEventDependencies.UNVERIFIED
        )
        if concurrent
        else CrossLaneEventDependencies.ABSENT
    )
    return PerformanceScheduleEvidence(
        status=(
            DiagnosticSidecarStatus.PARTIAL
            if reasons
            else DiagnosticSidecarStatus.AVAILABLE
        ),
        workload_uuid=dispatches[0].workload_uuid,
        candidate_sha256=dispatches[0].candidate_sha256,
        same_process=scope_verified,
        same_gpu=scope_verified,
        marker_contained=scope_verified,
        dispatch_ids=[item.dispatch_id for item in dispatches],
        edges=edges if scope_verified else [],
        cross_lane_event_dependencies=cross_lane,
        reason_codes=reasons,
    )


def schedule_predecessor_indices(
    dispatches: Sequence[DispatchEvidence],
) -> dict[int, set[int]]:
    """Return same-lane predecessors only when identity and order are known."""
    predecessors = {index: set() for index in range(len(dispatches))}
    if dispatch_identity_reason(dispatches) or _lane_order_reason(dispatches):
        return predecessors
    ordered = sorted(
        range(len(dispatches)),
        key=lambda index: (
            dispatches[index].start_timestamp_ns
            if dispatches[index].start_timestamp_ns is not None
            else -1,
            dispatches[index].dispatch_id,
        ),
    )
    last_by_lane: dict[tuple[str | None, str | None], int] = {}
    for index in ordered:
        dispatch = dispatches[index]
        lane = dispatch_lane(dispatch)
        if lane == (None, None):
            continue
        if lane in last_by_lane:
            predecessors[index].add(last_by_lane[lane])
        last_by_lane[lane] = index
    return predecessors


def dispatch_lane(
    dispatch: DispatchEvidence,
) -> tuple[str | None, str | None]:
    """Identify a lane within an already verified process and device scope."""
    return dispatch.queue_id, dispatch.stream_id


def dispatch_identity_reason(
    dispatches: Sequence[DispatchEvidence],
) -> str | None:
    """Reject a schedule combining different workloads or candidate builds."""
    identities = {
        (item.workload_uuid, item.candidate_sha256) for item in dispatches
    }
    if len(identities) > 1:
        return "dispatch_workload_identity_conflict"
    return None


def dispatch_intervals_overlap(
    left: DispatchEvidence, right: DispatchEvidence
) -> bool:
    """Detect contradictory same-lane intervals without inferring dependencies."""
    left_start, left_end = left.start_timestamp_ns, left.end_timestamp_ns
    right_start, right_end = right.start_timestamp_ns, right.end_timestamp_ns
    if (
        left_start is None
        or left_end is None
        or right_start is None
        or right_end is None
    ):
        return False
    return left_start < right_end and right_start < left_end


def _lane_order_reason(dispatches: Sequence[DispatchEvidence]) -> str | None:
    lanes: dict[tuple[str | None, str | None], list[DispatchEvidence]] = (
        defaultdict(list)
    )
    for dispatch in dispatches:
        if dispatch_lane(dispatch) != (None, None):
            lanes[dispatch_lane(dispatch)].append(dispatch)
    for lane in lanes.values():
        if len(lane) < 2:
            continue
        if any(
            item.start_timestamp_ns is None or item.end_timestamp_ns is None
            for item in lane
        ):
            return "dispatch_lane_order_unverified"
        ordered = sorted(lane, key=lambda item: item.start_timestamp_ns or 0)
        for left, right in pairwise(ordered):
            if dispatch_intervals_overlap(left, right):
                return "same_lane_dispatch_overlap"
            if left.start_timestamp_ns == right.start_timestamp_ns:
                return "dispatch_lane_order_unverified"
    return None


__all__ = [
    "build_schedule_evidence",
    "dispatch_identity_reason",
    "dispatch_intervals_overlap",
    "dispatch_lane",
    "schedule_predecessor_indices",
]
