# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Duration-free dispatch topology for controlled concurrent replay."""

from __future__ import annotations

from collections.abc import Sequence

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.models import (
    DispatchEvidence,
    DispatchScheduleEdge,
    PerformanceScheduleEvidence,
)


def build_schedule_evidence(
    dispatches: Sequence[DispatchEvidence],
    *,
    scope_verified: bool,
) -> PerformanceScheduleEvidence | None:
    """Build lane dependencies; cross-lane timestamps are observations only."""
    if not dispatches:
        return None
    reasons: list[str] = []
    if not scope_verified:
        reasons.append("controlled_replay_scope_unverified")
    if (
        len(
            {(item.workload_uuid, item.candidate_sha256) for item in dispatches}
        )
        != 1
    ):
        reasons.append("dispatch_workload_identity_conflict")
        scope_verified = False
    if any(
        dispatch.queue_id is None and dispatch.stream_id is None
        for dispatch in dispatches
    ):
        reasons.append("dispatch_lane_identity_unverified")
    concurrent = len({_dispatch_lane(item) for item in dispatches}) > 1
    if concurrent:
        reasons.append("cross_lane_event_dependencies_unverified")
    if len(dispatches) > 1 and any(
        item.start_timestamp_ns is None for item in dispatches
    ):
        reasons.append("dispatch_lane_order_unverified")
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
    reasons = list(dict.fromkeys(reasons))
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
        edges=edges
        if scope_verified and "dispatch_lane_order_unverified" not in reasons
        else [],
        reason_codes=reasons,
    )


def schedule_predecessor_indices(
    dispatches: Sequence[DispatchEvidence],
) -> dict[int, set[int]]:
    """Return a deterministic acyclic predecessor map."""
    ordered = sorted(
        range(len(dispatches)),
        key=lambda index: (
            dispatches[index].start_timestamp_ns
            if dispatches[index].start_timestamp_ns is not None
            else -1,
            dispatches[index].dispatch_id,
        ),
    )
    predecessors = {index: set() for index in range(len(dispatches))}
    last_by_lane: dict[tuple[str | None, str | None], int] = {}
    for index in ordered:
        dispatch = dispatches[index]
        lane = _dispatch_lane(dispatch)
        if lane == (None, None):
            continue
        if lane in last_by_lane:
            predecessors[index].add(last_by_lane[lane])
        last_by_lane[lane] = index
    return predecessors


def _dispatch_lane(
    dispatch: DispatchEvidence,
) -> tuple[str | None, str | None]:
    return dispatch.queue_id, dispatch.stream_id


__all__ = ["build_schedule_evidence", "schedule_predecessor_indices"]
