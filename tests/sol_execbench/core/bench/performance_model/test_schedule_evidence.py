# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Causal schedule evidence is narrower than observed completion order."""

from itertools import permutations

import pytest

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.models import DispatchEvidence
from sol_execbench.core.bench.performance_model.schedule_evidence import (
    build_schedule_evidence,
    schedule_predecessor_indices,
)


def dispatch(identifier, queue, start):
    return DispatchEvidence(
        workload_uuid="workload",
        candidate_sha256="a" * 64,
        dispatch_id=identifier,
        queue_id=queue,
        kernel_symbol="kernel",
        grid=(1, 1, 1),
        workgroup=(1, 1, 1),
        iteration_ordinal=0,
        start_timestamp_ns=start,
        end_timestamp_ns=start + 1,
    )


def test_cross_lane_temporal_order_is_not_a_dependency():
    evidence = build_schedule_evidence(
        [dispatch("a", "0", 0), dispatch("b", "1", 5)], scope_verified=True
    )
    assert evidence is not None
    assert evidence.status is DiagnosticSidecarStatus.PARTIAL
    assert evidence.edges == []
    assert "cross_lane_event_dependencies_unverified" in evidence.reason_codes


def test_same_lane_retains_verified_order():
    evidence = build_schedule_evidence(
        [dispatch("a", "0", 0), dispatch("b", "0", 5)], scope_verified=True
    )
    assert evidence is not None
    assert evidence.status is DiagnosticSidecarStatus.AVAILABLE
    assert len(evidence.edges) == 1
    assert evidence.edges[0].reason == "same_lane"


def test_unknown_scope_cannot_emit_dependencies():
    evidence = build_schedule_evidence(
        [dispatch("a", "0", 0), dispatch("b", "0", 5)], scope_verified=False
    )
    assert evidence is not None
    assert evidence.status is DiagnosticSidecarStatus.PARTIAL
    assert evidence.edges == []


@pytest.mark.parametrize(
    ("first_end", "updates", "reason"),
    [
        (1, {"start_timestamp_ns": 0}, "same_lane_dispatch_overlap"),
        (
            0,
            {"start_timestamp_ns": 0, "end_timestamp_ns": 0},
            "dispatch_lane_order_unverified",
        ),
        (1, {"start_timestamp_ns": None}, "dispatch_lane_order_unverified"),
        (1, {"end_timestamp_ns": None}, "dispatch_lane_order_unverified"),
        (1, {"workload_uuid": "other"}, "dispatch_workload_identity_conflict"),
    ],
    ids=[
        "overlap",
        "tied-start",
        "missing-start",
        "missing-end",
        "mixed-workload",
    ],
)
def test_ambiguous_lane_cannot_publish_precedence(first_end, updates, reason):
    first = dispatch("z", "0", 0).model_copy(
        update={"end_timestamp_ns": first_end}
    )
    second = dispatch("a", "0", 5).model_copy(update=updates)
    evidence = build_schedule_evidence([first, second], scope_verified=True)
    assert evidence is not None
    assert evidence.status is DiagnosticSidecarStatus.PARTIAL
    assert evidence.edges == []
    assert reason in evidence.reason_codes
    assert schedule_predecessor_indices([first, second]) == {0: set(), 1: set()}


@pytest.mark.parametrize("lane_field", ["queue_id", "stream_id"])
def test_multilane_order_keeps_only_verified_local_edges(lane_field):
    records = [
        dispatch("a", None, 0).model_copy(update={lane_field: "0"}),
        dispatch("b", None, 3).model_copy(update={lane_field: "1"}),
        dispatch("c", None, 6).model_copy(update={lane_field: "0"}),
    ]
    for order in permutations(records):
        evidence = build_schedule_evidence(order, scope_verified=True)
        assert evidence is not None
        assert evidence.status is DiagnosticSidecarStatus.PARTIAL
        assert {
            (edge.predecessor_dispatch_id, edge.successor_dispatch_id)
            for edge in evidence.edges
        } == {("a", "c")}
        assert evidence.reason_codes == [
            "cross_lane_event_dependencies_unverified"
        ]


def test_unknown_lane_cannot_publish_a_dependency():
    records = [dispatch("a", None, 0), dispatch("b", None, 5)]
    evidence = build_schedule_evidence(records, scope_verified=True)
    assert evidence is not None
    assert evidence.status is DiagnosticSidecarStatus.PARTIAL
    assert evidence.edges == []
    assert "dispatch_lane_identity_unverified" in evidence.reason_codes
    assert schedule_predecessor_indices(records) == {0: set(), 1: set()}


def test_verified_event_links_publish_cross_lane_edges():
    records = [dispatch("a", "0", 0), dispatch("b", "1", 5)]
    evidence = build_schedule_evidence(
        records,
        scope_verified=True,
        event_links=[("a", "b")],
        cross_lane_verified=True,
    )
    assert evidence is not None
    assert evidence.status is DiagnosticSidecarStatus.AVAILABLE
    assert not evidence.reason_codes
    assert evidence.cross_lane_event_dependencies == "verified"
    assert [
        (edge.predecessor_dispatch_id, edge.successor_dispatch_id, edge.reason)
        for edge in evidence.edges
    ] == [("a", "b", "event_wait")]


def test_verified_link_with_unknown_endpoint_is_rejected():
    records = [dispatch("a", "0", 0), dispatch("b", "1", 5)]
    with pytest.raises(ValueError, match="unknown dispatch"):
        build_schedule_evidence(
            records,
            scope_verified=True,
            event_links=[("a", "unknown")],
            cross_lane_verified=True,
        )


def test_verified_links_still_require_scope_verification():
    records = [dispatch("a", "0", 0), dispatch("b", "1", 5)]
    evidence = build_schedule_evidence(
        records,
        scope_verified=False,
        event_links=[("a", "b")],
        cross_lane_verified=True,
    )
    assert evidence is not None
    assert evidence.status is DiagnosticSidecarStatus.PARTIAL
    assert evidence.edges == []
    assert evidence.cross_lane_event_dependencies == "unverified"
    assert "controlled_replay_scope_unverified" in evidence.reason_codes


def test_single_lane_evidence_records_absent_cross_lane_state():
    evidence = build_schedule_evidence(
        [dispatch("a", "0", 0), dispatch("b", "0", 5)], scope_verified=True
    )
    assert evidence is not None
    assert evidence.status is DiagnosticSidecarStatus.AVAILABLE
    assert evidence.cross_lane_event_dependencies == "absent"
