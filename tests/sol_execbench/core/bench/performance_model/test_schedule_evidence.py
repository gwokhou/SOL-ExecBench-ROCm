# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Causal schedule evidence is narrower than observed completion order."""

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.models import DispatchEvidence
from sol_execbench.core.bench.performance_model.schedule_evidence import (
    build_schedule_evidence,
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
