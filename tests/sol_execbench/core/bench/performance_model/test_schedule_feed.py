# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""L2 M-d schedule-feed bridge tests (spec solar-l2-md-feed)."""

from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.core.bench.performance_model.schedule_feed import (
    load_scope_schedule_feed,
)

REPO = Path(__file__).resolve().parents[5]
REAL_CAPTURE = (
    REPO / "data/outputs/solar-l2-schedule-20261106/full-captures/0-event_chain"
)


def test_real_l2_capture_feeds_dispatch_edges() -> None:
    """M1: the real event_chain archive bridges to str dispatch ids."""
    ids = {
        "_workload": frozenset(
            str(item["dispatch_id"])
            for item in json.loads((REAL_CAPTURE / "scope.json").read_text())[
                "dispatches"
            ]
        ),
    }
    feed = load_scope_schedule_feed(REAL_CAPTURE, ids)
    assert feed.fed
    assert feed.event_edges == (("11", "12"),)
    assert feed.reasons == ("schedule_feed_edges_verified",)


def test_unknown_endpoint_is_fail_closed(tmp_path: Path) -> None:
    """M1: an endpoint outside the counter space refuses the whole feed."""
    ids = {"_workload": frozenset({"1", "2"})}
    feed = load_scope_schedule_feed(REAL_CAPTURE, ids)
    assert not feed.fed
    assert feed.reasons[0].startswith("scope_dispatch_identity_unresolved")


def _copy_capture(tmp_path: Path, mutate, *, rebind: bool = True) -> Path:
    """Copy the real archive, apply a mutation, and rebind its hashes."""
    import hashlib

    target = tmp_path / "capture"
    target.mkdir(parents=True)
    document = json.loads((REAL_CAPTURE / "scope.json").read_text())
    mutate(document, target)
    # Narrow the binding to the copied sibling so mutations test one
    # thing; the mismatch test keeps the pristine hash on purpose.
    import hashlib as _hashlib

    document["artifact_sha256"] = {
        "workload.json": _hashlib.sha256(
            (REAL_CAPTURE / "workload.json").read_bytes()
        ).hexdigest(),
    }
    if not (target / "workload.json").exists():
        (target / "workload.json").write_bytes(
            (REAL_CAPTURE / "workload.json").read_bytes()
        )
    if rebind:
        document["artifact_sha256"]["workload.json"] = hashlib.sha256(
            (target / "workload.json").read_bytes()
        ).hexdigest()
    (target / "scope.json").write_text(json.dumps(document))
    return target


def test_incomplete_schedule_refuses(tmp_path: Path) -> None:
    """M1: schedule_complete=false never feeds."""

    def mutate(document, target):
        document["schedule"]["schedule_complete"] = False

    feed = load_scope_schedule_feed(
        _copy_capture(tmp_path, mutate), {"_workload": frozenset({"11", "12"})}
    )
    assert not feed.fed
    assert "scope_schedule_incomplete" in feed.reasons[0]


def test_copy_only_links_do_not_feed(tmp_path: Path) -> None:
    """M1: links without two dispatch endpoints yield no dispatch edges."""

    def mutate(document, target):
        document["links"] = [
            {
                "basis": "sdk_enqueue_order",
                "predecessor_copy_correlation_id": 5,
                "predecessor_dispatch_id": None,
                "successor_dispatch_id": 12,
            }
        ]

    feed = load_scope_schedule_feed(
        _copy_capture(tmp_path, mutate),
        {"_workload": frozenset({"7", "11", "12"})},
    )
    assert not feed.fed
    assert feed.reasons == ("scope_event_links_not_dispatch_bounded",)


def test_binding_mismatch_refuses(tmp_path: Path) -> None:
    """M1: a damaged sibling hash refuses the feed."""

    def mutate(document, target):
        (target / "workload.json").write_text("{}")

    feed = load_scope_schedule_feed(
        _copy_capture(tmp_path, mutate, rebind=False),
        {"_workload": frozenset({"11", "12"})},
    )
    assert not feed.fed
    assert feed.reasons[0].startswith("scope_artifact")


def test_builder_request_carries_optional_scope_path() -> None:
    """M2/M3: the request and CLI plumbing carry the opt-in field."""
    from sol_execbench.core.bench.performance_model.builder import (
        PerformanceDiagnosticBuildRequest,
    )

    request = PerformanceDiagnosticBuildRequest(
        evidence_manifest_path=Path("m.json"),
        solar_manifest_path=Path("s.json"),
        output_path=Path("o.json"),
    )
    assert request.schedule_scope_path is None
    request = PerformanceDiagnosticBuildRequest(
        evidence_manifest_path=Path("m.json"),
        solar_manifest_path=Path("s.json"),
        output_path=Path("o.json"),
        schedule_scope_path=Path("capture"),
    )
    assert request.schedule_scope_path == Path("capture")


def test_builder_feed_seam_default_is_empty() -> None:
    """M5: without the opt-in the feed contributes nothing at all."""
    from types import SimpleNamespace

    from sol_execbench.core.bench.performance_model.builder import (
        PerformanceDiagnosticBuildRequest,
        _load_schedule_feed,
    )

    request = PerformanceDiagnosticBuildRequest(
        evidence_manifest_path=Path("m.json"),
        solar_manifest_path=Path("s.json"),
        output_path=Path("o.json"),
    )
    feed = _load_schedule_feed(request, SimpleNamespace(dispatches=()))
    assert feed == feed.__class__(
        cross_lane_verified=False,
        event_edges=(),
        workload_uuid=None,
        reasons=(),
    )


def test_builder_feed_seam_bridges_real_archive() -> None:
    """M2: the opted-in real archive reaches the workload id space."""
    from types import SimpleNamespace

    from sol_execbench.core.bench.performance_model.builder import (
        PerformanceDiagnosticBuildRequest,
        _load_schedule_feed,
    )
    from sol_execbench.core.bench.performance_model.models import (
        DispatchEvidence,
    )

    def evidence(item: dict) -> DispatchEvidence:
        return DispatchEvidence(
            workload_uuid="w",
            candidate_sha256="0" * 64,
            dispatch_id=str(item["dispatch_id"]),
            kernel_symbol=str(item.get("kernel_name", "k")),
            grid=(1, 1, 1),
            workgroup=(1, 1, 1),
            iteration_ordinal=0,
        )

    dispatches = [
        evidence(item)
        for item in json.loads((REAL_CAPTURE / "scope.json").read_text())[
            "dispatches"
        ]
    ]
    request = PerformanceDiagnosticBuildRequest(
        evidence_manifest_path=Path("m.json"),
        solar_manifest_path=Path("s.json"),
        output_path=Path("o.json"),
        schedule_scope_path=REAL_CAPTURE,
    )
    feed = _load_schedule_feed(request, SimpleNamespace(dispatches=dispatches))
    assert feed.fed
    assert feed.event_edges == (("11", "12"),)
    assert feed.cross_lane_verified is True
