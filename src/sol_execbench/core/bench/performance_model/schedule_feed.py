# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""L2 M-d production feed: the scope-archive schedule identity bridge.

Loads one qualified hip_torch_scope archive, extracts verified cross-lane
event edges whose endpoints are both runtime dispatches, and converts the
scope's integer dispatch identifiers into the counter-evidence string
space. Fail-closed throughout: any unresolved identity, incomplete
schedule, or damaged artifact binding yields reasons instead of a partial
feed. The output is diagnostic-only; ``performance_authority`` stays false.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from solar.artifacts import sha256_file

SCOPE_SCHEMA_VERSION = "sol_execbench.hip_torch_scope.v2"


@dataclass(frozen=True, slots=True)
class ScopeScheduleFeed:
    """One archive's verified schedule contribution, or its refusal."""

    cross_lane_verified: bool
    event_edges: tuple[tuple[str, str], ...]
    workload_uuid: str | None
    reasons: tuple[str, ...]

    @property
    def fed(self) -> bool:
        """Whether this feed supplies verified cross-lane edges."""
        return self.cross_lane_verified and bool(self.event_edges)


def load_scope_schedule_feed(
    scope_path: Path,
    known_dispatch_ids: Mapping[str, frozenset[str]],
) -> ScopeScheduleFeed:
    """Bridge one scope archive into the counter dispatch-id space.

    ``known_dispatch_ids`` maps each sidecar workload to its counter
    dispatch ids; the feed binds to the workload whose identity matches
    (archive uuid when present, otherwise the unique covering set).
    """
    scope_path = Path(scope_path).resolve()
    try:
        document = json.loads((scope_path / "scope.json").read_text())
    except (OSError, ValueError) as error:
        return _refused(f"scope_archive_unreadable:{type(error).__name__}")
    if document.get("schema_version") != SCOPE_SCHEMA_VERSION:
        return _refused("scope_schema_unsupported")
    binding = document.get("artifact_sha256")
    if not isinstance(binding, dict):
        return _refused("scope_artifact_binding_missing")
    for name, digest in sorted(binding.items()):
        sibling = scope_path / str(name)
        if not sibling.is_file() or sha256_file(sibling) != digest:
            return _refused("scope_artifact_binding_mismatch")
    schedule = document.get("schedule") or {}
    if not schedule.get("schedule_complete"):
        return _refused("scope_schedule_incomplete")
    workload_uuid = _archive_workload_uuid(scope_path)
    edges: list[tuple[str, str]] = []
    dispatches = {
        item.get("dispatch_id"): item
        for item in document.get("dispatches") or []
        if isinstance(item, Mapping)
    }
    for link in document.get("links") or []:
        predecessor = link.get("predecessor_dispatch_id")
        successor = link.get("successor_dispatch_id")
        if predecessor is None or successor is None:
            continue
        lanes = (
            _lane(dispatches.get(predecessor)),
            _lane(dispatches.get(successor)),
        )
        if lanes[0] is None or lanes[1] is None or lanes[0] == lanes[1]:
            continue
        edges.append((str(predecessor), str(successor)))
    if not edges:
        return ScopeScheduleFeed(
            False,
            (),
            workload_uuid,
            ("scope_event_links_not_dispatch_bounded",),
        )
    unresolved = sorted(
        {
            endpoint
            for edge in edges
            for endpoint in edge
            if not any(endpoint in ids for ids in known_dispatch_ids.values())
        }
    )
    if unresolved:
        return _refused(f"scope_dispatch_identity_unresolved:{unresolved[:4]}")
    covering = [
        uuid
        for uuid, ids in known_dispatch_ids.items()
        if all(endpoint in ids for edge in edges for endpoint in edge)
    ]
    if workload_uuid is not None:
        if workload_uuid not in covering:
            return _refused("scope_workload_unresolved")
        bound = workload_uuid
    else:
        if len(covering) != 1:
            return _refused("scope_workload_ambiguous")
        bound = covering[0]
    return ScopeScheduleFeed(
        True,
        tuple(edges),
        bound,
        ("schedule_feed_edges_verified",),
    )


def _archive_workload_uuid(scope_path: Path) -> str | None:
    """Read the archive's workload identity when the replay recorded one."""
    sibling = scope_path / "workload.json"
    try:
        document = json.loads(sibling.read_text())
    except OSError, ValueError:
        return None
    value = (
        document.get("workload_uuid") if isinstance(document, Mapping) else None
    )
    return value if isinstance(value, str) and value else None


def _lane(
    dispatch: Mapping[str, object] | None,
) -> tuple[object, object] | None:
    """Return the lane identity (queue, stream) of one scope dispatch."""
    if not isinstance(dispatch, Mapping):
        return None
    queue = dispatch.get("queue_id")
    stream = dispatch.get("stream_id", dispatch.get("stream_handle"))
    if queue is None and stream is None:
        return None
    return (queue, stream)


def _refused(reason: str) -> ScopeScheduleFeed:
    return ScopeScheduleFeed(False, (), None, (reason,))


__all__ = [
    "SCOPE_SCHEMA_VERSION",
    "ScopeScheduleFeed",
    "load_scope_schedule_feed",
]
