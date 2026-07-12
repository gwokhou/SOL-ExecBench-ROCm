"""Parsing helpers for persisted AMD score sidecars."""

from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.core.data.path_access import (
    path_dict,
    path_get,
    path_require,
)
from sol_execbench.core.scoring.amd_sol import (
    AMD_SOL_SCHEMA_VERSION,
    AmdSolBoundArtifact,
    amd_sol_bound_from_dict,
)
from sol_execbench.core.scoring.solar_derivation import SolarAggregateStatus


def read_json_object(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def amd_sol_bound_from_payload(payload: dict) -> AmdSolBoundArtifact | None:
    """Parse a persisted sidecar, rejecting old or malformed contracts."""
    version = path_get(payload, "schema_version")
    try:
        if version == AMD_SOL_SCHEMA_VERSION:
            return amd_sol_bound_from_dict(payload)
        return None
    except (KeyError, TypeError, ValueError):
        return None


def minimal_solar_aggregate_from_payload(
    payload: dict,
) -> SolarAggregateStatus | None:
    aggregate_payload = path_dict(payload, "aggregate_status")
    if not aggregate_payload:
        return None
    try:
        return SolarAggregateStatus(
            status=str(
                path_require(aggregate_payload, "status", source="aggregate_status")
            ),
            score_eligible=bool(
                path_require(
                    aggregate_payload, "score_eligible", source="aggregate_status"
                )
            ),
            reason=str(
                path_require(aggregate_payload, "reason", source="aggregate_status")
            ),
            group_ids=tuple(
                str(item)
                for item in path_require(
                    aggregate_payload, "group_ids", source="aggregate_status"
                )
            ),
            node_ids=tuple(
                str(item)
                for item in path_require(
                    aggregate_payload, "node_ids", source="aggregate_status"
                )
            ),
            warnings=tuple(
                str(item)
                for item in path_require(
                    aggregate_payload, "warnings", source="aggregate_status"
                )
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None
