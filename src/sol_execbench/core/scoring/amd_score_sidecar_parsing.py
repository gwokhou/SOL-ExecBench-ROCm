"""Parsing helpers for persisted AMD score sidecars."""

from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.core.scoring.amd_hardware_models import (
    EstimateConfidence,
    amd_hardware_model_from_dict,
)
from sol_execbench.core.scoring.amd_sol_v2 import (
    AMD_SOL_V2_SCHEMA_VERSION,
    AmdSolBoundV2Artifact,
    AmdSolV2AggregateBound,
    AmdSolV2CoverageSummary,
)
from sol_execbench.core.scoring.solar_derivation import SolarAggregateStatus


def read_json_object(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def minimal_amd_sol_bound_v2_from_payload(
    payload: dict,
) -> AmdSolBoundV2Artifact | None:
    """Parse only score-critical fields from a persisted AMD SOL v2 sidecar."""
    if payload.get("schema_version") != AMD_SOL_V2_SCHEMA_VERSION:
        return None
    aggregate_payload = payload.get("aggregate_bound")
    hardware_payload = payload.get("hardware_model")
    coverage_payload = payload.get("coverage_summary")
    if not (
        isinstance(aggregate_payload, dict)
        and isinstance(hardware_payload, dict)
        and isinstance(coverage_payload, dict)
    ):
        return None

    try:
        aggregate = AmdSolV2AggregateBound(
            status=str(aggregate_payload["status"]),
            scored=bool(aggregate_payload["scored"]),
            sol_bound_ms=float(aggregate_payload["sol_bound_ms"]),
            reason=str(aggregate_payload["reason"]),
            node_ids=tuple(str(item) for item in aggregate_payload["node_ids"]),
        )
        coverage = AmdSolV2CoverageSummary(
            total_ops=int(coverage_payload.get("total_ops", 0)),
            supported_ops=int(coverage_payload.get("supported_ops", 0)),
            inexact_ops=int(coverage_payload.get("inexact_ops", 0)),
            unsupported_ops=int(coverage_payload.get("unsupported_ops", 0)),
            op_family_counts={
                str(key): int(value)
                for key, value in dict(
                    coverage_payload.get("op_family_counts", {})
                ).items()
            },
            confidence_counts_by_family={
                str(family): {
                    str(key): int(value) for key, value in dict(counts).items()
                }
                for family, counts in dict(
                    coverage_payload.get("confidence_counts_by_family", {})
                ).items()
            },
            worst_confidence=EstimateConfidence(
                str(coverage_payload.get("worst_confidence", "unsupported"))
            ),
        )
        hardware_model = amd_hardware_model_from_dict(
            hardware_payload,
            source="AMD SOL v2 sidecar hardware_model",
        )
    except (KeyError, TypeError, ValueError):
        return None

    return AmdSolBoundV2Artifact(
        definition=str(payload.get("definition", "")),
        workload_uuid=str(payload.get("workload_uuid", "")),
        hardware_model_ref=(
            str(payload["hardware_model_ref"])
            if payload.get("hardware_model_ref") is not None
            else None
        ),
        hardware_model=hardware_model,
        bound_graph={},
        operator_work_estimates=(),
        op_bounds=(),
        aggregate_bound=aggregate,
        warnings=tuple(str(item) for item in payload.get("warnings", [])),
        coverage_summary=coverage,
    )


def minimal_solar_aggregate_from_payload(
    payload: dict,
) -> SolarAggregateStatus | None:
    aggregate_payload = payload.get("aggregate_status")
    if not isinstance(aggregate_payload, dict):
        return None
    try:
        return SolarAggregateStatus(
            status=str(aggregate_payload["status"]),
            score_eligible=bool(aggregate_payload["score_eligible"]),
            reason=str(aggregate_payload["reason"]),
            group_ids=tuple(str(item) for item in aggregate_payload["group_ids"]),
            node_ids=tuple(str(item) for item in aggregate_payload["node_ids"]),
            warnings=tuple(str(item) for item in aggregate_payload["warnings"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
