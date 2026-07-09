"""Parsing helpers for persisted AMD score sidecars."""

from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.core.data.path_access import (
    path_dict,
    path_get,
    path_list,
    path_require,
)
from sol_execbench.core.scoring.amd_hardware_models import (
    amd_hardware_model_from_dict,
)
from sol_execbench.core.scoring.confidence import EstimateConfidence
from sol_execbench.core.scoring.amd_sol.v2 import (
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
    if path_get(payload, "schema_version") != AMD_SOL_V2_SCHEMA_VERSION:
        return None
    aggregate_payload = path_dict(payload, "aggregate_bound")
    hardware_payload = path_dict(payload, "hardware_model")
    coverage_payload = path_dict(payload, "coverage_summary")
    if not (aggregate_payload and hardware_payload and coverage_payload):
        return None

    try:
        aggregate = AmdSolV2AggregateBound(
            status=str(
                path_require(aggregate_payload, "status", source="aggregate_bound")
            ),
            scored=bool(
                path_require(aggregate_payload, "scored", source="aggregate_bound")
            ),
            sol_bound_ms=float(
                path_require(
                    aggregate_payload, "sol_bound_ms", source="aggregate_bound"
                )
            ),
            reason=str(
                path_require(aggregate_payload, "reason", source="aggregate_bound")
            ),
            node_ids=tuple(
                str(item)
                for item in path_require(
                    aggregate_payload, "node_ids", source="aggregate_bound"
                )
            ),
        )
        coverage = AmdSolV2CoverageSummary(
            total_ops=int(path_get(coverage_payload, "total_ops", default=0)),
            supported_ops=int(path_get(coverage_payload, "supported_ops", default=0)),
            inexact_ops=int(path_get(coverage_payload, "inexact_ops", default=0)),
            unsupported_ops=int(
                path_get(coverage_payload, "unsupported_ops", default=0)
            ),
            op_family_counts={
                str(key): int(value)
                for key, value in dict(
                    path_get(coverage_payload, "op_family_counts", default={})
                ).items()
            },
            confidence_counts_by_family={
                str(family): {
                    str(key): int(value) for key, value in dict(counts).items()
                }
                for family, counts in dict(
                    path_get(
                        coverage_payload, "confidence_counts_by_family", default={}
                    )
                ).items()
            },
            worst_confidence=EstimateConfidence(
                str(
                    path_get(
                        coverage_payload,
                        "worst_confidence",
                        default="unsupported",
                    )
                )
            ),
        )
        hardware_model = amd_hardware_model_from_dict(
            hardware_payload,
            source="AMD SOL v2 sidecar hardware_model",
        )
    except (KeyError, TypeError, ValueError):
        return None

    hardware_model_ref = path_get(payload, "hardware_model_ref")
    return AmdSolBoundV2Artifact(
        definition=str(path_get(payload, "definition", default="")),
        workload_uuid=str(path_get(payload, "workload_uuid", default="")),
        hardware_model_ref=(
            str(hardware_model_ref) if hardware_model_ref is not None else None
        ),
        hardware_model=hardware_model,
        bound_graph={},
        operator_work_estimates=(),
        op_bounds=(),
        aggregate_bound=aggregate,
        warnings=tuple(str(item) for item in path_list(payload, "warnings")),
        coverage_summary=coverage,
    )


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
