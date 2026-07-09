# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Parsing helpers for AMD SOL v2 bound artifacts."""

from __future__ import annotations

from typing import Any

from sol_execbench.core.scoring.amd_hardware_models import amd_hardware_model_from_dict
from sol_execbench.core.scoring.amd_sol.v2_models import (
    AGGREGATE_STATUSES,
    AMD_SOL_V2_SCHEMA_VERSION,
    AmdSolBoundV2Artifact,
    AmdSolV2AggregateBound,
    AmdSolV2CoverageSummary,
    AmdSolV2OpBound,
)
from sol_execbench.core.scoring.parsing_utils import (
    ensure_dict as _ensure_dict,
    parse_confidence as _parse_confidence,
    parse_list as _parse_list,
    parse_str as _parse_str,
    parse_str_item as _parse_str_item,
)


def amd_sol_bound_v2_from_dict(payload: dict[str, Any]) -> AmdSolBoundV2Artifact:
    """Parse an AMD SOL bound artifact v2 sidecar payload."""
    if not isinstance(payload, dict):
        raise ValueError("AMD SOL v2 artifact payload must be an object")
    _require_keys(
        payload,
        {
            "schema_version",
            "derived",
            "definition",
            "workload_uuid",
            "hardware_model_ref",
            "hardware_model",
            "bound_graph",
            "operator_work_estimates",
            "op_bounds",
            "aggregate_bound",
            "warnings",
            "coverage_summary",
        },
        source="AMD SOL v2 artifact",
    )
    schema_version = _parse_str(payload, "schema_version", source="AMD SOL v2 artifact")
    if schema_version != AMD_SOL_V2_SCHEMA_VERSION:
        raise ValueError(
            f"AMD SOL v2 artifact has invalid schema_version '{schema_version}', "
            f"expected '{AMD_SOL_V2_SCHEMA_VERSION}'"
        )

    bound_graph = _parse_dict(payload, "bound_graph", source="AMD SOL v2 artifact")
    estimate_payloads = _parse_list(
        payload, "operator_work_estimates", source="AMD SOL v2 artifact"
    )
    operator_work_estimates = tuple(
        _ensure_dict(item, source=f"operator_work_estimates[{index}]")
        for index, item in enumerate(estimate_payloads)
    )
    return AmdSolBoundV2Artifact(
        definition=_parse_str(payload, "definition", source="AMD SOL v2 artifact"),
        workload_uuid=_parse_str(
            payload, "workload_uuid", source="AMD SOL v2 artifact"
        ),
        hardware_model_ref=(
            str(payload["hardware_model_ref"])
            if payload["hardware_model_ref"] is not None
            else None
        ),
        hardware_model=amd_hardware_model_from_dict(
            _parse_dict(payload, "hardware_model", source="AMD SOL v2 artifact"),
            source="AMD SOL v2 artifact hardware_model",
        ),
        bound_graph=bound_graph,
        operator_work_estimates=operator_work_estimates,
        op_bounds=tuple(
            _op_bound_from_dict(raw, index)
            for index, raw in enumerate(
                _parse_list(payload, "op_bounds", source="AMD SOL v2 artifact")
            )
        ),
        aggregate_bound=_aggregate_from_dict(
            _parse_dict(payload, "aggregate_bound", source="AMD SOL v2 artifact")
        ),
        warnings=tuple(
            _parse_str_item(item, source=f"warnings[{index}]")
            for index, item in enumerate(
                _parse_list(payload, "warnings", source="AMD SOL v2 artifact")
            )
        ),
        coverage_summary=_coverage_from_dict(
            _parse_dict(payload, "coverage_summary", source="AMD SOL v2 artifact")
        ),
        schema_version=schema_version,
        derived=bool(payload["derived"]),
    )


def _op_bound_from_dict(payload: Any, index: int) -> AmdSolV2OpBound:
    raw = _ensure_dict(payload, source=f"op_bounds[{index}]")
    _require_keys(
        raw,
        {
            "node_id",
            "op_family",
            "op_name",
            "compute_bound_ms",
            "memory_bound_ms",
            "sol_bound_ms",
            "limiting_resource",
            "confidence",
            "rationale",
            "estimate_warnings",
        },
        source=f"op_bounds[{index}]",
    )
    return AmdSolV2OpBound(
        node_id=_parse_str(raw, "node_id", source=f"op_bounds[{index}]"),
        op_family=_parse_str(raw, "op_family", source=f"op_bounds[{index}]"),
        op_name=_parse_str(raw, "op_name", source=f"op_bounds[{index}]"),
        compute_bound_ms=_parse_float(
            raw, "compute_bound_ms", source=f"op_bounds[{index}]"
        ),
        memory_bound_ms=_parse_float(
            raw, "memory_bound_ms", source=f"op_bounds[{index}]"
        ),
        sol_bound_ms=_parse_float(raw, "sol_bound_ms", source=f"op_bounds[{index}]"),
        limiting_resource=_parse_str(
            raw, "limiting_resource", source=f"op_bounds[{index}]"
        ),
        confidence=_parse_confidence(raw, "confidence", source=f"op_bounds[{index}]"),
        rationale=_parse_str(raw, "rationale", source=f"op_bounds[{index}]"),
        estimate_warnings=tuple(
            _parse_str_item(item, source=f"op_bounds[{index}].estimate_warnings[{w_i}]")
            for w_i, item in enumerate(
                _parse_list(raw, "estimate_warnings", source=f"op_bounds[{index}]")
            )
        ),
    )


def _aggregate_from_dict(payload: dict[str, Any]) -> AmdSolV2AggregateBound:
    _require_keys(
        payload,
        {"status", "scored", "sol_bound_ms", "reason", "node_ids"},
        source="aggregate_bound",
    )
    status = _parse_str(payload, "status", source="aggregate_bound")
    if status not in AGGREGATE_STATUSES:
        raise ValueError(f"aggregate_bound has invalid status '{status}'")
    return AmdSolV2AggregateBound(
        status=status,
        scored=bool(payload["scored"]),
        sol_bound_ms=_parse_float(payload, "sol_bound_ms", source="aggregate_bound"),
        reason=_parse_str(payload, "reason", source="aggregate_bound"),
        node_ids=tuple(
            _parse_str_item(item, source=f"aggregate_bound.node_ids[{index}]")
            for index, item in enumerate(
                _parse_list(payload, "node_ids", source="aggregate_bound")
            )
        ),
    )


def _coverage_from_dict(payload: dict[str, Any]) -> AmdSolV2CoverageSummary:
    _require_keys(
        payload,
        {
            "total_ops",
            "supported_ops",
            "inexact_ops",
            "unsupported_ops",
            "op_family_counts",
            "confidence_counts_by_family",
            "worst_confidence",
        },
        source="coverage_summary",
    )
    return AmdSolV2CoverageSummary(
        total_ops=_parse_int(payload, "total_ops", source="coverage_summary"),
        supported_ops=_parse_int(payload, "supported_ops", source="coverage_summary"),
        inexact_ops=_parse_int(payload, "inexact_ops", source="coverage_summary"),
        unsupported_ops=_parse_int(
            payload, "unsupported_ops", source="coverage_summary"
        ),
        op_family_counts=_parse_int_map(
            payload, "op_family_counts", source="coverage_summary"
        ),
        confidence_counts_by_family=_parse_nested_int_map(
            payload, "confidence_counts_by_family", source="coverage_summary"
        ),
        worst_confidence=_parse_confidence(
            payload, "worst_confidence", source="coverage_summary"
        ),
    )


def _require_keys(payload: dict[str, Any], required: set[str], *, source: str) -> None:
    for key in sorted(required):
        if key not in payload:
            raise ValueError(f"{source} missing required field: {key}")


def _parse_dict(payload: dict[str, Any], key: str, *, source: str) -> dict[str, Any]:
    value = payload[key]
    return _ensure_dict(value, source=f"{source}.{key}")


def _parse_float(payload: dict[str, Any], key: str, *, source: str) -> float:
    try:
        return float(payload[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}.{key} must be numeric") from exc


def _parse_int(payload: dict[str, Any], key: str, *, source: str) -> int:
    value = payload[key]
    if not isinstance(value, int):
        raise ValueError(f"{source}.{key} must be an integer")
    return value


def _parse_int_map(payload: dict[str, Any], key: str, *, source: str) -> dict[str, int]:
    value = _parse_dict(payload, key, source=source)
    parsed: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"{source}.{key} keys must be strings")
        if not isinstance(raw_value, int):
            raise ValueError(f"{source}.{key}.{raw_key} must be an integer")
        parsed[raw_key] = raw_value
    return parsed


def _parse_nested_int_map(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> dict[str, dict[str, int]]:
    value = _parse_dict(payload, key, source=source)
    parsed: dict[str, dict[str, int]] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"{source}.{key} keys must be strings")
        if not isinstance(raw_value, dict):
            raise ValueError(f"{source}.{key}.{raw_key} must be an object")
        parsed[raw_key] = {}
        for inner_key, inner_value in raw_value.items():
            if not isinstance(inner_key, str):
                raise ValueError(f"{source}.{key}.{raw_key} keys must be strings")
            if not isinstance(inner_value, int):
                raise ValueError(
                    f"{source}.{key}.{raw_key}.{inner_key} must be an integer"
                )
            parsed[raw_key][inner_key] = inner_value
    return parsed
