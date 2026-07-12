# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict parsing for the internal fusion-aware AMD SOL base payload."""

from __future__ import annotations

import math
from typing import Any

from sol_execbench.core.platform.arch_capabilities import (
    arch_capability_budget_from_dict,
)
from sol_execbench.core.scoring.amd_hardware_models import amd_hardware_model_from_dict
from sol_execbench.core.scoring.amd_sol.fusion import FusionGroup
from sol_execbench.core.scoring.amd_sol.models import (
    AGGREGATE_STATUSES,
    AmdSolAggregateBound,
    AmdSolCoverageSummary,
    AmdSolGroupBound,
    _AmdSolBoundBase,
)
from sol_execbench.core.scoring.confidence import EstimateConfidence


def _amd_sol_bound_base_from_dict(payload: dict[str, Any]) -> _AmdSolBoundBase:
    """Parse an internal base payload after the public v4 fields are stripped."""
    _exact_keys(
        payload,
        {
            "schema_version",
            "derived",
            "definition",
            "workload_uuid",
            "hardware_model_ref",
            "hardware_model",
            "capability_budget_ref",
            "capability_budget",
            "bound_graph",
            "operator_work_estimates",
            "fusion_groups",
            "group_bounds",
            "aggregate_bound",
            "warnings",
            "coverage_summary",
        },
        "AMD SOL v3 artifact",
    )
    if payload["schema_version"] != "sol_execbench.amd_sol_bound.v3":
        raise ValueError("internal AMD SOL base has invalid schema_version")
    if not isinstance(payload["definition"], str) or not isinstance(
        payload["workload_uuid"], str
    ):
        raise ValueError(
            "AMD SOL v3 artifact definition and workload_uuid must be strings"
        )
    if not isinstance(payload["derived"], bool):
        raise ValueError("AMD SOL v3 artifact derived must be a boolean")
    if payload["hardware_model_ref"] is not None and not isinstance(
        payload["hardware_model_ref"], str
    ):
        raise ValueError(
            "AMD SOL v3 artifact hardware_model_ref must be a string or null"
        )
    if payload["capability_budget_ref"] is not None and not isinstance(
        payload["capability_budget_ref"], str
    ):
        raise ValueError(
            "AMD SOL v3 artifact capability_budget_ref must be a string or null"
        )
    if payload["capability_budget"] is not None and not isinstance(
        payload["capability_budget"], dict
    ):
        raise ValueError(
            "AMD SOL v3 artifact capability_budget must be an object or null"
        )
    if (payload["capability_budget_ref"] is None) != (
        payload["capability_budget"] is None
    ):
        raise ValueError(
            "AMD SOL v3 artifact capability_budget and capability_budget_ref "
            "must be provided together"
        )
    if not isinstance(payload["bound_graph"], dict):
        raise ValueError("AMD SOL v3 artifact bound_graph must be an object")
    if not isinstance(payload["operator_work_estimates"], list) or not all(
        isinstance(item, dict) for item in payload["operator_work_estimates"]
    ):
        raise ValueError(
            "AMD SOL v3 artifact operator_work_estimates must be object list"
        )
    if not isinstance(payload["warnings"], list) or not all(
        isinstance(item, str) for item in payload["warnings"]
    ):
        raise ValueError("AMD SOL v3 artifact warnings must be string list")
    if not isinstance(payload["hardware_model"], dict):
        raise ValueError("AMD SOL v3 artifact hardware_model must be an object")
    fusion_groups = tuple(
        _fusion_group_from_dict(item, index)
        for index, item in enumerate(_object_list(payload, "fusion_groups"))
    )
    group_bounds = tuple(
        _group_bound_from_dict(item, index)
        for index, item in enumerate(_object_list(payload, "group_bounds"))
    )
    _validate_group_partition(fusion_groups, group_bounds)
    aggregate_payload = _object(payload, "aggregate_bound")
    _exact_keys(
        aggregate_payload,
        {"status", "scored", "sol_bound_ms", "reason", "node_ids"},
        "aggregate_bound",
    )
    coverage_payload = _object(payload, "coverage_summary")
    _exact_keys(
        coverage_payload,
        {
            "total_ops",
            "supported_ops",
            "inexact_ops",
            "unsupported_ops",
            "op_family_counts",
            "confidence_counts_by_family",
            "worst_confidence",
        },
        "coverage_summary",
    )
    hardware_model = amd_hardware_model_from_dict(
        payload["hardware_model"], source="AMD SOL v3 artifact hardware_model"
    )
    capability_budget = (
        arch_capability_budget_from_dict(
            payload["capability_budget"],
            source="AMD SOL v3 artifact capability_budget",
            expected_architecture=hardware_model.architecture,
        )
        if payload["capability_budget"] is not None
        else None
    )
    return _AmdSolBoundBase(
        definition=payload["definition"],
        workload_uuid=payload["workload_uuid"],
        hardware_model_ref=payload["hardware_model_ref"],
        hardware_model=hardware_model,
        capability_budget_ref=payload["capability_budget_ref"],
        capability_budget=capability_budget,
        bound_graph=dict(payload["bound_graph"]),
        operator_work_estimates=tuple(
            dict(item) for item in payload["operator_work_estimates"]
        ),
        fusion_groups=fusion_groups,
        group_bounds=group_bounds,
        aggregate_bound=_aggregate_from_dict(aggregate_payload),
        warnings=tuple(payload["warnings"]),
        coverage_summary=_coverage_from_dict(coverage_payload),
        derived=payload["derived"],
    )


def _fusion_group_from_dict(payload: dict[str, Any], index: int) -> FusionGroup:
    source = f"fusion_groups[{index}]"
    _exact_keys(
        payload,
        {
            "group_id",
            "pattern_id",
            "pattern_version",
            "node_ids",
            "external_input_tensor_ids",
            "external_output_tensor_ids",
            "internal_tensor_ids",
            "flops",
            "external_read_bytes",
            "external_write_bytes",
            "external_bytes",
            "eliminated_intermediate_bytes",
            "required_lds_bytes",
            "confidence",
            "warnings",
        },
        source,
    )
    group = FusionGroup(
        group_id=_string(payload, "group_id", source),
        pattern_id=_string(payload, "pattern_id", source),
        pattern_version=_positive_int(payload, "pattern_version", source),
        node_ids=_string_tuple(payload, "node_ids", source),
        external_input_tensor_ids=_string_tuple(
            payload, "external_input_tensor_ids", source
        ),
        external_output_tensor_ids=_string_tuple(
            payload, "external_output_tensor_ids", source
        ),
        internal_tensor_ids=_string_tuple(payload, "internal_tensor_ids", source),
        flops=_nonnegative_float(payload, "flops", source),
        external_read_bytes=_nonnegative_float(payload, "external_read_bytes", source),
        external_write_bytes=_nonnegative_float(
            payload, "external_write_bytes", source
        ),
        eliminated_intermediate_bytes=_nonnegative_float(
            payload, "eliminated_intermediate_bytes", source
        ),
        required_lds_bytes=_optional_nonnegative_int(
            payload, "required_lds_bytes", source
        ),
        confidence=_confidence(payload, "confidence", source),
        warnings=_string_tuple(payload, "warnings", source),
    )
    if not group.node_ids or len(set(group.node_ids)) != len(group.node_ids):
        raise ValueError(f"{source}.node_ids must be a non-empty unique list")
    if abs(float(payload["external_bytes"]) - group.external_bytes) > 1e-12:
        raise ValueError(f"{source}.external_bytes does not match read/write bytes")
    return group


def _group_bound_from_dict(payload: dict[str, Any], index: int) -> AmdSolGroupBound:
    source = f"group_bounds[{index}]"
    _exact_keys(
        payload,
        {
            "group_id",
            "pattern_id",
            "node_ids",
            "compute_bound_ms",
            "memory_bound_ms",
            "sol_bound_ms",
            "limiting_resource",
            "confidence",
            "rationale",
            "warnings",
        },
        source,
    )
    compute = _nonnegative_float(payload, "compute_bound_ms", source)
    memory = _nonnegative_float(payload, "memory_bound_ms", source)
    bound = _nonnegative_float(payload, "sol_bound_ms", source)
    if abs(bound - max(compute, memory)) > 1e-12:
        raise ValueError(f"{source}.sol_bound_ms must equal max resource bound")
    limiting = _string(payload, "limiting_resource", source)
    if limiting not in {"compute", "memory"}:
        raise ValueError(f"{source}.limiting_resource is invalid")
    return AmdSolGroupBound(
        group_id=_string(payload, "group_id", source),
        pattern_id=_string(payload, "pattern_id", source),
        node_ids=_string_tuple(payload, "node_ids", source),
        compute_bound_ms=compute,
        memory_bound_ms=memory,
        sol_bound_ms=bound,
        limiting_resource=limiting,
        confidence=_confidence(payload, "confidence", source),
        rationale=_string(payload, "rationale", source),
        warnings=_string_tuple(payload, "warnings", source),
    )


def _aggregate_from_dict(payload: dict[str, Any]) -> AmdSolAggregateBound:
    status = _string(payload, "status", "aggregate_bound")
    if status not in AGGREGATE_STATUSES:
        raise ValueError("aggregate_bound.status is invalid")
    scored = payload["scored"]
    if not isinstance(scored, bool):
        raise ValueError("aggregate_bound.scored must be a boolean")
    return AmdSolAggregateBound(
        status=status,
        scored=scored,
        sol_bound_ms=_nonnegative_float(payload, "sol_bound_ms", "aggregate_bound"),
        reason=_string(payload, "reason", "aggregate_bound"),
        node_ids=_string_tuple(payload, "node_ids", "aggregate_bound"),
    )


def _coverage_from_dict(payload: dict[str, Any]) -> AmdSolCoverageSummary:
    counts = _int_map(payload["op_family_counts"], "coverage_summary.op_family_counts")
    nested = payload["confidence_counts_by_family"]
    if not isinstance(nested, dict):
        raise ValueError(
            "coverage_summary.confidence_counts_by_family must be an object"
        )
    confidence_counts = {
        str(family): _int_map(
            value, f"coverage_summary.confidence_counts_by_family.{family}"
        )
        for family, value in nested.items()
    }
    return AmdSolCoverageSummary(
        total_ops=_nonnegative_int(payload, "total_ops", "coverage_summary"),
        supported_ops=_nonnegative_int(payload, "supported_ops", "coverage_summary"),
        inexact_ops=_nonnegative_int(payload, "inexact_ops", "coverage_summary"),
        unsupported_ops=_nonnegative_int(
            payload, "unsupported_ops", "coverage_summary"
        ),
        op_family_counts=counts,
        confidence_counts_by_family=confidence_counts,
        worst_confidence=_confidence(payload, "worst_confidence", "coverage_summary"),
    )


def _validate_group_partition(
    groups: tuple[FusionGroup, ...], bounds: tuple[AmdSolGroupBound, ...]
) -> None:
    if len({group.group_id for group in groups}) != len(groups):
        raise ValueError("fusion_groups contains duplicate group_id")
    if len({node_id for group in groups for node_id in group.node_ids}) != sum(
        len(group.node_ids) for group in groups
    ):
        raise ValueError("fusion_groups assigns a node more than once")
    if {group.group_id for group in groups} != {bound.group_id for bound in bounds}:
        raise ValueError("fusion_groups and group_bounds must have matching group IDs")
    if len({bound.group_id for bound in bounds}) != len(bounds):
        raise ValueError("group_bounds contains duplicate group_id")
    groups_by_id = {group.group_id: group for group in groups}
    for bound in bounds:
        group = groups_by_id[bound.group_id]
        if bound.node_ids != group.node_ids:
            raise ValueError("group_bounds node_ids must match fusion group")
        if bound.pattern_id != group.pattern_id:
            raise ValueError("group_bounds pattern_id must match fusion group")


def _exact_keys(payload: object, expected: set[str], source: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must be an object")
    unknown = sorted(set(payload) - expected)
    missing = sorted(expected - set(payload))
    if unknown or missing:
        details = [
            *(f"unknown field: {key}" for key in unknown),
            *(f"missing required field: {key}" for key in missing),
        ]
        raise ValueError(f"{source} has " + ", ".join(details))


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload[key]
    if not isinstance(value, dict):
        raise ValueError(f"AMD SOL v3 artifact {key} must be an object")
    return value


def _object_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload[key]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"AMD SOL v3 artifact {key} must be an object list")
    return value


def _string(payload: dict[str, Any], key: str, source: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f"{source}.{key} must be a string")
    return value


def _string_tuple(payload: dict[str, Any], key: str, source: str) -> tuple[str, ...]:
    value = payload[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{source}.{key} must be a string list")
    return tuple(value)


def _positive_int(payload: dict[str, Any], key: str, source: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{source}.{key} must be a positive integer")
    return value


def _nonnegative_int(payload: dict[str, Any], key: str, source: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{source}.{key} must be a non-negative integer")
    return value


def _optional_nonnegative_int(
    payload: dict[str, Any], key: str, source: str
) -> int | None:
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{source}.{key} must be a non-negative integer or null")
    return value


def _int_map(value: object, source: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{source} must be an object")
    result: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, int):
            raise ValueError(f"{source} must map strings to integers")
        result[key] = item
    return result


def _nonnegative_float(payload: dict[str, Any], key: str, source: str) -> float:
    try:
        value = float(payload[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}.{key} must be numeric") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{source}.{key} must be finite and non-negative")
    return value


def _confidence(payload: dict[str, Any], key: str, source: str) -> EstimateConfidence:
    try:
        return EstimateConfidence(_string(payload, key, source))
    except ValueError as exc:
        raise ValueError(f"{source}.{key} is invalid") from exc
