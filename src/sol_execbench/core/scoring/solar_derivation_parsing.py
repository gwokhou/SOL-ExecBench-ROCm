# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Parsing helpers for SOLAR derivation evidence sidecars."""

from __future__ import annotations

from math import isfinite
from typing import Any

from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence
from sol_execbench.core.scoring.parsing_utils import (
    ensure_dict as _ensure_dict,
    parse_confidence as _parse_confidence,
    parse_list as _parse_list,
    parse_optional_str as _parse_optional_str,
    parse_str as _parse_str,
    parse_str_item as _parse_str_item,
)
from sol_execbench.core.scoring.solar_derivation_coverage import (
    _aggregate_status_for_groups,
    _coverage_for_groups,
)
from sol_execbench.core.scoring.solar_derivation_models import (
    SOLAR_BOUND_LIMITING_RESOURCES,
    SOLAR_DERIVATION_SCHEMA_VERSION,
    SolarAggregateStatus,
    SolarBoundEvidence,
    SolarByteEvidence,
    SolarCoveragePattern,
    SolarCoverageSourceRef,
    SolarCoverageSummary,
    SolarDerivationEvidence,
    SolarEvidenceSource,
    SolarFamilyCoverage,
    SolarFormulaEvidence,
    SolarSemanticGroupEvidence,
    SolarSubroleEvidence,
    SolarTensorEvidence,
)
from sol_execbench.core.scoring.solar_derivation_status import (
    SOLAR_DERIVATION_SOURCE_BOUNDARY_FIELDS,
    SOLAR_DERIVATION_STATUSES,
)


def solar_derivation_from_dict(payload: dict[str, Any]) -> SolarDerivationEvidence:
    """Parse an internal SOLAR derivation evidence sidecar payload."""
    if not isinstance(payload, dict):
        raise ValueError("SOLAR derivation evidence payload must be an object")
    legacy_keys = {
        "schema_version",
        "derived",
        "definition",
        "workload_uuid",
        "groups",
        "tensors",
        "warnings",
        "source_boundary",
    }
    phase51_keys = legacy_keys | {"coverage_summary", "aggregate_status"}
    raw_keys = set(payload)
    if raw_keys == legacy_keys:
        has_phase51_fields = False
    elif raw_keys == phase51_keys:
        has_phase51_fields = True
    else:
        _require_exact_keys(payload, phase51_keys, source="SOLAR derivation evidence")
        has_phase51_fields = True
    schema_version = _parse_str(
        payload, "schema_version", source="SOLAR derivation evidence"
    )
    if schema_version != SOLAR_DERIVATION_SCHEMA_VERSION:
        raise ValueError(
            "SOLAR derivation evidence has invalid schema_version "
            f"'{schema_version}', expected '{SOLAR_DERIVATION_SCHEMA_VERSION}'"
        )
    derived = payload["derived"]
    if not isinstance(derived, bool):
        raise ValueError("SOLAR derivation evidence.derived must be a boolean")

    groups = tuple(
        _group_from_dict(raw, index)
        for index, raw in enumerate(
            _parse_list(payload, "groups", source="SOLAR derivation evidence")
        )
    )
    warnings = tuple(
        _parse_str_item(item, source=f"warnings[{index}]")
        for index, item in enumerate(
            _parse_list(payload, "warnings", source="SOLAR derivation evidence")
        )
    )
    if has_phase51_fields:
        provided_coverage = _coverage_summary_from_dict(
            _parse_dict(payload, "coverage_summary", source="SOLAR derivation evidence")
        )
        provided_aggregate = _aggregate_status_from_dict(
            _parse_dict(payload, "aggregate_status", source="SOLAR derivation evidence")
        )
        expected_coverage = _coverage_for_groups(groups)
        if provided_coverage.to_dict() != expected_coverage.to_dict():
            raise ValueError("coverage_summary does not match semantic groups")
        expected_aggregate = _aggregate_status_for_groups(groups, warnings)
        if provided_aggregate.to_dict() != expected_aggregate.to_dict():
            raise ValueError(
                "aggregate_status does not match semantic groups and warnings"
            )
        coverage_summary = provided_coverage
        aggregate_status = provided_aggregate
    else:
        coverage_summary = _coverage_for_groups(groups)
        aggregate_status = _aggregate_status_for_groups(groups, warnings)

    return SolarDerivationEvidence(
        definition=_parse_str(
            payload, "definition", source="SOLAR derivation evidence"
        ),
        workload_uuid=_parse_str(
            payload, "workload_uuid", source="SOLAR derivation evidence"
        ),
        groups=groups,
        tensors=tuple(
            _tensor_from_dict(raw, index)
            for index, raw in enumerate(
                _parse_list(payload, "tensors", source="SOLAR derivation evidence")
            )
        ),
        warnings=warnings,
        source_boundary=_source_boundary_from_dict(
            _parse_dict(payload, "source_boundary", source="SOLAR derivation evidence")
        ),
        coverage_summary=coverage_summary,
        aggregate_status=aggregate_status,
        schema_version=schema_version,
        derived=derived,
    )


def _group_from_dict(payload: Any, index: int) -> SolarSemanticGroupEvidence:
    source = f"groups[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {
            "family",
            "group_id",
            "node_ids",
            "subroles",
            "confidence",
            "status",
            "required_evidence",
            "missing_evidence",
            "warning_prefixes",
            "source",
            "rationale",
            "formula_evidence",
            "byte_evidence",
            "bound_evidence",
        },
        source=source,
    )
    status = _parse_status(raw, "status", source=source)
    return SolarSemanticGroupEvidence(
        family=_parse_str(raw, "family", source=source),
        group_id=_parse_str(raw, "group_id", source=source),
        node_ids=_parse_str_tuple(raw, "node_ids", source=source),
        subroles=tuple(
            _subrole_from_dict(item, subrole_index, group_index=index)
            for subrole_index, item in enumerate(
                _parse_list(raw, "subroles", source=source)
            )
        ),
        confidence=_parse_confidence(raw, "confidence", source=source),
        status=status,
        required_evidence=_parse_str_tuple(raw, "required_evidence", source=source),
        missing_evidence=_parse_str_tuple(raw, "missing_evidence", source=source),
        warning_prefixes=_parse_str_tuple(raw, "warning_prefixes", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        rationale=_parse_str(raw, "rationale", source=source),
        formula_evidence=tuple(
            _formula_evidence_from_dict(item, evidence_index, group_index=index)
            for evidence_index, item in enumerate(
                _parse_list(raw, "formula_evidence", source=source)
            )
        ),
        byte_evidence=tuple(
            _byte_evidence_from_dict(item, evidence_index, group_index=index)
            for evidence_index, item in enumerate(
                _parse_list(raw, "byte_evidence", source=source)
            )
        ),
        bound_evidence=tuple(
            _bound_evidence_from_dict(item, evidence_index, group_index=index)
            for evidence_index, item in enumerate(
                _parse_list(raw, "bound_evidence", source=source)
            )
        ),
    )


def _formula_evidence_from_dict(
    payload: Any,
    index: int,
    *,
    group_index: int,
) -> SolarFormulaEvidence:
    source = f"groups[{group_index}].formula_evidence[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {
            "node_id",
            "family",
            "formula_kind",
            "formula",
            "formula_inputs",
            "source",
            "confidence",
            "rationale",
        },
        source=source,
    )
    return SolarFormulaEvidence(
        node_id=_parse_str(raw, "node_id", source=source),
        family=_parse_str(raw, "family", source=source),
        formula_kind=_parse_str(raw, "formula_kind", source=source),
        formula=_parse_str(raw, "formula", source=source),
        formula_inputs=_parse_object_map(raw, "formula_inputs", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        confidence=_parse_confidence(raw, "confidence", source=source),
        rationale=_parse_str(raw, "rationale", source=source),
    )


def _byte_evidence_from_dict(
    payload: Any,
    index: int,
    *,
    group_index: int,
) -> SolarByteEvidence:
    source = f"groups[{group_index}].byte_evidence[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {
            "node_id",
            "family",
            "read_bytes",
            "write_bytes",
            "intermediate_bytes",
            "movement_bytes",
            "total_bytes",
            "dtype_inputs",
            "tensor_ids",
            "source",
            "confidence",
            "rationale",
        },
        source=source,
    )
    return SolarByteEvidence(
        node_id=_parse_str(raw, "node_id", source=source),
        family=_parse_str(raw, "family", source=source),
        read_bytes=_parse_non_negative_float(raw, "read_bytes", source=source),
        write_bytes=_parse_non_negative_float(raw, "write_bytes", source=source),
        intermediate_bytes=_parse_non_negative_float(
            raw, "intermediate_bytes", source=source
        ),
        movement_bytes=_parse_non_negative_float(raw, "movement_bytes", source=source),
        total_bytes=_parse_non_negative_float(raw, "total_bytes", source=source),
        dtype_inputs=_parse_str_map(raw, "dtype_inputs", source=source),
        tensor_ids=_parse_str_tuple(raw, "tensor_ids", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        confidence=_parse_confidence(raw, "confidence", source=source),
        rationale=_parse_str(raw, "rationale", source=source),
    )


def _bound_evidence_from_dict(
    payload: Any,
    index: int,
    *,
    group_index: int,
) -> SolarBoundEvidence:
    source = f"groups[{group_index}].bound_evidence[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {
            "node_id",
            "family",
            "compute_bound_ms",
            "memory_bound_ms",
            "limiting_resource",
            "sol_bound_ms",
            "source",
            "confidence",
            "rationale",
        },
        source=source,
    )
    limiting_resource = _parse_str(raw, "limiting_resource", source=source)
    if limiting_resource not in SOLAR_BOUND_LIMITING_RESOURCES:
        valid = ", ".join(sorted(SOLAR_BOUND_LIMITING_RESOURCES))
        raise ValueError(
            f"{source}.limiting_resource has invalid value "
            f"'{limiting_resource}', expected one of: {valid}"
        )
    return SolarBoundEvidence(
        node_id=_parse_str(raw, "node_id", source=source),
        family=_parse_str(raw, "family", source=source),
        compute_bound_ms=_parse_non_negative_float(
            raw, "compute_bound_ms", source=source
        ),
        memory_bound_ms=_parse_non_negative_float(
            raw, "memory_bound_ms", source=source
        ),
        limiting_resource=limiting_resource,
        sol_bound_ms=_parse_non_negative_float(raw, "sol_bound_ms", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        confidence=_parse_confidence(raw, "confidence", source=source),
        rationale=_parse_str(raw, "rationale", source=source),
    )


def _subrole_from_dict(
    payload: Any,
    index: int,
    *,
    group_index: int,
) -> SolarSubroleEvidence:
    source = f"groups[{group_index}].subroles[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {
            "name",
            "node_ids",
            "tensor_ids",
            "source",
            "confidence",
            "rationale",
            "missing_evidence",
        },
        source=source,
    )
    return SolarSubroleEvidence(
        name=_parse_str(raw, "name", source=source),
        node_ids=_parse_str_tuple(raw, "node_ids", source=source),
        tensor_ids=_parse_str_tuple(raw, "tensor_ids", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        confidence=_parse_confidence(raw, "confidence", source=source),
        rationale=_parse_str(raw, "rationale", source=source),
        missing_evidence=_parse_str_tuple(raw, "missing_evidence", source=source),
    )


def _tensor_from_dict(payload: Any, index: int) -> SolarTensorEvidence:
    source = f"tensors[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {
            "tensor_id",
            "name",
            "shape",
            "dtype",
            "semantic_axes",
            "source",
            "producer_node_id",
            "missing_evidence",
        },
        source=source,
    )
    return SolarTensorEvidence(
        tensor_id=_parse_str(raw, "tensor_id", source=source),
        name=_parse_str(raw, "name", source=source),
        shape=_parse_shape(raw, "shape", source=source),
        dtype=_parse_str(raw, "dtype", source=source),
        semantic_axes=_parse_str_tuple(raw, "semantic_axes", source=source),
        source=_evidence_source_from_dict(
            _parse_dict(raw, "source", source=source), source=f"{source}.source"
        ),
        producer_node_id=_parse_optional_str(raw, "producer_node_id", source=source),
        missing_evidence=_parse_str_tuple(raw, "missing_evidence", source=source),
    )


def _evidence_source_from_dict(
    payload: dict[str, Any],
    *,
    source: str,
) -> SolarEvidenceSource:
    _require_exact_keys(
        payload, {"kind", "detail", "node_id", "tensor_id"}, source=source
    )
    return SolarEvidenceSource(
        kind=_parse_str(payload, "kind", source=source),
        detail=_parse_str(payload, "detail", source=source),
        node_id=_parse_optional_str(payload, "node_id", source=source),
        tensor_id=_parse_optional_str(payload, "tensor_id", source=source),
    )


def _source_boundary_from_dict(payload: dict[str, Any]) -> dict[str, bool]:
    _require_exact_keys(
        payload,
        SOLAR_DERIVATION_SOURCE_BOUNDARY_FIELDS,
        source="source_boundary",
    )
    parsed: dict[str, bool] = {}
    for key in sorted(SOLAR_DERIVATION_SOURCE_BOUNDARY_FIELDS):
        value = payload[key]
        if not isinstance(value, bool):
            raise ValueError(f"source_boundary.{key} must be a boolean")
        parsed[key] = value
    return parsed


def _coverage_summary_from_dict(payload: dict[str, Any]) -> SolarCoverageSummary:
    source = "coverage_summary"
    _require_exact_keys(
        payload,
        {
            "family_counts",
            "status_counts",
            "families",
            "missing_patterns",
            "unsupported_patterns",
            "degraded_node_ids",
            "unsupported_node_ids",
            "estimated_node_ids",
            "provenance",
        },
        source=source,
    )
    return SolarCoverageSummary(
        family_counts=_parse_count_map(payload, "family_counts", source=source),
        status_counts=_parse_status_count_map(payload, "status_counts", source=source),
        families=tuple(
            _family_coverage_from_dict(item, index)
            for index, item in enumerate(
                _parse_list(payload, "families", source=source)
            )
        ),
        missing_patterns=tuple(
            _coverage_pattern_from_dict(item, index, field="missing_patterns")
            for index, item in enumerate(
                _parse_list(payload, "missing_patterns", source=source)
            )
        ),
        unsupported_patterns=tuple(
            _coverage_pattern_from_dict(item, index, field="unsupported_patterns")
            for index, item in enumerate(
                _parse_list(payload, "unsupported_patterns", source=source)
            )
        ),
        degraded_node_ids=_parse_str_tuple(payload, "degraded_node_ids", source=source),
        unsupported_node_ids=_parse_str_tuple(
            payload, "unsupported_node_ids", source=source
        ),
        estimated_node_ids=_parse_str_tuple(
            payload, "estimated_node_ids", source=source
        ),
        provenance=tuple(
            _coverage_source_ref_from_dict(item, index, field="provenance")
            for index, item in enumerate(
                _parse_list(payload, "provenance", source=source)
            )
        ),
    )


def _family_coverage_from_dict(payload: Any, index: int) -> SolarFamilyCoverage:
    source = f"coverage_summary.families[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(raw, {"family", "group_count", "status_counts"}, source=source)
    return SolarFamilyCoverage(
        family=_parse_str(raw, "family", source=source),
        group_count=_parse_non_negative_int(raw, "group_count", source=source),
        status_counts=_parse_status_count_map(raw, "status_counts", source=source),
    )


def _coverage_pattern_from_dict(
    payload: Any,
    index: int,
    *,
    field: str,
) -> SolarCoveragePattern:
    source = f"coverage_summary.{field}[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {"pattern", "group_ids", "node_ids", "sources"},
        source=source,
    )
    return SolarCoveragePattern(
        pattern=_parse_str(raw, "pattern", source=source),
        group_ids=_parse_str_tuple(raw, "group_ids", source=source),
        node_ids=_parse_str_tuple(raw, "node_ids", source=source),
        sources=tuple(
            _coverage_source_ref_from_dict(
                item, source_index, field=f"{field}[{index}].sources"
            )
            for source_index, item in enumerate(
                _parse_list(raw, "sources", source=source)
            )
        ),
    )


def _coverage_source_ref_from_dict(
    payload: Any,
    index: int,
    *,
    field: str,
) -> SolarCoverageSourceRef:
    source = f"coverage_summary.{field}[{index}]"
    raw = _ensure_dict(payload, source=source)
    _require_exact_keys(
        raw,
        {"group_id", "node_id", "tensor_id", "kind", "detail"},
        source=source,
    )
    return SolarCoverageSourceRef(
        group_id=_parse_str(raw, "group_id", source=source),
        node_id=_parse_optional_str(raw, "node_id", source=source),
        tensor_id=_parse_optional_str(raw, "tensor_id", source=source),
        kind=_parse_str(raw, "kind", source=source),
        detail=_parse_str(raw, "detail", source=source),
    )


def _aggregate_status_from_dict(payload: dict[str, Any]) -> SolarAggregateStatus:
    source = "aggregate_status"
    _require_exact_keys(
        payload,
        {"status", "score_eligible", "reason", "group_ids", "node_ids", "warnings"},
        source=source,
    )
    status = _parse_status(payload, "status", source=source)
    score_eligible = payload["score_eligible"]
    if not isinstance(score_eligible, bool):
        raise ValueError("aggregate_status.score_eligible must be a boolean")
    return SolarAggregateStatus(
        status=status,
        score_eligible=score_eligible,
        reason=_parse_str(payload, "reason", source=source),
        group_ids=_parse_str_tuple(payload, "group_ids", source=source),
        node_ids=_parse_str_tuple(payload, "node_ids", source=source),
        warnings=_parse_str_tuple(payload, "warnings", source=source),
    )


def _require_keys(
    payload: dict[str, Any], required: frozenset[str] | set[str], *, source: str
) -> None:
    for key in sorted(required):
        if key not in payload:
            raise ValueError(f"{source} missing required field: {key}")


def _require_exact_keys(
    payload: dict[str, Any],
    allowed: frozenset[str] | set[str],
    *,
    source: str,
) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise ValueError(f"{source} contains unknown field(s): {', '.join(unknown)}")
    _require_keys(payload, allowed, source=source)


def _parse_dict(payload: dict[str, Any], key: str, *, source: str) -> dict[str, Any]:
    return _ensure_dict(payload[key], source=f"{source}.{key}")


def _parse_str_tuple(
    payload: dict[str, Any], key: str, *, source: str
) -> tuple[str, ...]:
    return tuple(
        _parse_str_item(item, source=f"{source}.{key}[{index}]")
        for index, item in enumerate(_parse_list(payload, key, source=source))
    )


def _parse_object_map(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> dict[str, Any]:
    value = _parse_dict(payload, key, source=source)
    parsed: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"{source}.{key} keys must be strings")
        _ensure_json_value(raw_value, source=f"{source}.{key}.{raw_key}")
        parsed[raw_key] = raw_value
    return parsed


def _parse_str_map(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> dict[str, str]:
    value = _parse_dict(payload, key, source=source)
    parsed: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"{source}.{key} keys must be strings")
        if not isinstance(raw_value, str):
            raise ValueError(f"{source}.{key}.{raw_key} must be a string")
        if not raw_value:
            raise ValueError(f"{source}.{key}.{raw_key} must be non-empty")
        parsed[raw_key] = raw_value
    return parsed


def _parse_count_map(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> dict[str, int]:
    value = _parse_dict(payload, key, source=source)
    parsed: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError(f"{source}.{key} keys must be non-empty strings")
        if type(raw_value) is not int:
            raise ValueError(f"{source}.{key}.{raw_key} must be an integer")
        if raw_value < 0:
            raise ValueError(f"{source}.{key}.{raw_key} must be non-negative")
        parsed[raw_key] = raw_value
    return dict(sorted(parsed.items()))


def _parse_status_count_map(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> dict[str, int]:
    value = _parse_dict(payload, key, source=source)
    _require_exact_keys(value, SOLAR_DERIVATION_STATUSES, source=f"{source}.{key}")
    parsed: dict[str, int] = {}
    for status in sorted(SOLAR_DERIVATION_STATUSES):
        count = value[status]
        if type(count) is not int:
            raise ValueError(f"{source}.{key}.{status} must be an integer")
        if count < 0:
            raise ValueError(f"{source}.{key}.{status} must be non-negative")
        parsed[status] = count
    return parsed


def _ensure_json_value(value: object, *, source: str) -> None:
    if value is None or isinstance(value, str):
        return
    if type(value) in {int, float, bool}:
        if isinstance(value, float) and not isfinite(value):
            raise ValueError(f"{source} must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_json_value(item, source=f"{source}[{index}]")
        return
    if isinstance(value, dict):
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise ValueError(f"{source} keys must be strings")
            _ensure_json_value(raw_value, source=f"{source}.{raw_key}")
        return
    raise ValueError(f"{source} must be JSON-compatible")


def _parse_non_negative_float(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> float:
    value = payload[key]
    if isinstance(value, bool):
        raise ValueError(f"{source}.{key} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}.{key} must be numeric") from exc
    if not isfinite(parsed):
        raise ValueError(f"{source}.{key} must be finite")
    if parsed < 0.0:
        raise ValueError(f"{source}.{key} must be non-negative")
    return parsed


def _parse_non_negative_int(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> int:
    value = payload[key]
    if type(value) is not int:
        raise ValueError(f"{source}.{key} must be an integer")
    if value < 0:
        raise ValueError(f"{source}.{key} must be non-negative")
    return value


def _parse_shape(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> tuple[int, ...] | None:
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{source}.{key} must be a list or null")
    shape: list[int] = []
    for index, item in enumerate(value):
        if type(item) is not int:
            raise ValueError(f"{source}.{key}[{index}] must be an integer")
        if item < 0:
            raise ValueError(f"{source}.{key}[{index}] must be non-negative")
        shape.append(item)
    return tuple(shape)


def _parse_status(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> str:
    status = _parse_str(payload, key, source=source)
    if status not in SOLAR_DERIVATION_STATUSES:
        valid = ", ".join(sorted(SOLAR_DERIVATION_STATUSES))
        raise ValueError(
            f"{source}.{key} has invalid status '{status}', expected one of: {valid}"
        )
    return status


def _confidence_value(confidence: EstimateConfidence | str) -> str:
    if isinstance(confidence, EstimateConfidence):
        return confidence.value
    return confidence
