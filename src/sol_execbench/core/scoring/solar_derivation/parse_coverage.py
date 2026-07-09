# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Coverage and aggregate status parsing for SOLAR derivation sidecars."""

from __future__ import annotations

from typing import Any

from sol_execbench.core.scoring.parsing_utils import (
    ensure_dict as _ensure_dict,
    parse_list as _parse_list,
    parse_optional_str as _parse_optional_str,
    parse_str as _parse_str,
)
from sol_execbench.core.scoring.solar_derivation.coverage_models import (
    SolarAggregateStatus,
    SolarCoveragePattern,
    SolarCoverageSourceRef,
    SolarCoverageSummary,
    SolarFamilyCoverage,
)

from .parse_utils import (
    _parse_count_map,
    _parse_non_negative_int,
    _parse_status,
    _parse_status_count_map,
    _parse_str_tuple,
    _require_exact_keys,
)


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
