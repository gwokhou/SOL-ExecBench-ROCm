# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Top-level parser for SOLAR derivation sidecars."""

from __future__ import annotations

from typing import Any

from sol_execbench.core.scoring.parsing_utils import (
    parse_list as _parse_list,
    parse_str as _parse_str,
    parse_str_item as _parse_str_item,
)
from sol_execbench.core.scoring.solar_derivation.coverage import (
    _aggregate_status_for_groups,
    _coverage_for_groups,
)
from sol_execbench.core.scoring.solar_derivation.models import (
    SOLAR_DERIVATION_SCHEMA_VERSION,
    SolarDerivationEvidence,
)

from .parse_coverage import (
    _aggregate_status_from_dict,
    _coverage_summary_from_dict,
)
from .parse_evidence import _group_from_dict, _tensor_from_dict
from .parse_sources import _source_boundary_from_dict
from .parse_utils import _parse_dict, _require_exact_keys


def solar_derivation_from_dict(payload: dict[str, Any]) -> SolarDerivationEvidence:
    """Parse an internal SOLAR derivation evidence sidecar payload."""
    if not isinstance(payload, dict):
        raise ValueError("SOLAR derivation evidence payload must be an object")
    required_keys = {
        "schema_version",
        "derived",
        "definition",
        "workload_uuid",
        "groups",
        "tensors",
        "warnings",
        "source_boundary",
        "coverage_summary",
        "aggregate_status",
    }
    _require_exact_keys(payload, required_keys, source="SOLAR derivation evidence")
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
    coverage_summary = _coverage_summary_from_dict(
        _parse_dict(payload, "coverage_summary", source="SOLAR derivation evidence")
    )
    aggregate_status = _aggregate_status_from_dict(
        _parse_dict(payload, "aggregate_status", source="SOLAR derivation evidence")
    )
    expected_coverage = _coverage_for_groups(groups)
    if coverage_summary.to_dict() != expected_coverage.to_dict():
        raise ValueError("coverage_summary does not match semantic groups")
    expected_aggregate = _aggregate_status_for_groups(groups, warnings)
    if aggregate_status.to_dict() != expected_aggregate.to_dict():
        raise ValueError("aggregate_status does not match semantic groups and warnings")

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
