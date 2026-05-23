# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-side loader for SOLAR derivation contract fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FIXTURE_ROOT = Path(__file__).with_name("fixtures") / "solar_derivation"

TARGET_FAMILIES = frozenset(
    {
        "attention",
        "moe",
        "convolution",
        "ssm_mamba",
        "embedding_positional",
        "linear_projection",
    }
)
REQUIRED_TOP_LEVEL = frozenset(
    {
        "case_id",
        "family",
        "fixture_class",
        "negative_category",
        "description",
        "source_kind",
        "reference",
        "workload_axes",
        "expectation",
        "scope_boundary",
    }
)
REQUIRED_EXPECTATION = frozenset(
    {
        "expected_family",
        "expected_subroles",
        "expected_confidence",
        "expected_status",
        "required_evidence",
        "missing_evidence",
        "warning_prefixes",
        "degradation_rationale",
    }
)
REQUIRED_SCOPE_BOUNDARY = frozenset(
    {
        "paper_scale_dataset",
        "hosted_leaderboard_ready",
        "nvidia_blackwell_b200_equivalence",
        "real_hardware_validation",
    }
)
VALID_FIXTURE_CLASSES = frozenset({"positive", "degraded", "unsupported", "negative"})
VALID_NEGATIVE_CATEGORIES = frozenset(
    {"dynamic", "partial", "unsupported", "taxonomy_only", "missing_metadata"}
)
VALID_CONFIDENCES = frozenset({"supported", "inexact", "unsupported"})
VALID_STATUSES = frozenset({"scored", "degraded", "unscored"})
VALID_WARNING_PREFIXES = (
    "graph_warning:",
    "estimate_warning:",
    "inexact_operator:",
    "unsupported_operator:",
    "aggregate_degraded:",
    "aggregate_unscored:",
)


def load_solar_derivation_fixtures(
    root: Path | None = None,
) -> tuple[dict[str, object], ...]:
    """Load and validate all SOLAR derivation fixture JSON files."""
    fixture_root = root or FIXTURE_ROOT
    if not fixture_root.exists():
        return ()
    fixtures: list[dict[str, object]] = []
    for path in sorted(fixture_root.glob("*.json")):
        payload = json.loads(path.read_text())
        validate_solar_derivation_fixture(payload, source=str(path))
        fixtures.append(payload)
    return tuple(fixtures)


def validate_solar_derivation_fixture(payload: Any, *, source: str) -> None:
    """Validate one SOLAR derivation fixture payload."""
    fixture = _ensure_dict(payload, source=source)
    _require_keys(fixture, REQUIRED_TOP_LEVEL, source=source)

    family = _parse_non_empty_str(fixture, "family", source=source)
    _require_value(family, TARGET_FAMILIES, source=f"{source}.family")
    fixture_class = _parse_non_empty_str(fixture, "fixture_class", source=source)
    _require_value(
        fixture_class, VALID_FIXTURE_CLASSES, source=f"{source}.fixture_class"
    )
    negative_category = fixture["negative_category"]
    if negative_category is not None:
        if not isinstance(negative_category, str) or not negative_category:
            raise ValueError(f"{source}.negative_category must be null or non-empty string")
        _require_value(
            negative_category,
            VALID_NEGATIVE_CATEGORIES,
            source=f"{source}.negative_category",
        )

    for key in ("case_id", "description", "source_kind", "reference"):
        _parse_non_empty_str(fixture, key, source=source)
    _ensure_dict(fixture["workload_axes"], source=f"{source}.workload_axes")

    expectation = _ensure_dict(fixture["expectation"], source=f"{source}.expectation")
    _require_keys(
        expectation, REQUIRED_EXPECTATION, source=f"{source}.expectation"
    )
    _validate_expectation(
        expectation,
        family=family,
        fixture_class=fixture_class,
        source=f"{source}.expectation",
    )

    scope_boundary = _ensure_dict(
        fixture["scope_boundary"], source=f"{source}.scope_boundary"
    )
    _require_keys(
        scope_boundary, REQUIRED_SCOPE_BOUNDARY, source=f"{source}.scope_boundary"
    )
    for key in sorted(REQUIRED_SCOPE_BOUNDARY):
        if not isinstance(scope_boundary[key], bool):
            raise ValueError(f"{source}.scope_boundary.{key} must be a boolean")


def _validate_expectation(
    expectation: dict[str, Any],
    *,
    family: str,
    fixture_class: str,
    source: str,
) -> None:
    expected_family = _parse_non_empty_str(
        expectation, "expected_family", source=source
    )
    if expected_family != family:
        raise ValueError(f"{source}.expected_family must match fixture family")
    confidence = _parse_non_empty_str(
        expectation, "expected_confidence", source=source
    )
    _require_value(confidence, VALID_CONFIDENCES, source=f"{source}.expected_confidence")
    status = _parse_non_empty_str(expectation, "expected_status", source=source)
    _require_value(status, VALID_STATUSES, source=f"{source}.expected_status")

    _parse_str_list(
        expectation,
        "expected_subroles",
        source=source,
        allow_empty=fixture_class != "positive",
    )
    _parse_str_list(expectation, "required_evidence", source=source)
    _parse_str_list(
        expectation,
        "missing_evidence",
        source=source,
        allow_empty=fixture_class == "positive",
    )
    warning_prefixes = _parse_str_list(
        expectation,
        "warning_prefixes",
        source=source,
        allow_empty=fixture_class == "positive",
    )
    for warning in warning_prefixes:
        if not warning.startswith(VALID_WARNING_PREFIXES):
            valid = ", ".join(VALID_WARNING_PREFIXES)
            raise ValueError(
                f"{source}.warning_prefixes contains invalid prefix '{warning}', "
                f"expected one of: {valid}"
            )
    rationale = expectation["degradation_rationale"]
    if fixture_class == "positive":
        if rationale is not None:
            raise ValueError(f"{source}.degradation_rationale must be null")
    elif not isinstance(rationale, str) or not rationale:
        raise ValueError(f"{source}.degradation_rationale must be non-empty string")


def _require_keys(payload: dict[str, Any], required: frozenset[str], *, source: str) -> None:
    for key in sorted(required):
        if key not in payload:
            raise ValueError(f"{source} missing required field: {key}")


def _ensure_dict(value: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{source} must be an object")
    return value


def _parse_non_empty_str(payload: dict[str, Any], key: str, *, source: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f"{source}.{key} must be a string")
    if not value:
        raise ValueError(f"{source}.{key} must be non-empty")
    return value


def _parse_str_list(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"{source}.{key} must be a list")
    if not value and not allow_empty:
        raise ValueError(f"{source}.{key} must be non-empty")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{source}.{key}[{index}] must be a string")
        if not item:
            raise ValueError(f"{source}.{key}[{index}] must be non-empty")
        items.append(item)
    return tuple(items)


def _require_value(value: str, valid_values: frozenset[str], *, source: str) -> None:
    if value not in valid_values:
        valid = ", ".join(sorted(valid_values))
        raise ValueError(f"{source} has invalid value '{value}', expected one of: {valid}")
