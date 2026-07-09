# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Low-level parsing utilities for SOLAR derivation sidecars."""

from __future__ import annotations

from math import isfinite
from typing import Any

from sol_execbench.core.scoring.confidence import EstimateConfidence
from sol_execbench.core.scoring.parsing_utils import (
    ensure_dict as _ensure_dict,
    parse_list as _parse_list,
    parse_str as _parse_str,
    parse_str_item as _parse_str_item,
)
from sol_execbench.core.scoring.solar_derivation.status import SOLAR_DERIVATION_STATUSES


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
