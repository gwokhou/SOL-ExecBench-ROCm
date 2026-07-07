# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared parsers for strict JSON sidecar payloads."""

from __future__ import annotations

from typing import Any

from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence


def ensure_dict(value: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{source} must be an object")
    return value


def parse_list(payload: dict[str, Any], key: str, *, source: str) -> list[Any]:
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"{source}.{key} must be a list")
    return value


def parse_str(payload: dict[str, Any], key: str, *, source: str) -> str:
    return parse_str_item(payload[key], source=f"{source}.{key}")


def parse_optional_str(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> str | None:
    value = payload[key]
    if value is None:
        return None
    return parse_str_item(value, source=f"{source}.{key}")


def parse_str_item(value: Any, *, source: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{source} must be a string")
    if not value:
        raise ValueError(f"{source} must be non-empty")
    return value


def parse_confidence(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> EstimateConfidence:
    raw = parse_str(payload, key, source=source)
    try:
        return EstimateConfidence(raw)
    except ValueError as exc:
        valid_values = ", ".join(value.value for value in EstimateConfidence)
        raise ValueError(
            f"{source}.{key} has invalid confidence '{raw}', expected one of: {valid_values}"
        ) from exc
