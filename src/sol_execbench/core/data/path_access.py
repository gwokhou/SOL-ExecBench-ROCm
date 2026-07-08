"""Helpers for nested payload access."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from glom import PathAccessError, glom


def path_get(payload: object, path: str, *, default: Any = None) -> Any:
    """Return a nested payload value or a default when the path is absent."""
    return glom(payload, path, default=default)


def path_require(payload: object, path: str, *, source: str = "payload") -> Any:
    """Return a nested payload value, raising ValueError when it is absent."""
    try:
        return glom(payload, path)
    except PathAccessError as exc:
        raise ValueError(f"{source} missing required field: {path}") from exc


def path_dict(
    payload: object,
    path: str,
    *,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = path_get(payload, path)
    if isinstance(value, Mapping):
        return dict(value)
    return {} if default is None else dict(default)


def path_list(payload: object, path: str) -> list[Any]:
    value = path_get(payload, path)
    return list(value) if isinstance(value, list) else []


def path_str_or_none(payload: object, path: str) -> str | None:
    value = path_get(payload, path)
    return value if isinstance(value, str) else None


def path_int(payload: object, path: str, *, default: int = 0) -> int:
    value = path_get(payload, path)
    if isinstance(value, bool):
        return default
    return value if isinstance(value, int) else default


def path_float_or_none(payload: object, path: str) -> float | None:
    value = path_get(payload, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None
