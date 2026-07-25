# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Timestamp helpers for generated artifacts."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_timestamp() -> str:
    """Return a second-resolution UTC ISO-8601 timestamp (Z-suffixed)."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_utc_timestamp(value: object) -> str:
    """Return one canonical second-resolution UTC timestamp."""

    timestamp = str(value)
    try:
        parsed = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(
            "timestamp must use canonical YYYY-MM-DDTHH:MM:SSZ form"
        ) from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != timestamp:
        raise ValueError("timestamp is not canonical UTC")
    return timestamp


__all__ = ["utc_timestamp", "validate_utc_timestamp"]
