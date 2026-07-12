# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Timestamp helpers for generated artifacts."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_timestamp() -> str:
    """Return a second-resolution UTC ISO-8601 timestamp (Z-suffixed)."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
