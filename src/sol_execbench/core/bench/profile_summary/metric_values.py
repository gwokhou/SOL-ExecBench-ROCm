# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Canonical normalization for profile-summary metric values."""

from __future__ import annotations

import math


def normalize_metric_key(value: str | None) -> str:
    """Return the case-insensitive alphanumeric metric identity."""
    return "".join(
        character for character in (value or "").lower() if character.isalnum()
    )


def finite_number_or_none(value: float) -> int | float | None:
    """Return finite metric values and reject NaN or infinity."""
    return value if math.isfinite(value) else None
