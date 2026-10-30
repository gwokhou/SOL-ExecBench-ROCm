# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Interpret raw AMD-SMI throttle fields with unsupported values kept unknown."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1


class ThrottleStatus(StrEnum):
    """Known active, known inactive, or unavailable throttle evidence."""

    THROTTLED = "THROTTLED"
    UNTHROTTLED = "UNTHROTTLED"
    UNKNOWN = "UNKNOWN"


def classify_throttle(
    value: object, *, maximum: int = UINT32_MAX
) -> ThrottleStatus:
    """Maximum unsigned values are unavailable, not a set of throttle bits."""
    if type(value) is not int or not 0 <= value < maximum:
        return ThrottleStatus.UNKNOWN
    return ThrottleStatus.THROTTLED if value else ThrottleStatus.UNTHROTTLED


def evidence_status(value: object) -> ThrottleStatus:
    """Recompute state from raw evidence, never trust a serialized status label."""
    if not isinstance(value, Mapping) or value.get("error") is not None:
        return ThrottleStatus.UNKNOWN
    raw = value.get("raw")
    header = value.get("metrics_header")
    version = value.get("api_version")
    if (
        not isinstance(raw, Mapping)
        or not isinstance(header, Mapping)
        or not isinstance(version, Mapping)
    ):
        return ThrottleStatus.UNKNOWN
    for name in ("format_revision", "content_revision", "structure_size"):
        number = header.get(name)
        if type(number) is not int or number < 0:
            return ThrottleStatus.UNKNOWN
    if not header.get("format_revision") or not header.get("structure_size"):
        return ThrottleStatus.UNKNOWN
    major = version.get("major")
    if type(major) is not int or major <= 0 or not value.get("cli_version"):
        return ThrottleStatus.UNKNOWN
    primary = raw.get("throttle_status")
    independent = raw.get("indep_throttle_status")
    if not isinstance(primary, Mapping) or primary.get("type") != "int":
        return ThrottleStatus.UNKNOWN
    state = classify_throttle(primary.get("value"))
    if isinstance(independent, Mapping):
        other = classify_throttle(independent.get("value"), maximum=UINT64_MAX)
        if other != ThrottleStatus.UNKNOWN and state != other:
            return ThrottleStatus.UNKNOWN
    return state


def snapshot_throttle_status(
    evidence: object, cli_status: object
) -> ThrottleStatus:
    """Do not certify no-throttle when the adjacent CLI observation disagrees."""
    state = evidence_status(evidence)
    if (
        state == ThrottleStatus.UNTHROTTLED
        and cli_status != ThrottleStatus.UNTHROTTLED
    ):
        return ThrottleStatus.UNKNOWN
    return state
