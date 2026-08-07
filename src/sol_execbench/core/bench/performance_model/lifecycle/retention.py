# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Retention policy for lifecycle retention classes."""

from __future__ import annotations

from datetime import datetime
from typing import Final

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticRetentionClass,
)

_RETENTION_DURATIONS: Final[dict[DiagnosticRetentionClass, float | None]] = {
    # ``None`` means "no age-based reclamation": retained until registry
    # reachability proves otherwise.
    DiagnosticRetentionClass.CACHE: 0.0,
    DiagnosticRetentionClass.DEBUG: 14.0,
    DiagnosticRetentionClass.PROCESS_EVIDENCE: 90.0,
    DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE: None,
    DiagnosticRetentionClass.PUBLICATION_RELEASE: None,
}


def retention_reclaimable(
    retention_class: DiagnosticRetentionClass,
    created_at: datetime,
    now: datetime,
) -> bool:
    """Return whether age policy alone permits reclaiming this object.

    Objects with ``None`` duration are never age-reclaimable; GC must still
    prove they are unreachable before any deletion. Cache objects are
    immediately age-reclaimable; debug and process evidence use bounded
    day durations.
    """
    retention_class = DiagnosticRetentionClass(retention_class)
    days = _RETENTION_DURATIONS[retention_class]
    if days is None:
        return False
    elapsed = now - created_at
    return elapsed.total_seconds() >= float(days) * 24 * 60 * 60


def retention_duration_days(
    retention_class: DiagnosticRetentionClass,
) -> float | None:
    """Return the age-based retention window in days, if any."""
    return _RETENTION_DURATIONS[DiagnosticRetentionClass(retention_class)]


__all__ = ["retention_duration_days", "retention_reclaimable"]
