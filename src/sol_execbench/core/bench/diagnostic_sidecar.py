# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for diagnostic sidecar freshness and governance classification.

These helpers are extracted verbatim from ``agent_feedback.py`` and
``profile_summary.py`` so the two sidecars share one freshness matching rule and
one governance state machine. This module depends on neither sidecar module
(both import it, not vice-versa) to avoid circular imports.
"""

from __future__ import annotations

from pathlib import Path


def compact_path(path: str | None) -> str | None:
    """Return the basename of ``path``, or ``None`` if ``path`` is ``None``."""

    if path is None:
        return None
    return Path(path).name


def match_optional(
    reasons: list[str],
    field: str,
    actual: str | None,
    expected: str | None,
) -> None:
    """Append ``"<field>_mismatch"`` when ``expected`` is set and differs."""

    if expected is not None and actual != expected:
        reasons.append(f"{field}_mismatch")


def match_required_optional(
    reasons: list[str],
    field: str,
    actual: str | None,
    expected: str | None,
) -> None:
    """Append ``"<field>_missing"`` / ``"<field>_mismatch"`` per the required rule."""

    if expected is None:
        return
    if actual is None:
        reasons.append(f"{field}_missing")
        return
    if actual != expected:
        reasons.append(f"{field}_mismatch")


def classify_freshness(
    reasons: list[str],
    *,
    any_expected: bool,
) -> tuple[str, list[str]]:
    """Tail classification shared by both freshness validators.

    Returns ``(status_value, reason_codes)`` where ``status_value`` is one of
    ``"current"`` / ``"stale"`` / ``"unknown"``. Callers map the string to their
    own enum via ``MyFreshnessStatus(status_value)``.

    ``any_expected`` is true iff the caller received at least one expected-identity
    argument. The caller owns this predicate because the two sidecars validate
    different identity field sets.
    """

    if reasons:
        return "stale", list(reasons)
    if not any_expected:
        return "unknown", ["insufficient_expected_identity"]
    return "current", []


def classify_diagnostic_governance(
    *,
    sidecar_present: bool,
    freshness_status: str | None,
    freshness_reason_codes: list[str] | None,
    parse_error: str | None,
) -> tuple[str, list[str]]:
    """Shared governance state machine for diagnostic sidecars.

    Returns ``(status_value, reason_codes)`` where ``status_value`` is one of
    ``"usable_diagnostic"`` / ``"stale_diagnostic"`` / ``"unavailable"`` /
    ``"invalid_diagnostic"``. Callers map via ``MyGovernanceStatus(status_value)``.

    Branch order mirrors both originals verbatim:

      - ``parse_error`` -> ``invalid_diagnostic`` / ``["sidecar_parse_error"]``
      - no sidecar -> ``unavailable`` / ``["sidecar_missing"]``
      - freshness ``stale`` -> ``stale_diagnostic`` / (reasons or ``["sidecar_stale"]``)
      - freshness ``unknown`` -> ``unavailable`` / (reasons or ``["sidecar_freshness_unknown"]``)
      - otherwise -> ``usable_diagnostic`` / ``[]``
    """

    if parse_error is not None:
        return "invalid_diagnostic", ["sidecar_parse_error"]
    if not sidecar_present:
        return "unavailable", ["sidecar_missing"]
    if freshness_status == "stale":
        return "stale_diagnostic", freshness_reason_codes or ["sidecar_stale"]
    if freshness_status == "unknown":
        return "unavailable", freshness_reason_codes or ["sidecar_freshness_unknown"]
    return "usable_diagnostic", []
