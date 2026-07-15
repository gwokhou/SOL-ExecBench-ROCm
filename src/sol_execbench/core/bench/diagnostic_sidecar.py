# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared contract primitives for diagnostic sidecars.

Holds the freshness/governance classification helpers and the authority base
model that ``agent_feedback.py`` and ``profile_summary.py`` share. This module
depends on neither sidecar module (both import it, not vice-versa) to avoid
circular imports.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict

from sol_execbench.core.data.base_model import BaseModelWithDocstrings


class DiagnosticSidecarStatus(str, Enum):
    """Availability vocabulary shared by diagnostic-only sidecars."""

    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class DiagnosticFreshnessStatus(str, Enum):
    """Freshness vocabulary shared by diagnostic-only sidecars."""

    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


class DiagnosticGovernanceStatus(str, Enum):
    """Governance vocabulary shared by diagnostic-only sidecars."""

    USABLE_DIAGNOSTIC = "usable_diagnostic"
    STALE_DIAGNOSTIC = "stale_diagnostic"
    UNAVAILABLE = "unavailable"
    INVALID_DIAGNOSTIC = "invalid_diagnostic"


class DiagnosticSidecarAuthority(BaseModelWithDocstrings):
    """Shared diagnostic-only authority boundary for sidecar guardrails.

    Carries the frozen + extra-forbid config and the closed set of authority
    booleans that keep every diagnostic sidecar guardrail non-authoritative for
    benchmark truth (correctness / timing / scoring / release / ...).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    diagnostic_only: Literal[True] = True
    correctness_authority: Literal[False] = False
    performance_authority: Literal[False] = False
    timing_authority: Literal[False] = False
    score_authority: Literal[False] = False
    evidence_tier_authority: Literal[False] = False
    confirmed_improvement_authority: Literal[False] = False
    release_gate_authority: Literal[False] = False
    cutover_authority: Literal[False] = False
    paper_parity_authority: Literal[False] = False
    leaderboard_authority: Literal[False] = False
    claim_upgrade_authority: Literal[False] = False


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
) -> tuple[DiagnosticFreshnessStatus, list[str]]:
    """Tail classification shared by both freshness validators.

    Returns ``(status, reason_codes)`` using the shared freshness enum.

    ``any_expected`` is true iff the caller received at least one expected-identity
    argument. The caller owns this predicate because the two sidecars validate
    different identity field sets.
    """

    if reasons:
        return DiagnosticFreshnessStatus.STALE, list(reasons)
    if not any_expected:
        return DiagnosticFreshnessStatus.UNKNOWN, ["insufficient_expected_identity"]
    return DiagnosticFreshnessStatus.CURRENT, []


def classify_diagnostic_governance(
    *,
    sidecar_present: bool,
    freshness_status: DiagnosticFreshnessStatus | None,
    freshness_reason_codes: list[str] | None,
    parse_error: str | None,
) -> tuple[DiagnosticGovernanceStatus, list[str]]:
    """Shared governance state machine for diagnostic sidecars.

    Returns ``(status, reason_codes)`` using the shared governance enum.

    Branch order mirrors both originals verbatim:

      - ``parse_error`` -> ``invalid_diagnostic`` / ``["sidecar_parse_error"]``
      - no sidecar -> ``unavailable`` / ``["sidecar_missing"]``
      - freshness ``stale`` -> ``stale_diagnostic`` / (reasons or ``["sidecar_stale"]``)
      - freshness ``unknown`` -> ``unavailable`` / (reasons or ``["sidecar_freshness_unknown"]``)
      - otherwise -> ``usable_diagnostic`` / ``[]``
    """

    if parse_error is not None:
        return DiagnosticGovernanceStatus.INVALID_DIAGNOSTIC, ["sidecar_parse_error"]
    if not sidecar_present:
        return DiagnosticGovernanceStatus.UNAVAILABLE, ["sidecar_missing"]
    if freshness_status == DiagnosticFreshnessStatus.STALE:
        return DiagnosticGovernanceStatus.STALE_DIAGNOSTIC, freshness_reason_codes or [
            "sidecar_stale"
        ]
    if freshness_status == DiagnosticFreshnessStatus.UNKNOWN:
        return DiagnosticGovernanceStatus.UNAVAILABLE, freshness_reason_codes or [
            "sidecar_freshness_unknown"
        ]
    return DiagnosticGovernanceStatus.USABLE_DIAGNOSTIC, []
