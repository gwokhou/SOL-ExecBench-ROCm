# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Measured baseline coverage validation (BASE-02).

Classifies each expected workload's measured baseline into one of five stable
states — ``confirmed``, ``missing``, ``stale``, ``mismatched``, ``placeholder``
— each with a stable reason code, by comparing each baseline entry's recorded
provenance against an explicit current-run environment.

Authority boundary: this module is a coverage classifier only. It does NOT
compute scores, mutate canonical Trace JSONL, or promote any evidence tier. The
official score gate (:mod:`sol_execbench.core.scoring.official_score`) consumes
the report as a precondition; it does not import this module's internals beyond
the public types.

The ``missing`` and ``placeholder`` reason-code literals deliberately match the
existing official-score blockers ``missing_baseline`` / ``placeholder_baseline``
(D-08: no parallel namespace). A consistency test in
``test_official_score_evidence.py`` pins that equality.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Coverage states (D-02) -------------------------------------------------

CONFIRMED = "confirmed"
MISSING = "missing"
STALE = "stale"
MISMATCHED = "mismatched"
PLACEHOLDER = "placeholder"

COVERAGE_STATES: tuple[str, ...] = (
    CONFIRMED,
    MISSING,
    STALE,
    MISMATCHED,
    PLACEHOLDER,
)

# --- Reason codes (D-06 / D-07 / D-08) --------------------------------------
#
# missing/placeholder REUSE the official-score blocker literals (D-08). The new
# stale/mismatched codes use the ``baseline_*`` prefix (D-07). Sub-codes (D-06)
# give each success-criterion-#4 test case its own code.

MISSING_BASELINE_CODE = "missing_baseline"
PLACEHOLDER_BASELINE_CODE = "placeholder_baseline"
BASELINE_STALE_CODE = "baseline_stale"
BASELINE_STALE_TRACE_CODE = "baseline_stale_trace"
BASELINE_MISMATCHED_CODE = "baseline_mismatched"
BASELINE_HARDWARE_MISMATCH_CODE = "baseline_hardware_mismatch"
BASELINE_TIMING_POLICY_MISMATCH_CODE = "baseline_timing_policy_mismatch"

# Reference-latency-derived baseline sources are placeholder evidence, mirroring
# ``official_score._PLACEHOLDER_BASELINE_SOURCES``. Kept as a local literal so
# this module does not import from ``sol_execbench.core.scoring``.
_PLACEHOLDER_SOURCE_MARKERS: tuple[str, ...] = (
    "reference_latency",
    "trace_reference_latency",
    "trace.evaluation.performance.reference_latency_ms",
)


@dataclass(frozen=True)
class CurrentRunEnvironment:
    """Identity of the run that coverage is being validated for (D-03).

    Each field is compared against a baseline entry's recorded provenance to
    detect ``mismatched``. A field set to ``None`` is treated as "do not
    compare" so callers can validate a subset of the identity.
    """

    hardware: str | None = None
    rocm_version: str | None = None
    target_id: str | None = None
    timing_policy: str | None = None


@dataclass(frozen=True)
class BaselineCoverageEntry:
    """Per-workload coverage classification (D-05)."""

    workload_key: str
    state: str
    reason_code: str | None
    detail: str = ""
    sub_codes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_confirmed(self) -> bool:
        return self.state == CONFIRMED

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_key": self.workload_key,
            "state": self.state,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "sub_codes": list(self.sub_codes),
        }


@dataclass(frozen=True)
class BaselineCoverageReport:
    """Suite-level measured baseline coverage report (D-05).

    Mirrors the layering of ``OfficialScoreSuiteEvidence``: per-workload rows
    plus a suite-level state summary.
    """

    entries: tuple[BaselineCoverageEntry, ...]
    expected_workload_keys: tuple[str, ...]

    @property
    def state_summary(self) -> dict[str, int]:
        """Count of entries per state (present states only).

        Mirrors ``OfficialScoreSuiteEvidence.blocker_summary``, which only
        lists blockers that occurred. Use ``.get(state, 0)`` for absent states.
        """
        summary: dict[str, int] = {}
        for entry in self.entries:
            summary[entry.state] = summary.get(entry.state, 0) + 1
        return summary

    @property
    def all_confirmed(self) -> bool:
        """True only when every expected workload has a confirmed baseline."""
        return bool(self.expected_workload_keys) and all(
            entry.is_confirmed for entry in self.entries
        )

    @property
    def blocker_reason_codes(self) -> tuple[str, ...]:
        """Non-confirmed reason codes (primary + sub-codes), stable-ordered.

        Consumed by the official-score gate: the umbrella
        ``baseline_coverage_failed`` blocker plus these propagated codes give HIP
        precise failure reasons (D-11). ``confirmed`` entries contribute nothing
        (D-09 — positive status, not a blocker).
        """
        codes: list[str] = []
        for entry in self.entries:
            if entry.is_confirmed:
                continue
            if entry.reason_code is not None:
                codes.append(entry.reason_code)
            codes.extend(entry.sub_codes)
        # Dedupe while preserving stable order.
        return tuple(dict.fromkeys(codes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_workload_keys": list(self.expected_workload_keys),
            "state_summary": self.state_summary,
            "all_confirmed": self.all_confirmed,
            "blocker_reason_codes": list(self.blocker_reason_codes),
            "entries": [entry.to_dict() for entry in self.entries],
        }


def validate_baseline_coverage(
    registry: Mapping[str, Any],
    *,
    current_run_environment: CurrentRunEnvironment | None = None,
    trace_root: Path | None = None,
) -> BaselineCoverageReport:
    """Classify measured baseline coverage for a registry (D-01..D-09).

    ``registry`` is the payload produced by
    :func:`sol_execbench.core.evidence.baseline_export.export_hip_baseline_registry`
    (or an equivalent JSON loaded into a mapping). Classification precedence per
    expected ``workload_key``:

    1. **missing** — no baseline entry covers the workload.
    2. **placeholder** — the entry is reference-latency-derived, not measured.
    3. **stale** — the entry's ``trace_ref`` file does not exist (primary stale
       signal, D-04). ``trace_root`` resolves relative refs; absolute refs are
       checked directly. When neither resolves, stale-via-file is skipped.
    4. **mismatched** — entry provenance differs from ``current_run_environment``
       (D-03). Hardware and timing-policy mismatches carry their own sub-codes.
    5. **confirmed** — the entry covers the workload with matching provenance.
    """
    expected_keys = tuple(_expected_workload_keys(registry))
    entries_by_key = {
        str(entry.get("workload_key")): entry
        for entry in registry.get("entries", [])
        if isinstance(entry, Mapping) and entry.get("workload_key") is not None
    }

    coverage_entries: list[BaselineCoverageEntry] = []
    for workload_key in expected_keys:
        entry = entries_by_key.get(workload_key)
        coverage_entries.append(
            _classify(
                workload_key=workload_key,
                entry=entry,
                current_run_environment=current_run_environment,
                trace_root=trace_root,
            )
        )

    return BaselineCoverageReport(
        entries=tuple(coverage_entries),
        expected_workload_keys=expected_keys,
    )


def _classify(
    *,
    workload_key: str,
    entry: Mapping[str, Any] | None,
    current_run_environment: CurrentRunEnvironment | None,
    trace_root: Path | None,
) -> BaselineCoverageEntry:
    if entry is None:
        return BaselineCoverageEntry(
            workload_key=workload_key,
            state=MISSING,
            reason_code=MISSING_BASELINE_CODE,
            detail="no baseline entry covers this workload",
        )

    placeholder_marker = _placeholder_marker(entry)
    if placeholder_marker is not None:
        return BaselineCoverageEntry(
            workload_key=workload_key,
            state=PLACEHOLDER,
            reason_code=PLACEHOLDER_BASELINE_CODE,
            detail=f"baseline derived from reference latency ({placeholder_marker})",
        )

    stale = _stale_trace(entry, trace_root=trace_root)
    if stale:
        return BaselineCoverageEntry(
            workload_key=workload_key,
            state=STALE,
            reason_code=BASELINE_STALE_CODE,
            detail=f"trace_ref file missing: {stale}",
            sub_codes=(BASELINE_STALE_TRACE_CODE,),
        )

    mismatch = _provenance_mismatch(entry, current_run_environment)
    if mismatch is not None:
        reason_code, sub_codes, detail = mismatch
        return BaselineCoverageEntry(
            workload_key=workload_key,
            state=MISMATCHED,
            reason_code=reason_code,
            detail=detail,
            sub_codes=sub_codes,
        )

    return BaselineCoverageEntry(
        workload_key=workload_key,
        state=CONFIRMED,
        reason_code=None,
        detail="baseline covers workload with matching provenance",
    )


def _placeholder_marker(entry: Mapping[str, Any]) -> str | None:
    """Return the reference-latency marker that marks this entry placeholder."""
    source = entry.get("source")
    baseline_source = entry.get("baseline_source")
    candidates = [source, baseline_source]
    if isinstance(entry.get("facts"), Mapping):
        candidates.append(entry["facts"].get("baseline_source"))
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        lowered = candidate.lower()
        for marker in _PLACEHOLDER_SOURCE_MARKERS:
            if marker in lowered:
                return marker
    return None


def _stale_trace(entry: Mapping[str, Any], *, trace_root: Path | None) -> str | None:
    """Return the trace_ref string if its file does not exist, else None."""
    trace_ref = entry.get("trace_ref")
    if not isinstance(trace_ref, str) or not trace_ref:
        return None
    trace_path = Path(trace_ref)
    if not trace_path.is_absolute() and trace_root is not None:
        trace_path = trace_root / trace_path
    elif not trace_path.is_absolute():
        # Relative ref with no trace_root: cannot verify existence.
        return None
    if not trace_path.exists():
        return trace_ref
    return None


def _provenance_mismatch(
    entry: Mapping[str, Any],
    current_run_environment: CurrentRunEnvironment | None,
) -> tuple[str, tuple[str, ...], str] | None:
    """Compare entry provenance against the current-run environment (D-03).

    Returns ``(reason_code, sub_codes, detail)`` for the first mismatching
    field, or ``None`` if provenance matches (or no environment was supplied).
    Hardware and timing-policy mismatches carry dedicated sub-codes (D-06);
    rocm_version/target_id mismatches surface in ``detail`` without a sub-code.
    """
    if current_run_environment is None:
        return None
    provenance = entry.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}

    env = current_run_environment
    if env.hardware is not None and str(provenance.get("hardware")) != env.hardware:
        return (
            BASELINE_MISMATCHED_CODE,
            (BASELINE_HARDWARE_MISMATCH_CODE,),
            f"hardware: baseline={provenance.get('hardware')!r} current={env.hardware!r}",
        )
    if (
        env.timing_policy is not None
        and str(provenance.get("timing_policy")) != env.timing_policy
    ):
        return (
            BASELINE_MISMATCHED_CODE,
            (BASELINE_TIMING_POLICY_MISMATCH_CODE,),
            f"timing_policy: baseline={provenance.get('timing_policy')!r} current={env.timing_policy!r}",
        )
    if (
        env.rocm_version is not None
        and str(provenance.get("rocm_version")) != env.rocm_version
    ):
        return (
            BASELINE_MISMATCHED_CODE,
            (),
            f"rocm_version: baseline={provenance.get('rocm_version')!r} current={env.rocm_version!r}",
        )
    if env.target_id is not None and str(provenance.get("target_id")) != env.target_id:
        return (
            BASELINE_MISMATCHED_CODE,
            (),
            f"target_id: baseline={provenance.get('target_id')!r} current={env.target_id!r}",
        )
    return None


def _expected_workload_keys(registry: Mapping[str, Any]) -> list[str]:
    raw = registry.get("expected_workload_keys")
    if not isinstance(raw, list):
        return []
    return [str(key) for key in raw]


__all__ = [
    "BASELINE_HARDWARE_MISMATCH_CODE",
    "BASELINE_MISMATCHED_CODE",
    "BASELINE_STALE_CODE",
    "BASELINE_STALE_TRACE_CODE",
    "BASELINE_TIMING_POLICY_MISMATCH_CODE",
    "CONFIRMED",
    "COVERAGE_STATES",
    "CurrentRunEnvironment",
    "MISSING",
    "MISSING_BASELINE_CODE",
    "MISMATCHED",
    "PLACEHOLDER",
    "PLACEHOLDER_BASELINE_CODE",
    "STALE",
    "BaselineCoverageEntry",
    "BaselineCoverageReport",
    "validate_baseline_coverage",
]
