# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Reason and evidence-gap helpers for paper denominator reports."""

from __future__ import annotations

from typing import Any

from sol_execbench.core.dataset.paper_denominator_models import (
    EVIDENCE_KEYS,
    PaperDenominatorEvidenceGap,
    PaperDenominatorNextEvidenceHint,
    PaperDenominatorReasonBucket,
)

def _evidence_from_reason(reason_code: str) -> str | None:
    lowered = reason_code.lower()
    if "timing" in lowered:
        return "timing"
    if "amd_score" in lowered or "native_score" in lowered:
        return "amd_score"
    if "amd_sol" in lowered or "sol_bound" in lowered:
        return "amd_sol"
    if "solar" in lowered:
        return "solar_derivation"
    return None


def _next_evidence(reason_code: str) -> str:
    evidence = _evidence_from_reason(reason_code)
    if evidence is None:
        return "Review the bounded sidecar evidence for this reason code."
    return f"Attach bounded {evidence} evidence refs/checksums before upgrading claims."


def _add_missing_evidence(
    *,
    reason_groups: dict[str, dict[str, Any]],
    evidence_groups: dict[tuple[str, str], dict[str, Any]],
    reason_code: str,
    example_ref: str,
    next_evidence: str | None = None,
) -> None:
    _add_reason(
        reason_groups,
        reason_code=reason_code,
        state="evidence_missing",
        example_ref=example_ref,
        next_evidence=next_evidence,
    )
    _add_evidence_gap(
        evidence_groups,
        reason_code=reason_code,
        example_ref=example_ref,
    )


def _add_reason(
    groups: dict[str, dict[str, Any]],
    *,
    reason_code: str,
    state: str,
    example_ref: str,
    next_evidence: str | None = None,
) -> None:
    group = groups.setdefault(
        reason_code,
        {
            "reason_code": reason_code,
            "count": 0,
            "states": set(),
            "example_refs": [],
            "next_evidence": set(),
        },
    )
    group["count"] += 1
    group["states"].add(state)
    if len(group["example_refs"]) < 5 and example_ref not in group["example_refs"]:
        group["example_refs"].append(example_ref)
    group["next_evidence"].add(next_evidence or _next_evidence(reason_code))


def _add_evidence_gap(
    groups: dict[tuple[str, str], dict[str, Any]],
    *,
    reason_code: str,
    example_ref: str,
) -> None:
    evidence = _evidence_from_reason(reason_code)
    if evidence is None:
        return
    key = (evidence, reason_code)
    group = groups.setdefault(
        key,
        {
            "evidence": evidence,
            "reason_code": reason_code,
            "count": 0,
            "example_refs": [],
            "next_evidence": _next_evidence(reason_code),
        },
    )
    group["count"] += 1
    if len(group["example_refs"]) < 5 and example_ref not in group["example_refs"]:
        group["example_refs"].append(example_ref)


def _readiness_state(status: str) -> str:
    lowered = status.lower()
    if lowered == "ready":
        return "ready"
    if "unsupported" in lowered:
        return "unsupported"
    return "blocked"


def _closure_state(status: str) -> str | None:
    if status == "skipped_existing_pass":
        return "skipped"
    if status == "missing_trace":
        return "attempted_failed"
    if status == "derived_evidence_missing":
        return "deferred"
    if status in {
        "attempted_passed",
        "attempted_failed",
        "filtered",
        "not_attempted",
    }:
        return status
    return None


def _sorted_reason_buckets(
    groups: dict[str, dict[str, Any]],
) -> list[PaperDenominatorReasonBucket]:
    buckets = [
        PaperDenominatorReasonBucket(
            reason_code=group["reason_code"],
            count=group["count"],
            states=sorted(group["states"]),
            example_refs=sorted(group["example_refs"]),
            next_evidence=sorted(group["next_evidence"]),
        )
        for group in groups.values()
    ]
    return sorted(buckets, key=lambda bucket: (-bucket.count, bucket.reason_code))


def _sorted_evidence_gaps(
    groups: dict[tuple[str, str], dict[str, Any]],
) -> list[PaperDenominatorEvidenceGap]:
    gaps = [
        PaperDenominatorEvidenceGap(
            evidence=group["evidence"],
            reason_code=group["reason_code"],
            count=group["count"],
            example_refs=sorted(group["example_refs"]),
            next_evidence=group["next_evidence"],
        )
        for group in groups.values()
    ]
    return sorted(
        gaps, key=lambda gap: (EVIDENCE_KEYS.index(gap.evidence), gap.reason_code)
    )


def _next_hints(
    reason_buckets: list[PaperDenominatorReasonBucket],
) -> list[PaperDenominatorNextEvidenceHint]:
    hints = []
    for bucket in reason_buckets:
        for hint in bucket.next_evidence:
            hints.append(
                PaperDenominatorNextEvidenceHint(
                    reason_code=bucket.reason_code,
                    next_evidence=hint,
                    example_refs=bucket.example_refs,
                )
            )
    return sorted(hints, key=lambda hint: (hint.reason_code, hint.next_evidence))
