# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Stable model-only selection with explicit unverified and unavailable coverage."""

from collections.abc import Sequence
from dataclasses import dataclass

from solar.analysis.order_contracts import (
    CandidateStatus,
    OrderCandidate,
    OrderSelectionReason,
)
from solar.analysis.views import AnalysisView, ViewAvailability


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderSelection:
    """A best known candidate, its coverage boundary and exact missing denominator."""

    candidate_id: str | None
    reason: OrderSelectionReason
    exhaustive_verified_domain: bool
    unexplored: int


def select_order(
    candidates: Sequence[OrderCandidate],
    view: AnalysisView,
    total: int,
    *,
    completed: bool,
) -> OrderSelection:
    """Select the first minimum in the original-first canonical enumeration."""
    accepted: list[tuple[float, str]] = []
    if len(candidates) > total:
        raise ValueError("candidate list exceeds the declared search domain")
    for candidate in candidates:
        if candidate.status is CandidateStatus.ANALYZED:
            estimate = (
                candidate.views[view] if candidate.views is not None else None
            )
            if (
                estimate is None
                or estimate.status is not ViewAvailability.AVAILABLE
                or estimate.seconds is None
            ):
                raise ValueError(
                    "analyzed candidate lacks its available objective"
                )
            accepted.append((estimate.seconds, candidate.candidate_id))
        elif candidate.status is CandidateStatus.UNAVAILABLE:
            if (
                candidate.views is None
                or candidate.views[view].status is ViewAvailability.AVAILABLE
            ):
                raise ValueError(
                    "unavailable candidate disagrees with its selected view"
                )
    complete = (
        completed
        and len(candidates) == total
        and all(
            item.status
            in {CandidateStatus.ANALYZED, CandidateStatus.VERIFICATION_FAILED}
            for item in candidates
        )
    )
    selected = min(accepted, key=lambda item: item[0])[1] if accepted else None
    reason = (
        OrderSelectionReason.NONE
        if selected is None
        else (
            OrderSelectionReason.MINIMUM
            if complete
            else OrderSelectionReason.BEST_AVAILABLE
        )
    )
    return OrderSelection(
        candidate_id=selected,
        reason=reason,
        exhaustive_verified_domain=complete,
        unexplored=total - sum(item.attempted for item in candidates),
    )
