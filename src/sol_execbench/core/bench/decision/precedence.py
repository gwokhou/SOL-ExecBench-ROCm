# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Cross-sidecar precedence: runtime measured hints supersede static inferred ones.

The Decision sidecar carries static-inferred Layer R hints. When a runtime
profile (``profile_summary.v2``) is also available, its measured bottleneck
classifications take precedence and the static hints become supplementary
(decision-modeling-research.md §8.4). This module annotates that relationship
on the sidecar without ever overriding hint confidence or runtime truth.
"""

from __future__ import annotations

from sol_execbench.core.bench.decision.decision_models import (
    DecisionBottleneckClass,
    DecisionConfidence,
    DecisionSidecar,
    DecisionStatus,
)

_RUNTIME_PRECEDENCE_NOTE = (
    "Runtime profile takes precedence over static decision hints; static hints "
    "are supplementary and must not override measured bottleneck classifications."
)


def runtime_precedence_limitation() -> str:
    """Return the stable runtime-precedence limitation string."""

    return _RUNTIME_PRECEDENCE_NOTE


_DEMOTED_LIMITATION = (
    "Demoted to inferred_low: a runtime profile supersedes this static hint."
)


def apply_runtime_precedence(
    sidecar: DecisionSidecar,
    *,
    runtime_profile_available: bool,
    demoted_classes: set[DecisionBottleneckClass] | None = None,
) -> DecisionSidecar:
    """Apply runtime-over-static precedence to a decision sidecar.

    When a runtime profile is available, the precedence note is appended to
    ``limitations[]`` (idempotent), and any static hint whose bottleneck class
    is in ``demoted_classes`` has its confidence lowered to ``inferred_low`` with
    a per-hint limitation — a real merge, not just an annotation. The caller
    decides which classes a runtime profile supersedes. Unavailable sidecars and
    no-runtime cases are returned unchanged.
    """

    if not runtime_profile_available:
        return sidecar
    if sidecar.status == DecisionStatus.UNAVAILABLE:
        return sidecar
    limitations = list(sidecar.limitations)
    if _RUNTIME_PRECEDENCE_NOTE not in limitations:
        limitations.append(_RUNTIME_PRECEDENCE_NOTE)
    demoted = demoted_classes or set()
    hints = []
    for hint in sidecar.hints:
        if (
            hint.bottleneck_class in demoted
            and hint.confidence != DecisionConfidence.INFERRED_LOW
        ):
            hint_limitations = list(hint.limitations)
            if _DEMOTED_LIMITATION not in hint_limitations:
                hint_limitations.append(_DEMOTED_LIMITATION)
            hint = hint.model_copy(
                update={
                    "confidence": DecisionConfidence.INFERRED_LOW,
                    "limitations": hint_limitations,
                }
            )
        hints.append(hint)
    return sidecar.model_copy(update={"limitations": limitations, "hints": hints})
