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


def apply_runtime_precedence(
    sidecar: DecisionSidecar,
    *,
    runtime_profile_available: bool,
) -> DecisionSidecar:
    """Annotate a decision sidecar with the runtime-precedence limitation.

    When a runtime profile is available, the precedence note is appended to
    ``limitations[]`` (idempotent). Unavailable sidecars and no-runtime cases
    are returned unchanged. Never overrides hint confidence or runtime
    classifications.
    """

    if not runtime_profile_available:
        return sidecar
    if sidecar.status == DecisionStatus.UNAVAILABLE:
        return sidecar
    limitations = list(sidecar.limitations)
    if _RUNTIME_PRECEDENCE_NOTE not in limitations:
        limitations.append(_RUNTIME_PRECEDENCE_NOTE)
    return sidecar.model_copy(update={"limitations": limitations})
