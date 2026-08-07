# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Legal monotonic transitions between lifecycle stages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticLifecycleStage,
)

LEGAL_TRANSITIONS: Final[
    Mapping[DiagnosticLifecycleStage, frozenset[DiagnosticLifecycleStage]]
] = {
    DiagnosticLifecycleStage.DESIGN: frozenset(
        {DiagnosticLifecycleStage.COLLECTION_RUN}
    ),
    DiagnosticLifecycleStage.COLLECTION_RUN: frozenset(
        {DiagnosticLifecycleStage.CORPUS_SNAPSHOT}
    ),
    DiagnosticLifecycleStage.CORPUS_SNAPSHOT: frozenset(
        {DiagnosticLifecycleStage.MODEL_BUILD}
    ),
    DiagnosticLifecycleStage.MODEL_BUILD: frozenset(
        {DiagnosticLifecycleStage.ACCEPTANCE}
    ),
    DiagnosticLifecycleStage.ACCEPTANCE: frozenset(
        {DiagnosticLifecycleStage.PUBLICATION}
    ),
    DiagnosticLifecycleStage.PUBLICATION: frozenset(
        {DiagnosticLifecycleStage.RELEASE}
    ),
    DiagnosticLifecycleStage.RELEASE: frozenset(),
}


def require_legal_transition(
    stage_from: DiagnosticLifecycleStage,
    stage_to: DiagnosticLifecycleStage,
) -> None:
    """Reject any transition outside the monotonic chain.

    A frozen stage may also be marked ``superseded`` by a successor
    generation, which is a status change on the same stage rather than a
    transition to a different stage kind.
    """
    successors = LEGAL_TRANSITIONS.get(stage_from, frozenset())
    if stage_to not in successors:
        raise ValueError(
            f"illegal lifecycle transition {stage_from.value} -> "
            f"{stage_to.value}",
        )


__all__ = ["LEGAL_TRANSITIONS", "require_legal_transition"]
