# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Legal monotonic transitions between lifecycle stages."""

from __future__ import annotations

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticLifecycleStage,
)
from sol_execbench.core.bench.performance_model.lifecycle.stage_specs import (
    LEGAL_TRANSITIONS,
)


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
