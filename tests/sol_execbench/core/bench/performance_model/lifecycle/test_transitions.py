from __future__ import annotations

import itertools

import pytest

from sol_execbench.core.bench.performance_model.lifecycle import (
    DiagnosticLifecycleStage,
    require_legal_transition,
)


def test_full_chain_is_legal() -> None:
    chain = (
        DiagnosticLifecycleStage.DESIGN,
        DiagnosticLifecycleStage.COLLECTION_RUN,
        DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
        DiagnosticLifecycleStage.MODEL_BUILD,
        DiagnosticLifecycleStage.ACCEPTANCE,
        DiagnosticLifecycleStage.PUBLICATION,
        DiagnosticLifecycleStage.RELEASE,
    )
    for stage_from, stage_to in itertools.pairwise(chain):
        require_legal_transition(stage_from, stage_to)


@pytest.mark.parametrize(
    ("stage_from", "stage_to"),
    [
        (
            DiagnosticLifecycleStage.DESIGN,
            DiagnosticLifecycleStage.RELEASE,
        ),
        (
            DiagnosticLifecycleStage.PUBLICATION,
            DiagnosticLifecycleStage.DESIGN,
        ),
        (
            DiagnosticLifecycleStage.ACCEPTANCE,
            DiagnosticLifecycleStage.COLLECTION_RUN,
        ),
    ],
)
def test_illegal_jumps_are_rejected(
    stage_from: DiagnosticLifecycleStage,
    stage_to: DiagnosticLifecycleStage,
) -> None:
    with pytest.raises(ValueError, match="illegal lifecycle transition"):
        require_legal_transition(stage_from, stage_to)
