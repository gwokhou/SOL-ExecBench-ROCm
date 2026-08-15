"""Invariants for the single declarative lifecycle stage table."""

from __future__ import annotations

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticLifecycleStage,
)
from sol_execbench.core.bench.performance_model.lifecycle.stage_specs import (
    CHAIN,
    CHAIN_INDEX,
    DEPENDENCIES,
    LEGAL_TRANSITIONS,
    STAGE_SPECS,
)


def test_stage_specs_derive_unique_ordered_acyclic_dependencies() -> None:
    stages = tuple(spec.stage for spec in STAGE_SPECS)
    assert stages == CHAIN
    assert len(stages) == len(set(stages)) == len(DiagnosticLifecycleStage)
    assert {stage: index for index, stage in enumerate(CHAIN)} == CHAIN_INDEX
    assert {
        spec.stage: spec.dependencies for spec in STAGE_SPECS
    } == DEPENDENCIES
    for stage, dependencies in DEPENDENCIES.items():
        assert len(dependencies) == len(set(dependencies))
        assert all(
            CHAIN_INDEX[dependency] < CHAIN_INDEX[stage]
            for dependency in dependencies
        )


def test_legal_transitions_are_derived_from_public_progression() -> None:
    progression = tuple(
        stage
        for stage in CHAIN
        if stage is not DiagnosticLifecycleStage.CALIBRATION
    )
    assert {
        stage: (
            frozenset({progression[index + 1]})
            if index + 1 < len(progression)
            else frozenset()
        )
        for index, stage in enumerate(progression)
    } == LEGAL_TRANSITIONS
