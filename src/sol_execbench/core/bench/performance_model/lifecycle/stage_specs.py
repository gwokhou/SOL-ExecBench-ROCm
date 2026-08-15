# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Canonical lifecycle stage order, dependencies, and transitions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticLifecycleStage,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class StageSpec:
    """Immutable orchestration facts for one lifecycle stage."""

    stage: DiagnosticLifecycleStage
    dependencies: tuple[DiagnosticLifecycleStage, ...]


STAGE_SPECS: Final[tuple[StageSpec, ...]] = (
    StageSpec(stage=DiagnosticLifecycleStage.DESIGN, dependencies=()),
    StageSpec(stage=DiagnosticLifecycleStage.CALIBRATION, dependencies=()),
    StageSpec(
        stage=DiagnosticLifecycleStage.COLLECTION_RUN,
        dependencies=(DiagnosticLifecycleStage.DESIGN,),
    ),
    StageSpec(
        stage=DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
        dependencies=(DiagnosticLifecycleStage.COLLECTION_RUN,),
    ),
    StageSpec(
        stage=DiagnosticLifecycleStage.MODEL_BUILD,
        dependencies=(DiagnosticLifecycleStage.CALIBRATION,),
    ),
    StageSpec(
        stage=DiagnosticLifecycleStage.ACCEPTANCE,
        dependencies=(
            DiagnosticLifecycleStage.MODEL_BUILD,
            DiagnosticLifecycleStage.CALIBRATION,
            DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
        ),
    ),
    StageSpec(
        stage=DiagnosticLifecycleStage.PUBLICATION,
        dependencies=(
            DiagnosticLifecycleStage.ACCEPTANCE,
            DiagnosticLifecycleStage.CALIBRATION,
            DiagnosticLifecycleStage.MODEL_BUILD,
        ),
    ),
    StageSpec(
        stage=DiagnosticLifecycleStage.RELEASE,
        dependencies=(DiagnosticLifecycleStage.PUBLICATION,),
    ),
)

CHAIN: Final = tuple(spec.stage for spec in STAGE_SPECS)
STAGE_SPEC_BY_STAGE: Final = MappingProxyType(
    {spec.stage: spec for spec in STAGE_SPECS}
)
DEPENDENCIES: Final = MappingProxyType(
    {spec.stage: spec.dependencies for spec in STAGE_SPECS}
)
CHAIN_INDEX: Final = MappingProxyType(
    {stage: index for index, stage in enumerate(CHAIN)}
)

# Calibration is an independently established prerequisite, not a public
# progression state. The externally visible monotonic transition chain skips it.
TRANSITION_CHAIN: Final = tuple(
    stage
    for stage in CHAIN
    if stage is not DiagnosticLifecycleStage.CALIBRATION
)
LEGAL_TRANSITIONS: Final = MappingProxyType(
    {
        stage: (
            frozenset({TRANSITION_CHAIN[index + 1]})
            if index + 1 < len(TRANSITION_CHAIN)
            else frozenset()
        )
        for index, stage in enumerate(TRANSITION_CHAIN)
    }
)

__all__ = [
    "CHAIN",
    "CHAIN_INDEX",
    "DEPENDENCIES",
    "LEGAL_TRANSITIONS",
    "STAGE_SPECS",
    "STAGE_SPEC_BY_STAGE",
    "StageSpec",
]
