# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Benchmark protocol for cross-hardware GPU Kernel Agent evaluation."""

from sol_execbench.core.generalization.models import (
    CorpusAgentView,
    HardwareGeneralizationCell,
    HardwareGeneralizationPlan,
    HardwareGeneralizationReport,
)
from sol_execbench.core.generalization.workflow import (
    aggregate_study,
    build_study_plan,
    seal_cell,
)

__all__ = [
    "CorpusAgentView",
    "HardwareGeneralizationCell",
    "HardwareGeneralizationPlan",
    "HardwareGeneralizationReport",
    "aggregate_study",
    "build_study_plan",
    "seal_cell",
]
