# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""6-family ROCm inference primitive workloads in FlashInfer-Trace format."""

from __future__ import annotations

from sol_execbench.vertical.workloads.cases import build_runtime_case
from sol_execbench.vertical.workloads.specs import (
    RuntimeCase,
    SweepMode,
    WorkloadSpec,
    build_trace_definitions,
    build_workload_specs,
    materialize_trace_corpus,
    to_trace_workload,
)

__all__ = [
    "RuntimeCase",
    "SweepMode",
    "WorkloadSpec",
    "build_runtime_case",
    "build_trace_definitions",
    "build_workload_specs",
    "materialize_trace_corpus",
    "to_trace_workload",
]
