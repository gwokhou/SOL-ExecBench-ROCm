# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Typed contracts for the generated workload evaluation driver."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import torch

from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload


@dataclass(frozen=True)
class EvaluationDependencies:
    """Injected functions and process state used by the evaluation loop."""

    ref_namespace: dict[str, Any]
    ref_fn: Callable[..., Any]
    user_fn: Callable[..., Any]
    integrity_snapshot: dict[str, int]
    check_integrity: Callable[[dict[str, int], dict[str, Any]], None]
    driver_globals: dict[str, Any]
    real_stdout: TextIO


@dataclass(frozen=True, kw_only=True)
class WorkloadEvaluationRequest:
    """Data configuration for evaluating a homogeneous workload collection."""

    definition: Definition
    workloads: list[Workload]
    solution_name: str
    device: str
    output_names: list[str]
    output_dtypes_torch: dict[str, torch.dtype]
    bench_config: BenchmarkConfig
    destination_passing_style: bool
    safetensors_roots: Iterable[Path]
    dependencies: EvaluationDependencies


__all__ = ["EvaluationDependencies", "WorkloadEvaluationRequest"]
