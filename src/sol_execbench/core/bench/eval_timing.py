# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Workload support helpers for staged workload evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sol_execbench.core.bench.eval_runtime import (
    measure_latency,
    measure_reference_latency,
)
from sol_execbench.core.bench.io import allocate_outputs
from sol_execbench.core.bench.io import load_safetensors


def measure_solution_latency(
    *,
    definition: Any,
    resolved_axes: dict[str, int],
    device: str,
    destination_passing_style: bool,
    user_fn: Callable[..., Any],
    inputs: dict[str, Any],
    warmup: int,
    rep: int,
) -> float:
    """Measure user solution latency and raise when timing failed."""
    timing_outputs = (
        allocate_outputs(definition, resolved_axes, device)
        if destination_passing_style
        else []
    )
    timing = measure_latency(
        user_fn,
        inputs,
        timing_outputs,
        device,
        warmup=warmup,
        rep=rep,
    )
    if timing.failure is not None:
        raise RuntimeError(timing.failure)
    return timing.latency_ms


def measure_optional_reference_latency(
    *,
    benchmark_reference: bool,
    ref_fn: Callable[..., Any],
    inputs: dict[str, Any],
    device: str,
    warmup: int,
    rep: int,
) -> tuple[float, str | None]:
    """Measure reference latency when configured."""
    if not benchmark_reference:
        return 0.0, None

    timing = measure_reference_latency(
        ref_fn,
        inputs,
        device,
        warmup=warmup,
        rep=rep,
    )
    return timing.latency_ms, timing.failure


def load_required_safetensors(
    *,
    definition: Any,
    workload: Any,
    safetensors_roots: list[Any],
) -> dict[str, Any]:
    """Load safetensors inputs only when the workload references them."""
    if not any(value.type == "safetensors" for value in workload.inputs.values()):
        return {}
    return load_safetensors(definition, workload, safetensors_roots)
