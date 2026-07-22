# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Validated timing helpers for staged workload evaluation."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from sol_execbench.core.bench.correctness import (
    check_output_shape_dtype,
    compute_error_stats,
)
from sol_execbench.core.bench.eval_runtime import TimingResult, measure_latency
from sol_execbench.core.bench.evaluation_requests import WorkloadEvaluationRequest
from sol_execbench.core.bench.io import allocate_outputs, normalize_outputs
from sol_execbench.core.bench.reward_hack import RewardHackDetected
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.platform.runtime import (
    CacheClearPolicy,
    cache_clear_policy_for_device,
)


@dataclass(frozen=True)
class SolutionTimingResult:
    """Aggregate solution latency and exact sample count for every trial."""

    latency_ms: float
    timed_iterations_per_trial: tuple[int, ...]
    cache_clear_policy: CacheClearPolicy | None = None

    @property
    def uniform_timed_iterations(self) -> int:
        """Return the shared trial count, or zero when trial counts differ."""
        counts = set(self.timed_iterations_per_trial)
        return counts.pop() if len(counts) == 1 else 0


def measure_solution_latency(
    *,
    request: WorkloadEvaluationRequest,
    workload: Workload,
    resolved_axes: dict[str, int],
    inputs: list[Any],
    expected_outputs: list[torch.Tensor],
) -> SolutionTimingResult:
    """Measure all paper trials and validate every timed invocation's output."""
    validator = _build_timed_output_validator(
        request=request,
        workload=workload,
        inputs=inputs,
        expected=expected_outputs,
    )
    cache_policy = (
        cache_clear_policy_for_device(request.device)
        if request.device.split(":", maxsplit=1)[0] == "cuda"
        else None
    )
    trials = [
        _measure_solution_trial(request, resolved_axes, inputs, validator, cache_policy)
        for _ in range(request.bench_config.trials)
    ]
    return SolutionTimingResult(
        latency_ms=statistics.mean(trial.latency_ms for trial in trials),
        timed_iterations_per_trial=tuple(trial.timed_iterations for trial in trials),
        cache_clear_policy=cache_policy,
    )


def _measure_solution_trial(
    request: WorkloadEvaluationRequest,
    resolved_axes: dict[str, int],
    inputs: list[Any],
    validator: Callable[[list[Any], Any], None],
    cache_clear_policy: CacheClearPolicy | None = None,
) -> TimingResult:
    outputs = (
        allocate_outputs(request.definition, resolved_axes, request.device)
        if request.destination_passing_style
        else []
    )
    config = request.bench_config
    timing = measure_latency(
        request.dependencies.user_fn,
        inputs,
        outputs,
        request.device,
        warmup=config.warmup_runs,
        rep=config.iterations,
        min_measurement_time_seconds=config.min_measurement_time_seconds,
        validator=validator,
        cache_clear_policy=cache_clear_policy,
    )
    if timing.failure is not None:
        raise RuntimeError(timing.failure)
    return timing


def _build_timed_output_validator(
    *,
    request: WorkloadEvaluationRequest,
    workload: Workload,
    inputs: list[Any],
    expected: list[torch.Tensor],
) -> Callable[[list[Any], Any], None]:
    def validate(args: list[Any], result: Any) -> None:
        request.dependencies.check_integrity(
            request.dependencies.integrity_snapshot,
            request.dependencies.driver_globals,
        )
        actual = _timed_outputs(request, inputs, args, result)
        issue = check_output_shape_dtype(expected, actual)
        if issue is not None:
            raise RewardHackDetected(
                f"timed invocation returned invalid output shape or dtype: {issue.value}"
            )
        for reference, candidate in zip(expected, actual, strict=True):
            _, exceeds = compute_error_stats(candidate, reference, workload.tolerance)
            if exceeds:
                raise RewardHackDetected(
                    "timed invocation output differs from the reference; "
                    "correctness and timing phases must execute identical behavior"
                )

    return validate


def _timed_outputs(
    request: WorkloadEvaluationRequest,
    inputs: list[Any],
    args: list[Any],
    result: Any,
) -> list[torch.Tensor]:
    if request.destination_passing_style:
        return list(args[len(inputs) :])
    normalized = normalize_outputs(
        result,
        device=torch.device(request.device),
        output_names=request.output_names,
        output_dtypes=request.output_dtypes_torch,
    )
    return [normalized[name] for name in request.output_names]


__all__ = [
    "measure_solution_latency",
]
