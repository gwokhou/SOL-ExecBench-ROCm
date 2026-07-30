# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Validated timing helpers for staged workload evaluation."""

from __future__ import annotations

import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from sol_execbench.core.bench.correctness import check_output_shape_dtype
from sol_execbench.core.bench.eval_runtime import TimingResult, measure_latency
from sol_execbench.core.bench.evaluation_requests import (
    WorkloadEvaluationRequest,
)
from sol_execbench.core.bench.io import allocate_outputs, normalize_outputs
from sol_execbench.core.bench.output_checks import compare_output_checks
from sol_execbench.core.bench.performance_model.replay_evidence import (
    REPLAY_EVIDENCE_ITERATIONS,
    REPLAY_WARMUP_RUNS,
)
from sol_execbench.core.bench.reward_hack import RewardHackError
from sol_execbench.core.bench.roctx_control import ROCTxReplayController
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.platform.runtime import (
    CacheClearPolicy,
    cache_clear_policy_for_device,
)
from sol_execbench.core.process.environment import (
    ENV_SOL_EXECBENCH_COUNTER_REPLAY,
)


@dataclass(frozen=True)
class SolutionTimingResult:
    """Aggregate solution latency and exact sample count for every trial."""

    latency_ms: float
    timed_iterations_per_trial: tuple[int, ...]
    trial_samples_ms: tuple[tuple[float, ...], ...]
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
    replay = _counter_replay_enabled()
    trial_count = 1 if replay else request.bench_config.trials
    trial_kwargs = {"workload": workload} if replay else {}
    trials = [
        _measure_solution_trial(
            request,
            resolved_axes,
            inputs,
            validator,
            cache_policy,
            **trial_kwargs,
        )
        for _ in range(trial_count)
    ]
    return SolutionTimingResult(
        latency_ms=statistics.mean(trial.latency_ms for trial in trials),
        timed_iterations_per_trial=tuple(
            trial.timed_iterations for trial in trials
        ),
        trial_samples_ms=tuple(trial.samples_ms for trial in trials),
        cache_clear_policy=cache_policy,
    )


def _measure_solution_trial(
    request: WorkloadEvaluationRequest,
    resolved_axes: dict[str, int],
    inputs: list[Any],
    validator: Callable[[list[Any], Any], None],
    cache_clear_policy: CacheClearPolicy | None = None,
    *,
    workload: Workload | None = None,
) -> TimingResult:
    outputs = (
        allocate_outputs(request.definition, resolved_axes, request.device)
        if request.destination_passing_style
        else []
    )
    config = request.bench_config
    replay = _counter_replay_enabled()
    if replay and workload is None:
        raise ValueError("counter replay requires workload identity")
    timed_fn = (
        _marked_replay_callable(
            request.dependencies.user_fn,
            workload.uuid if workload is not None else "",
            warmup=REPLAY_WARMUP_RUNS,
        )
        if replay
        else request.dependencies.user_fn
    )
    timing = measure_latency(
        timed_fn,
        inputs,
        outputs,
        request.device,
        warmup=REPLAY_WARMUP_RUNS if replay else config.warmup_runs,
        rep=REPLAY_EVIDENCE_ITERATIONS if replay else config.iterations,
        min_measurement_time_seconds=(
            None if replay else config.min_measurement_time_seconds
        ),
        validator=validator,
        cache_clear_policy=cache_clear_policy,
    )
    if timing.failure is not None:
        raise RuntimeError(timing.failure)
    return timing


def replay_marker_ranges(workload_uuid: str) -> list[str]:
    """Return the exact marker sequence required in every replay pass."""
    return [
        f"sol_execbench/{workload_uuid}/iteration/{index}"
        for index in range(REPLAY_EVIDENCE_ITERATIONS)
    ]


def _counter_replay_enabled() -> bool:
    return os.environ.get(ENV_SOL_EXECBENCH_COUNTER_REPLAY) == "1"


def _marked_replay_callable(
    fn: Callable[..., Any],
    workload_uuid: str,
    *,
    warmup: int,
) -> Callable[..., Any]:
    controller = ROCTxReplayController()
    controller.pause()
    marker_ranges = replay_marker_ranges(workload_uuid)
    invocation = 0

    def run(*args: Any) -> Any:
        nonlocal invocation
        index = invocation
        invocation += 1
        if index < warmup:
            return fn(*args)
        marker_index = index - warmup
        if marker_index >= len(marker_ranges):
            raise RuntimeError("replay iteration exceeds frozen protocol")
        return controller.run_range(marker_ranges[marker_index], fn, *args)

    return run


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
            raise RewardHackError(
                f"timed invocation returned invalid output shape or dtype: {issue}",
            )
        _, exceeds = compare_output_checks(
            request.definition,
            workload,
            inputs,
            expected,
            actual,
            0,
        )
        if exceeds:
            raise RewardHackError(
                "timed invocation output differs from the reference; "
                "correctness and timing phases must execute identical behavior",
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
    "replay_marker_ranges",
]
