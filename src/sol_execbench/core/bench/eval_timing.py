# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Candidate-side timing with trusted per-invocation validation.

Only the candidate call is enclosed by device events. Input generation,
reference execution, IPC, and correctness comparison stay outside the timed
region, but every warmup and timed call receives a fresh reference-service case
and must return a result on the selected GPU before the end event. The helper
then sends candidate outputs back for one-shot validation before advancing.

This module must not cache timing inputs, expose reference outputs, validate
only a sample of iterations, or let a candidate switch devices mid-timing.
Those changes would invalidate the v4 timing protocol rather than optimize it.
"""

from __future__ import annotations

import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from sol_execbench.core.bench.eval_runtime import TimingResult, measure_latency
from sol_execbench.core.bench.evaluation_requests import (
    WorkloadEvaluationRequest,
)
from sol_execbench.core.bench.io import allocate_outputs, normalize_outputs
from sol_execbench.core.bench.performance_model.replay_evidence import (
    REPLAY_EVIDENCE_ITERATIONS,
    REPLAY_WARMUP_RUNS,
)
from sol_execbench.core.bench.reference_protocol import (
    ReferenceExecutionError,
    ReferenceTimingCase,
)
from sol_execbench.core.bench.reward_hack import RewardHackError
from sol_execbench.core.bench.roctx_control import ROCTxReplayController
from sol_execbench.core.bench.timing_contracts import TimingCallbacks
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


@dataclass
class _TimedReferenceState:
    """Reference material bound to the current candidate invocation."""

    inputs: list[Any] = field(default_factory=list)


def measure_solution_latency(
    *,
    request: WorkloadEvaluationRequest,
    workload: Workload,
    row_index: int,
    resolved_axes: dict[str, int],
) -> SolutionTimingResult:
    """Measure all paper trials and validate every timed invocation's output."""
    cache_policy = (
        cache_clear_policy_for_device(request.device)
        if request.device.split(":", maxsplit=1)[0] == "cuda"
        else None
    )
    replay = _counter_replay_enabled()
    trial_count = 1 if replay else request.bench_config.trials
    trial_kwargs = {"workload": workload} if replay else {}
    trials: list[TimingResult] = []
    for trial_index in range(trial_count):
        state = _TimedReferenceState()
        provider = _build_iteration_provider(
            request=request,
            workload=workload,
            row_index=row_index,
            trial_index=trial_index,
            resolved_axes=resolved_axes,
            state=state,
        )
        validator = _build_timed_output_validator(
            request=request,
            state=state,
        )
        trials.append(
            _measure_solution_trial(
                request,
                provider,
                validator,
                cache_policy,
                **trial_kwargs,
            )
        )
    return SolutionTimingResult(
        latency_ms=statistics.mean(trial.latency_ms for trial in trials),
        timed_iterations_per_trial=tuple(
            trial.timed_iterations for trial in trials
        ),
        trial_samples_ms=tuple(trial.samples_ms for trial in trials),
        cache_clear_policy=cache_policy,
    )


def _build_iteration_provider(
    *,
    request: WorkloadEvaluationRequest,
    workload: Workload,
    row_index: int,
    trial_index: int,
    resolved_axes: dict[str, int],
    state: _TimedReferenceState,
) -> Callable[[], list[Any]]:
    invocation = 0
    input_hashes: set[str] = set()

    def provide() -> list[Any]:
        nonlocal invocation
        case: ReferenceTimingCase = (
            request.dependencies.reference_client.timing_iteration_case(
                workload_uuid=workload.uuid,
                row_index=row_index,
                trial_index=trial_index,
                iteration_index=invocation,
            )
        )
        invocation += 1
        if case.input_sha256 in input_hashes:
            raise RewardHackError(
                "timing protocol requires unique input values for every "
                "candidate invocation",
            )
        input_hashes.add(case.input_sha256)
        state.inputs = case.inputs
        if case.outputs:
            raise RewardHackError(
                "trusted worker disclosed timing reference outputs",
            )
        outputs = (
            allocate_outputs(request.definition, resolved_axes, request.device)
            if request.destination_passing_style
            else []
        )
        return [*case.inputs, *outputs]

    return provide


def _measure_solution_trial(
    request: WorkloadEvaluationRequest,
    argument_provider: Callable[[], list[Any]],
    validator: Callable[[list[Any], Any], None],
    cache_clear_policy: CacheClearPolicy | None = None,
    *,
    workload: Workload | None = None,
) -> TimingResult:
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
    timed_fn = _device_guarded_callable(request, timed_fn)
    timing = measure_latency(
        timed_fn,
        [],
        [],
        request.device,
        warmup=REPLAY_WARMUP_RUNS if replay else config.warmup_runs,
        rep=REPLAY_EVIDENCE_ITERATIONS if replay else config.iterations,
        min_measurement_time_seconds=(
            None if replay else config.min_measurement_time_seconds
        ),
        callbacks=TimingCallbacks(
            argument_provider=argument_provider,
            validator=validator,
        ),
        cache_clear_policy=cache_clear_policy,
    )
    if timing.failure is not None:
        raise RuntimeError(timing.failure)
    return timing


def _device_guarded_callable(
    request: WorkloadEvaluationRequest,
    fn: Callable[..., Any],
) -> Callable[..., Any]:
    """Reject host-resident outputs before the timed invocation completes."""
    if request.device.split(":", maxsplit=1)[0] != "cuda":
        return fn
    target = torch.device(request.device)

    def run(*args: Any) -> Any:
        result = fn(*args)
        values = (
            args[len(request.definition.inputs) :]
            if request.destination_passing_style
            else _result_tensors(result)
        )
        if not values or any(value.device != target for value in values):
            raise RewardHackError(
                "candidate outputs must reside on the timed GPU before "
                "the end event",
            )
        return result

    return run


def _result_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [
            tensor
            for item in value.values()
            for tensor in _result_tensors(item)
        ]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _result_tensors(item)]
    return []


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
    state: _TimedReferenceState,
) -> Callable[[list[Any], Any], None]:
    def validate(args: list[Any], result: Any) -> None:
        request.dependencies.check_integrity(
            request.dependencies.integrity_snapshot,
            request.dependencies.driver_globals,
        )
        actual = _timed_outputs(request, state.inputs, args, result)
        try:
            request.dependencies.reference_client.validate_timing_outputs(
                actual,
            )
        except ReferenceExecutionError as exc:
            raise RewardHackError(
                f"timed invocation failed trusted validation: {exc}; "
                "correctness and timing phases must execute identical behavior",
            ) from exc

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
