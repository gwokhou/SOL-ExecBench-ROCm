# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Correctness and timing stages for one generated-driver workload."""

from __future__ import annotations

import gc
import threading
from collections.abc import Callable
from typing import Any

import torch

from sol_execbench.core.bench.evaluation_requests import WorkloadEvaluationRequest
from sol_execbench.core.bench.eval_correctness import (
    emit_reward_hack_if_detected,
    run_correctness_rounds,
)
from sol_execbench.core.bench.eval_timing import (
    load_required_safetensors,
    measure_optional_reference_latency,
    measure_solution_latency,
)
from sol_execbench.core.bench.eval_trace_helpers import WorkloadTraceEmitter
from sol_execbench.core.bench.reward_hack import (
    check_monkey_patch,
    check_thread_injection,
)
from sol_execbench.core.data.trace import EvaluationStatus, Performance
from sol_execbench.core.data.workload import Workload


def evaluate_one_workload(
    request: WorkloadEvaluationRequest,
    emitter: WorkloadTraceEmitter,
    row_index: int,
    workload: Workload,
    custom_inputs_fn: Callable[..., Any] | None,
) -> None:
    """Run load, correctness, integrity, and timing stages for one workload."""
    _release_device_cache()
    resolved_axes = request.definition.get_resolved_axes_values(workload.axes)
    tensors_loaded, safe_tensors = _load_safetensors(request, emitter, workload)
    if not tensors_loaded:
        return
    deps = request.dependencies
    correctness_result = run_correctness_rounds(
        definition=request.definition,
        workload=workload,
        row_index=row_index,
        device=request.device,
        safe_tensors=safe_tensors or None,
        custom_inputs_fn=custom_inputs_fn,
        ref_fn=deps.ref_fn,
        user_fn=deps.user_fn,
        destination_passing_style=request.destination_passing_style,
        resolved_axes=resolved_axes,
        output_names=request.output_names,
        output_dtypes_torch=request.output_dtypes_torch,
        integrity_snapshot=deps.integrity_snapshot,
        check_integrity=deps.check_integrity,
        driver_globals=deps.driver_globals,
        emitter=emitter,
    )
    if correctness_result.failed or emit_reward_hack_if_detected(
        emitter=emitter, workload=workload, check_fn=check_monkey_patch
    ):
        return
    assert correctness_result.inputs is not None
    assert correctness_result.threads_before is not None
    _measure_and_emit(
        request,
        emitter,
        workload,
        resolved_axes,
        correctness_result.inputs,
        correctness_result.threads_before,
        correctness_result.correctness,
    )


def _load_safetensors(
    request: WorkloadEvaluationRequest,
    emitter: WorkloadTraceEmitter,
    workload: Workload,
) -> tuple[bool, dict[str, Any] | None]:
    try:
        return (
            True,
            load_required_safetensors(
                definition=request.definition,
                workload=workload,
                safetensors_roots=list(request.safetensors_roots),
            ),
        )
    except Exception as exc:
        emitter.emit_status(
            workload,
            EvaluationStatus.RUNTIME_ERROR,
            extra_msg=f"Failed to load safetensors: {exc}",
        )
        return False, None


def _measure_and_emit(
    request: WorkloadEvaluationRequest,
    emitter: WorkloadTraceEmitter,
    workload: Workload,
    resolved_axes: dict[str, int],
    inputs: list[Any],
    threads_before: int,
    correctness: Any,
) -> None:
    _release_device_cache()
    try:
        sol_latency_ms = measure_solution_latency(
            definition=request.definition,
            resolved_axes=resolved_axes,
            device=request.device,
            destination_passing_style=request.destination_passing_style,
            user_fn=request.dependencies.user_fn,
            inputs=inputs,
            warmup=request.bench_config.warmup_runs,
            rep=request.bench_config.iterations,
            min_measurement_time_seconds=request.bench_config.min_measurement_time_seconds,
        )
    except Exception as exc:
        emitter.emit_status(
            workload, EvaluationStatus.RUNTIME_ERROR, extra_msg=f"Timing failed: {exc}"
        )
        return
    if emit_reward_hack_if_detected(
        emitter=emitter,
        workload=workload,
        check_fn=check_thread_injection,
        args=(threads_before, threading.active_count()),
    ):
        return
    ref_latency_ms, failure = measure_optional_reference_latency(
        benchmark_reference=request.bench_config.benchmark_reference,
        ref_fn=request.dependencies.ref_fn,
        inputs=inputs,
        device=request.device,
        warmup=request.bench_config.warmup_runs,
        rep=request.bench_config.iterations,
        min_measurement_time_seconds=request.bench_config.min_measurement_time_seconds,
    )
    speedup = ref_latency_ms / sol_latency_ms if sol_latency_ms > 0 else 0.0
    emitter.emit_status(
        workload,
        EvaluationStatus.PASSED,
        correctness=correctness,
        performance=Performance(
            latency_ms=sol_latency_ms,
            reference_latency_ms=ref_latency_ms,
            speedup_factor=speedup,
        ),
        extra_msg=failure,
    )


def _release_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = ["evaluate_one_workload"]
