# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Workload execution loop used by the staged eval driver template."""

from __future__ import annotations

import gc
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TextIO

import torch

from sol_execbench.core.bench.clock_lock import are_clocks_locked
from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.bench.eval_correctness import (
    emit_reward_hack_if_detected,
    run_correctness_rounds,
    set_evaluation_seed,
)
from sol_execbench.core.bench.eval_runtime import (
    measure_latency,
    measure_reference_latency,
)
from sol_execbench.core.bench.eval_trace_helpers import WorkloadTraceEmitter
from sol_execbench.core.bench.io import (
    allocate_outputs,
    load_safetensors,
)
from sol_execbench.core.bench.reward_hack import (
    RewardHackDetected,
    check_monkey_patch,
    check_thread_injection,
)
from sol_execbench.core.data.trace import EvaluationStatus, Performance


def evaluate_workloads(
    *,
    definition: Any,
    workloads: list[Any],
    solution_name: str,
    device: str,
    output_names: list[str],
    output_dtypes_torch: dict[str, torch.dtype],
    bench_config: BenchmarkConfig,
    ref_namespace: dict[str, Any],
    ref_fn: Callable[..., Any],
    user_fn: Callable[..., Any],
    destination_passing_style: bool,
    safetensors_roots: Iterable[Path],
    integrity_snapshot: dict[str, int],
    check_integrity: Callable[[dict[str, int], dict[str, Any]], None],
    driver_globals: dict[str, Any],
    real_stdout: TextIO,
) -> None:
    """Evaluate all workloads and emit Trace JSONL records."""
    clocks_locked = are_clocks_locked()
    clock_status_msg = None
    if bench_config.lock_clocks:
        clock_status_msg = "Clocks locked: yes" if clocks_locked else "Clocks locked: no"

    emitter = WorkloadTraceEmitter(
        definition=definition,
        solution_name=solution_name,
        device=device,
        clock_status_msg=clock_status_msg,
        real_stdout=real_stdout,
    )

    try:
        check_integrity(integrity_snapshot, driver_globals)
    except RewardHackDetected as integrity_err:
        emitter.emit_status_for_workloads(
            workloads,
            EvaluationStatus.REWARD_HACK,
            extra_msg=str(integrity_err),
        )
        return

    if bench_config.lock_clocks and not clocks_locked:
        reject_msg = "lock_clocks=True but GPU clocks are not locked on this server"
        emitter.emit_status_for_workloads(
            workloads,
            EvaluationStatus.RUNTIME_ERROR,
            extra_msg=reject_msg,
        )
        return

    set_evaluation_seed(bench_config.seed)
    inputs = None
    timing_outputs = None
    custom_inputs_fn = (
        ref_namespace.get(definition.custom_inputs_entrypoint)
        if definition.custom_inputs_entrypoint
        else None
    )

    for row_index, workload in enumerate(workloads):
        inputs = None
        timing_outputs = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        resolved_axes = definition.get_resolved_axes_values(workload.axes)

        safe_tensors: dict = {}
        if any(value.type == "safetensors" for value in workload.inputs.values()):
            try:
                safe_tensors = load_safetensors(
                    definition, workload, list(safetensors_roots)
                )
            except Exception as exc:
                emitter.emit_status(
                    workload,
                    EvaluationStatus.RUNTIME_ERROR,
                    extra_msg=f"Failed to load safetensors: {exc}",
                )
                continue

        correctness_result = run_correctness_rounds(
            definition=definition,
            workload=workload,
            row_index=row_index,
            device=device,
            safe_tensors=safe_tensors or None,
            custom_inputs_fn=custom_inputs_fn,
            ref_fn=ref_fn,
            user_fn=user_fn,
            destination_passing_style=destination_passing_style,
            resolved_axes=resolved_axes,
            output_names=output_names,
            output_dtypes_torch=output_dtypes_torch,
            integrity_snapshot=integrity_snapshot,
            check_integrity=check_integrity,
            driver_globals=driver_globals,
            emitter=emitter,
        )
        if correctness_result.failed:
            continue

        inputs = correctness_result.inputs
        threads_before = correctness_result.threads_before
        correctness = correctness_result.correctness

        if emit_reward_hack_if_detected(
            emitter=emitter,
            workload=workload,
            check_fn=check_monkey_patch,
        ):
            continue

        assert inputs is not None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            timing_outputs = (
                allocate_outputs(definition, resolved_axes, device)
                if destination_passing_style
                else []
            )
            sol_timing = measure_latency(
                user_fn,
                inputs,
                timing_outputs,
                device,
                warmup=bench_config.warmup_runs,
                rep=bench_config.iterations,
            )
            if sol_timing.failure is not None:
                raise RuntimeError(sol_timing.failure)
            sol_latency_ms = sol_timing.latency_ms
        except Exception as exc:
            emitter.emit_status(
                workload,
                EvaluationStatus.RUNTIME_ERROR,
                extra_msg=f"Timing failed: {exc}",
            )
            continue

        if emit_reward_hack_if_detected(
            emitter=emitter,
            workload=workload,
            check_fn=check_thread_injection,
            args=(threads_before, threading.active_count()),
        ):
            continue

        ref_latency_ms = 0.0
        ref_timing_failure = None
        if bench_config.benchmark_reference:
            ref_timing = measure_reference_latency(
                ref_fn,
                inputs,
                device,
                warmup=bench_config.warmup_runs,
                rep=bench_config.iterations,
            )
            ref_latency_ms = ref_timing.latency_ms
            ref_timing_failure = ref_timing.failure

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
            extra_msg=ref_timing_failure,
        )
