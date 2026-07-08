# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Workload execution loop used by the staged eval driver template."""

from __future__ import annotations

import gc
import threading
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TextIO

import torch

from sol_execbench.core import (
    Correctness,
    Definition,
    EvaluationStatus,
    Performance,
    Trace,
    Workload,
)
from sol_execbench.core.bench.clock_lock import are_clocks_locked
from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.bench.correctness import (
    check_output_shape_dtype,
    compute_error_stats,
    set_seed,
)
from sol_execbench.core.bench.eval_output_integrity import stable_reference_outputs
from sol_execbench.core.bench.eval_runtime import (
    emit_trace_jsonl,
    measure_latency,
    measure_reference_latency,
    run_reward_hack_check,
)
from sol_execbench.core.bench.io import (
    CustomInputGenerationError,
    allocate_outputs,
    gen_inputs,
    load_safetensors,
)
from sol_execbench.core.bench.reward_hack import (
    RewardHackDetected,
    check_lazy_outputs,
    check_monkey_patch,
    check_thread_injection,
)
from sol_execbench.core.bench.utils import call_and_collect_outputs, make_eval


def evaluate_workloads(
    *,
    definition: Definition,
    workloads: list[Workload],
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

    def emit(trace: Trace) -> None:
        emit_trace_jsonl(trace, real_stdout)

    def make_driver_eval(
        status,
        log_path,
        *,
        correctness=None,
        performance=None,
        extra_msg=None,
    ):
        parts = [part for part in (clock_status_msg, extra_msg) if part]
        return make_eval(
            status,
            device,
            log_path,
            correctness=correctness,
            performance=performance,
            extra_msg="\n".join(parts) or None,
        )

    def reward_hack_check(workload, check_fn, *args, suppress_errors=False):
        message = run_reward_hack_check(
            check_fn,
            *args,
            suppress_errors=suppress_errors,
        )
        if message is not None:
            emit(
                Trace(
                    definition=definition.name,
                    solution=solution_name,
                    workload=workload,
                    evaluation=make_driver_eval(
                        EvaluationStatus.REWARD_HACK, None, extra_msg=message
                    ),
                )
            )
            return True
        return False

    try:
        check_integrity(integrity_snapshot, driver_globals)
    except RewardHackDetected as integrity_err:
        for workload in workloads:
            emit(
                Trace(
                    definition=definition.name,
                    solution=solution_name,
                    workload=workload,
                    evaluation=make_driver_eval(
                        EvaluationStatus.REWARD_HACK,
                        None,
                        extra_msg=str(integrity_err),
                    ),
                )
            )
        return

    if bench_config.lock_clocks and not clocks_locked:
        reject_msg = "lock_clocks=True but GPU clocks are not locked on this server"
        for workload in workloads:
            emit(
                Trace(
                    definition=definition.name,
                    solution=solution_name,
                    workload=workload,
                    evaluation=make_driver_eval(
                        EvaluationStatus.RUNTIME_ERROR, None, extra_msg=reject_msg
                    ),
                )
            )
        return

    set_seed(bench_config.seed)
    inputs = None
    ref_outputs = None
    user_outputs = None
    timing_outputs = None
    custom_inputs_fn = (
        ref_namespace.get(definition.custom_inputs_entrypoint)
        if definition.custom_inputs_entrypoint
        else None
    )

    for row_index, workload in enumerate(workloads):
        inputs = None
        ref_outputs = None
        user_outputs = None
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
                emit(
                    Trace(
                        definition=definition.name,
                        solution=solution_name,
                        workload=workload,
                        evaluation=make_driver_eval(
                            EvaluationStatus.RUNTIME_ERROR,
                            None,
                            extra_msg=f"Failed to load safetensors: {exc}",
                        ),
                    )
                )
                continue

        correctness_failed = False
        threads_before = None
        correctness = Correctness()

        for round_index in range(10):
            try:
                inputs = gen_inputs(
                    definition,
                    workload,
                    device=device,
                    safe_tensors=safe_tensors or None,
                    custom_inputs_fn=custom_inputs_fn,
                    row_index=row_index,
                )
            except CustomInputGenerationError as exc:
                emit(
                    Trace(
                        definition=definition.name,
                        solution=solution_name,
                        workload=workload,
                        evaluation=make_driver_eval(
                            EvaluationStatus.RUNTIME_ERROR,
                            None,
                            extra_msg=(
                                f"{exc.failure_class}: {exc}\n"
                                f"{exc.provenance.log_text()}"
                            ),
                        ),
                    )
                )
                correctness_failed = True
                break
            except Exception as exc:
                emit(
                    Trace(
                        definition=definition.name,
                        solution=solution_name,
                        workload=workload,
                        evaluation=make_driver_eval(
                            EvaluationStatus.RUNTIME_ERROR,
                            None,
                            extra_msg=f"gen_inputs failed: {exc}",
                        ),
                    )
                )
                correctness_failed = True
                break

            try:
                ref_outputs = call_and_collect_outputs(
                    ref_fn,
                    inputs,
                    destination_passing_style=False,
                    definition=definition,
                    resolved_axes=resolved_axes,
                    device=device,
                    output_names=output_names,
                    output_dtypes=output_dtypes_torch,
                )
                ref_outputs = stable_reference_outputs(ref_outputs, inputs)
            except Exception as exc:
                emit(
                    Trace(
                        definition=definition.name,
                        solution=solution_name,
                        workload=workload,
                        evaluation=make_driver_eval(
                            EvaluationStatus.INVALID_REFERENCE,
                            None,
                            extra_msg=(
                                f"Reference run() failed: {exc}\n"
                                f"{traceback.format_exc()}"
                            ),
                        ),
                    )
                )
                correctness_failed = True
                break

            try:
                user_outputs = call_and_collect_outputs(
                    user_fn,
                    inputs,
                    destination_passing_style=destination_passing_style,
                    definition=definition,
                    resolved_axes=resolved_axes,
                    device=device,
                    output_names=output_names,
                    output_dtypes=output_dtypes_torch,
                )
            except Exception as exc:
                emit(
                    Trace(
                        definition=definition.name,
                        solution=solution_name,
                        workload=workload,
                        evaluation=make_driver_eval(
                            EvaluationStatus.RUNTIME_ERROR,
                            None,
                            extra_msg=f"User function failed: {exc}\n{traceback.format_exc()}",
                        ),
                    )
                )
                correctness_failed = True
                break

            if round_index == 0:
                if reward_hack_check(
                    workload, check_integrity, integrity_snapshot, driver_globals
                ):
                    correctness_failed = True
                    break

                threads_before = threading.active_count()

                if reward_hack_check(workload, check_lazy_outputs, user_outputs):
                    correctness_failed = True
                    break

                shape_dtype_issue = check_output_shape_dtype(ref_outputs, user_outputs)
                if shape_dtype_issue is not None:
                    emit(
                        Trace(
                            definition=definition.name,
                            solution=solution_name,
                            workload=workload,
                            evaluation=make_driver_eval(shape_dtype_issue, None),
                        )
                    )
                    correctness_failed = True
                    break

            numerically_wrong = False
            for ref_out, usr_out in zip(ref_outputs, user_outputs):
                current, exceeds = compute_error_stats(
                    usr_out, ref_out, workload.tolerance
                )
                if current.max_absolute_error > correctness.max_absolute_error:
                    correctness = current
                if current.has_nan:
                    correctness = current
                elif current.has_inf and not correctness.has_nan:
                    correctness = current
                if exceeds:
                    numerically_wrong = True
                    break

            if numerically_wrong:
                emit(
                    Trace(
                        definition=definition.name,
                        solution=solution_name,
                        workload=workload,
                        evaluation=make_driver_eval(
                            EvaluationStatus.INCORRECT_NUMERICAL,
                            None,
                            correctness=correctness,
                        ),
                    )
                )
                correctness_failed = True
                break

            ref_outputs = None
            user_outputs = None

        if correctness_failed:
            continue

        if reward_hack_check(workload, check_monkey_patch):
            continue

        assert inputs is not None
        ref_outputs = None
        user_outputs = None
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
            emit(
                Trace(
                    definition=definition.name,
                    solution=solution_name,
                    workload=workload,
                    evaluation=make_driver_eval(
                        EvaluationStatus.RUNTIME_ERROR,
                        None,
                        extra_msg=f"Timing failed: {exc}",
                    ),
                )
            )
            continue

        if reward_hack_check(
            workload, check_thread_injection, threads_before, threading.active_count()
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
        emit(
            Trace(
                definition=definition.name,
                solution=solution_name,
                workload=workload,
                evaluation=make_driver_eval(
                    EvaluationStatus.PASSED,
                    None,
                    correctness=correctness,
                    performance=Performance(
                        latency_ms=sol_latency_ms,
                        reference_latency_ms=ref_latency_ms,
                        speedup_factor=speedup,
                    ),
                    extra_msg=ref_timing_failure,
                ),
            )
        )
