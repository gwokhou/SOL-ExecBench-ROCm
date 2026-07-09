# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Correctness rounds for staged workload evaluation."""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from sol_execbench.core.bench.correctness import (
    check_output_shape_dtype,
    compute_error_stats,
    set_seed,
)
from sol_execbench.core.bench.eval_output_integrity import stable_reference_outputs
from sol_execbench.core.bench.eval_runtime import run_reward_hack_check
from sol_execbench.core.bench.eval_trace_helpers import WorkloadTraceEmitter
from sol_execbench.core.bench.io import CustomInputGenerationError, gen_inputs
from sol_execbench.core.bench.reward_hack import check_lazy_outputs
from sol_execbench.core.bench.utils import call_and_collect_outputs
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.trace import Correctness, EvaluationStatus
from sol_execbench.core.data.workload import Workload


@dataclass
class CorrectnessRoundsResult:
    failed: bool
    inputs: list[Any] | None
    threads_before: int | None
    correctness: Correctness


def set_evaluation_seed(seed: int) -> None:
    set_seed(seed)


def emit_reward_hack_if_detected(
    *,
    emitter: WorkloadTraceEmitter,
    workload: Workload,
    check_fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    suppress_errors: bool = False,
) -> bool:
    message = run_reward_hack_check(
        check_fn,
        *args,
        suppress_errors=suppress_errors,
    )
    if message is None:
        return False

    emitter.emit_status(
        workload,
        EvaluationStatus.REWARD_HACK,
        extra_msg=message,
    )
    return True


def run_correctness_rounds(
    *,
    definition: Definition,
    workload: Workload,
    row_index: int,
    device: str,
    safe_tensors: dict[str, Any] | None,
    custom_inputs_fn: Callable[..., Any] | None,
    ref_fn: Callable[..., Any],
    user_fn: Callable[..., Any],
    destination_passing_style: bool,
    resolved_axes: dict[str, int],
    output_names: list[str],
    output_dtypes_torch: dict[str, torch.dtype],
    integrity_snapshot: dict[str, int],
    check_integrity: Callable[[dict[str, int], dict[str, Any]], None],
    driver_globals: dict[str, Any],
    emitter: WorkloadTraceEmitter,
) -> CorrectnessRoundsResult:
    inputs = None
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
            emitter.emit_status(
                workload,
                EvaluationStatus.RUNTIME_ERROR,
                extra_msg=(f"{exc.failure_class}: {exc}\n{exc.provenance.log_text()}"),
            )
            return CorrectnessRoundsResult(True, inputs, threads_before, correctness)
        except Exception as exc:
            emitter.emit_status(
                workload,
                EvaluationStatus.RUNTIME_ERROR,
                extra_msg=f"gen_inputs failed: {exc}",
            )
            return CorrectnessRoundsResult(True, inputs, threads_before, correctness)

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
            emitter.emit_status(
                workload,
                EvaluationStatus.INVALID_REFERENCE,
                extra_msg=f"Reference run() failed: {exc}\n{traceback.format_exc()}",
            )
            return CorrectnessRoundsResult(True, inputs, threads_before, correctness)

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
            emitter.emit_status(
                workload,
                EvaluationStatus.RUNTIME_ERROR,
                extra_msg=f"User function failed: {exc}\n{traceback.format_exc()}",
            )
            return CorrectnessRoundsResult(True, inputs, threads_before, correctness)

        if round_index == 0:
            if emit_reward_hack_if_detected(
                emitter=emitter,
                workload=workload,
                check_fn=check_integrity,
                args=(integrity_snapshot, driver_globals),
            ):
                return CorrectnessRoundsResult(
                    True, inputs, threads_before, correctness
                )

            threads_before = threading.active_count()

            if emit_reward_hack_if_detected(
                emitter=emitter,
                workload=workload,
                check_fn=check_lazy_outputs,
                args=(user_outputs,),
            ):
                return CorrectnessRoundsResult(
                    True, inputs, threads_before, correctness
                )

            shape_dtype_issue = check_output_shape_dtype(ref_outputs, user_outputs)
            if shape_dtype_issue is not None:
                emitter.emit_status(workload, shape_dtype_issue)
                return CorrectnessRoundsResult(
                    True, inputs, threads_before, correctness
                )

        numerically_wrong = False
        for ref_out, usr_out in zip(ref_outputs, user_outputs):
            current, exceeds = compute_error_stats(usr_out, ref_out, workload.tolerance)
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
            emitter.emit_status(
                workload,
                EvaluationStatus.INCORRECT_NUMERICAL,
                correctness=correctness,
            )
            return CorrectnessRoundsResult(True, inputs, threads_before, correctness)

    return CorrectnessRoundsResult(False, inputs, threads_before, correctness)
