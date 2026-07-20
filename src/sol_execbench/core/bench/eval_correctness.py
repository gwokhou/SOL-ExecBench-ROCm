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
from sol_execbench.core.bench.eval_runtime import run_reward_hack_check
from sol_execbench.core.bench.evaluation_requests import WorkloadEvaluationRequest
from sol_execbench.core.bench.eval_trace_helpers import WorkloadTraceEmitter
from sol_execbench.core.bench.reference_protocol import (
    ReferenceExecutionError,
    ReferenceFailureKind,
    ReferenceProtocolError,
)
from sol_execbench.core.bench.reward_hack import check_lazy_outputs
from sol_execbench.core.bench.utils import call_and_collect_outputs
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


def _prepare_framework_thread_baseline(
    user_fn: Callable[..., Any], device: str
) -> None:
    """Start trusted Torch compiler workers before sampling candidate threads."""
    if not hasattr(user_fn, "_torchdynamo_orig_callable"):
        return
    try:
        compiled_identity = torch.compile(lambda value: value + 0)
        compiled_identity(torch.zeros(1, device=device))
    except Exception:
        return


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
    request: WorkloadEvaluationRequest,
    workload: Workload,
    row_index: int,
    emitter: WorkloadTraceEmitter,
) -> CorrectnessRoundsResult:
    definition = request.definition
    resolved_axes = definition.get_resolved_axes_values(workload.axes)
    dependencies = request.dependencies
    inputs = None
    _prepare_framework_thread_baseline(dependencies.user_fn, request.device)
    threads_before = threading.active_count()
    correctness = Correctness()

    for round_index in range(10):
        try:
            case = dependencies.reference_client.correctness_case(
                workload_uuid=workload.uuid,
                row_index=row_index,
                round_index=round_index,
            )
            inputs = case.inputs
            ref_outputs = case.outputs
        except ReferenceExecutionError as exc:
            status = (
                EvaluationStatus.RUNTIME_ERROR
                if exc.kind is ReferenceFailureKind.INPUT_GENERATION
                else EvaluationStatus.INVALID_REFERENCE
            )
            emitter.emit_status(
                workload,
                status,
                extra_msg=str(exc),
            )
            return CorrectnessRoundsResult(True, inputs, threads_before, correctness)
        except ReferenceProtocolError as exc:
            emitter.emit_status(
                workload,
                EvaluationStatus.RUNTIME_ERROR,
                extra_msg=f"Trusted reference IPC failed: {exc}",
            )
            return CorrectnessRoundsResult(True, inputs, threads_before, correctness)

        try:
            user_outputs = call_and_collect_outputs(
                dependencies.user_fn,
                inputs,
                destination_passing_style=request.destination_passing_style,
                definition=definition,
                resolved_axes=resolved_axes,
                device=request.device,
                output_names=request.output_names,
                output_dtypes=request.output_dtypes_torch,
            )
        except Exception as exc:
            emitter.emit_status(
                workload,
                EvaluationStatus.RUNTIME_ERROR,
                extra_msg=f"User function failed: {exc}\n{traceback.format_exc()}",
            )
            return CorrectnessRoundsResult(True, inputs, threads_before, correctness)

        if emit_reward_hack_if_detected(
            emitter=emitter,
            workload=workload,
            check_fn=dependencies.check_integrity,
            args=(
                dependencies.integrity_snapshot,
                dependencies.driver_globals,
            ),
        ):
            return CorrectnessRoundsResult(True, inputs, threads_before, correctness)

        if round_index == 0:
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
