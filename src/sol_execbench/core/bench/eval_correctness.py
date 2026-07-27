# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Correctness rounds for staged workload evaluation."""

from __future__ import annotations

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
from sol_execbench.core.bench.eval_trace_helpers import WorkloadTraceEmitter
from sol_execbench.core.bench.evaluation_requests import (
    WorkloadEvaluationRequest,
)
from sol_execbench.core.bench.reference_protocol import (
    ReferenceCase,
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
    """Outcome and retained inputs from correctness rounds."""

    failed: bool
    inputs: list[Any] | None
    correctness: Correctness


def set_evaluation_seed(seed: int) -> None:
    """Set the deterministic seed used for evaluation."""
    set_seed(seed)


def _prepare_framework_thread_baseline(
    user_fn: Callable[..., Any],
    device: str,
) -> None:
    """Start trusted compiler workers before sampling candidate threads."""
    if not hasattr(user_fn, "_torchdynamo_orig_callable"):
        return
    try:
        compiled_identity = torch.compile(lambda value: value + 0)
        compiled_identity(torch.zeros(1, device=device))
    except Exception:  # noqa: BLE001 -- optional compiler prewarm
        return


def emit_reward_hack_if_detected(
    *,
    emitter: WorkloadTraceEmitter,
    workload: Workload,
    check_fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    suppress_errors: bool = False,
) -> bool:
    """Run one reward-hack check and emit a failure trace if detected."""
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
    """Execute correctness rounds and emit terminal failures."""
    definition = request.definition
    resolved_axes = definition.get_resolved_axes_values(workload.axes)
    dependencies = request.dependencies
    inputs = None
    _prepare_framework_thread_baseline(dependencies.user_fn, request.device)
    correctness = Correctness()

    for round_index in range(10):
        case = _load_reference_case(
            request,
            workload,
            row_index,
            round_index,
            emitter,
        )
        if case is None:
            return CorrectnessRoundsResult(True, inputs, correctness)
        inputs = case.inputs
        ref_outputs = case.outputs

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
        except Exception as exc:  # noqa: BLE001 -- candidate execution boundary
            emitter.emit_status(
                workload,
                EvaluationStatus.RUNTIME_ERROR,
                extra_msg=_candidate_failure_message(exc),
            )
            return CorrectnessRoundsResult(True, inputs, correctness)

        if emit_reward_hack_if_detected(
            emitter=emitter,
            workload=workload,
            check_fn=dependencies.check_integrity,
            args=(
                dependencies.integrity_snapshot,
                dependencies.driver_globals,
            ),
        ):
            return CorrectnessRoundsResult(True, inputs, correctness)

        if round_index == 0 and _first_round_failed(
            emitter,
            workload,
            ref_outputs,
            user_outputs,
        ):
            return CorrectnessRoundsResult(True, inputs, correctness)
        correctness, numerically_wrong = _compare_outputs(
            ref_outputs,
            user_outputs,
            workload,
            correctness,
        )
        if numerically_wrong:
            emitter.emit_status(
                workload,
                EvaluationStatus.INCORRECT_NUMERICAL,
                correctness=correctness,
            )
            return CorrectnessRoundsResult(True, inputs, correctness)

    return CorrectnessRoundsResult(False, inputs, correctness)


def _candidate_failure_message(exc: Exception) -> str:
    return f"User function failed: {exc}\n{traceback.format_exc()}"


def _load_reference_case(
    request: WorkloadEvaluationRequest,
    workload: Workload,
    row_index: int,
    round_index: int,
    emitter: WorkloadTraceEmitter,
) -> ReferenceCase | None:
    try:
        return request.dependencies.reference_client.correctness_case(
            workload_uuid=workload.uuid,
            row_index=row_index,
            round_index=round_index,
        )
    except ReferenceExecutionError as exc:
        status = (
            EvaluationStatus.RUNTIME_ERROR
            if exc.kind is ReferenceFailureKind.INPUT_GENERATION
            else EvaluationStatus.INVALID_REFERENCE
        )
        emitter.emit_status(workload, status, extra_msg=str(exc))
    except ReferenceProtocolError as exc:
        emitter.emit_status(
            workload,
            EvaluationStatus.RUNTIME_ERROR,
            extra_msg=f"Trusted reference IPC failed: {exc}",
        )
    return None


def _first_round_failed(
    emitter: WorkloadTraceEmitter,
    workload: Workload,
    reference_outputs: list[torch.Tensor],
    user_outputs: list[torch.Tensor],
) -> bool:
    if emit_reward_hack_if_detected(
        emitter=emitter,
        workload=workload,
        check_fn=check_lazy_outputs,
        args=(user_outputs,),
    ):
        return True
    issue = check_output_shape_dtype(reference_outputs, user_outputs)
    if issue is None:
        return False
    emitter.emit_status(workload, issue)
    return True


def _compare_outputs(
    reference_outputs: list[torch.Tensor],
    user_outputs: list[torch.Tensor],
    workload: Workload,
    correctness: Correctness,
) -> tuple[Correctness, bool]:
    for reference, user_output in zip(
        reference_outputs,
        user_outputs,
        strict=True,
    ):
        current, exceeds = compute_error_stats(
            user_output,
            reference,
            workload.tolerance,
        )
        if (
            current.max_absolute_error > correctness.max_absolute_error
            or current.has_nan
            or current.has_inf
            and not correctness.has_nan
        ):
            correctness = current
        if exceeds:
            return correctness, True
    return correctness, False
