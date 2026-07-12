# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Workload orchestration used by the staged evaluation driver."""

from __future__ import annotations

from sol_execbench.core.bench.clock_lock import are_clocks_locked
from sol_execbench.core.bench.evaluation_requests import WorkloadEvaluationRequest
from sol_execbench.core.bench.eval_correctness import set_evaluation_seed
from sol_execbench.core.bench.eval_trace_helpers import WorkloadTraceEmitter
from sol_execbench.core.bench.eval_workload_execution import evaluate_one_workload
from sol_execbench.core.bench.reward_hack import RewardHackDetected
from sol_execbench.core.data.trace import EvaluationStatus


def evaluate_workloads(request: WorkloadEvaluationRequest) -> None:
    """Validate process preconditions, then evaluate each requested workload."""
    clocks_locked = are_clocks_locked()
    clock_status = None
    if request.bench_config.lock_clocks:
        clock_status = "Clocks locked: yes" if clocks_locked else "Clocks locked: no"
    emitter = WorkloadTraceEmitter(
        definition=request.definition,
        solution_name=request.solution_name,
        device=request.device,
        clock_status_msg=clock_status,
        real_stdout=request.dependencies.real_stdout,
    )
    if not _preflight_succeeds(request, emitter, clocks_locked):
        return
    set_evaluation_seed(request.bench_config.seed)
    custom_inputs_fn = (
        request.dependencies.ref_namespace.get(
            request.definition.custom_inputs_entrypoint
        )
        if request.definition.custom_inputs_entrypoint
        else None
    )
    for row_index, workload in enumerate(request.workloads):
        evaluate_one_workload(request, emitter, row_index, workload, custom_inputs_fn)


def _preflight_succeeds(
    request: WorkloadEvaluationRequest,
    emitter: WorkloadTraceEmitter,
    clocks_locked: bool,
) -> bool:
    try:
        request.dependencies.check_integrity(
            request.dependencies.integrity_snapshot,
            request.dependencies.driver_globals,
        )
    except RewardHackDetected as integrity_err:
        emitter.emit_status_for_workloads(
            request.workloads,
            EvaluationStatus.REWARD_HACK,
            extra_msg=str(integrity_err),
        )
        return False
    if request.bench_config.lock_clocks and not clocks_locked:
        emitter.emit_status_for_workloads(
            request.workloads,
            EvaluationStatus.RUNTIME_ERROR,
            extra_msg="lock_clocks=True but GPU clocks are not locked on this server",
        )
        return False
    return True


__all__ = ["evaluate_workloads"]
