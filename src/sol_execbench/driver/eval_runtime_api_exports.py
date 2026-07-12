# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Public exports for generated evaluation drivers."""

from __future__ import annotations

from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.bench.correctness import compute_error_stats
from sol_execbench.core.bench.eval_runtime import (
    emit_trace_jsonl,
    load_reference_function,
    load_staged_problem,
    load_user_function,
    measure_latency,
    measure_reference_latency,
    run_reward_hack_check,
)
from sol_execbench.core.bench.evaluation_requests import (
    EvaluationDependencies,
    WorkloadEvaluationRequest,
)
from sol_execbench.core.bench.eval_workload_runner import evaluate_workloads
from sol_execbench.core.bench.io import allocate_outputs, gen_inputs
from sol_execbench.core.bench.reward_hack import (
    check_eval_integrity,
    check_lazy_outputs,
    check_monkey_patch,
    check_thread_injection,
    review_solution_sources,
    snapshot_critical_functions,
)
from sol_execbench.core.bench.utils import call_and_collect_outputs, make_eval
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype
from sol_execbench.core.data.solution import Solution
from sol_execbench.core.data.trace import EvaluationStatus, Trace
from sol_execbench.core.data.workload import Workload

__all__ = [
    "BenchmarkConfig",
    "Definition",
    "EvaluationStatus",
    "EvaluationDependencies",
    "Solution",
    "Trace",
    "Workload",
    "WorkloadEvaluationRequest",
    "allocate_outputs",
    "call_and_collect_outputs",
    "check_eval_integrity",
    "check_lazy_outputs",
    "check_monkey_patch",
    "check_thread_injection",
    "compute_error_stats",
    "dtype_str_to_torch_dtype",
    "emit_trace_jsonl",
    "evaluate_workloads",
    "gen_inputs",
    "load_reference_function",
    "load_staged_problem",
    "load_user_function",
    "make_eval",
    "measure_latency",
    "measure_reference_latency",
    "review_solution_sources",
    "run_reward_hack_check",
    "snapshot_critical_functions",
]
