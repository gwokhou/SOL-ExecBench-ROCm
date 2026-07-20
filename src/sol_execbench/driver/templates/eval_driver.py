#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SOL ExecBench evaluation driver.

Self-contained script written to the GPU staging directory.
Evaluates a user solution and outputs JSONL Trace objects to stdout.
All non-JSON output (library messages, Triton JIT logs) goes to stderr.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ── Redirect stdout → stderr BEFORE importing torch/triton ──────────────────
# Saves the original stdout fd so we can print JSON to it later.
_real_stdout_fd = os.dup(1)
_real_stdout = os.fdopen(_real_stdout_fd, "w", buffering=1)
os.dup2(2, 1)  # fd 1 now points at stderr
sys.stdout = open(1, "w", buffering=1, closefd=False)

import torch  # noqa: E402 — must come after redirect

# ── Staging directory ────────────────────────────────────────────────────────
STAGING_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(STAGING_DIR))

# ── Imports from sol_execbench runtime API ───────────────────────────────────
from sol_execbench.driver.eval_runtime_api import (  # noqa: E402,F401
    BenchmarkConfig,
    Definition,
    EvaluationDependencies,
    EvaluationStatus,
    Solution,
    Trace,
    Workload,
    WorkloadEvaluationRequest,
    allocate_outputs,
    call_and_collect_outputs,
    check_eval_integrity,
    check_lazy_outputs,
    check_monkey_patch,
    check_runtime_integrity,
    check_thread_injection,
    compute_error_stats,
    connect_reference_worker,
    dtype_str_to_torch_dtype,
    emit_trace_jsonl,
    evaluate_workloads,
    load_staged_problem,
    load_user_function,
    make_eval,
    measure_latency,
    review_solution_sources,
    run_reward_hack_check,
    snapshot_runtime_integrity,
)

# ── Load problem ─────────────────────────────────────────────────────────────
definition_dict, _workload_dicts = load_staged_problem(STAGING_DIR)

# ── Load config ───────────────────────────────────────────────────────────────
_config_path = STAGING_DIR / "config.json"
bench_config = (
    BenchmarkConfig(**json.loads(_config_path.read_text()))
    if _config_path.exists()
    else BenchmarkConfig()
)

# ── Parse definition ──────────────────────────────────────────────────────────
definition = Definition(**definition_dict)
workloads = [Workload(**w) for w in _workload_dicts]

# ── Parse solution ────────────────────────────────────────────────────────────
_solution = Solution(**json.loads((STAGING_DIR / "solution.json").read_text()))
_solution_name = _solution.name
_entry_point = _solution.spec.entry_point
_dps = _solution.spec.destination_passing_style

# ── Device and output metadata ───────────────────────────────────────────────
_device = "cuda:0" if torch.cuda.is_available() else "cpu"
_output_names = list(definition.outputs.keys())
_output_dtypes_torch = {
    k: dtype_str_to_torch_dtype(v.dtype) for k, v in definition.outputs.items()
}

# The trusted reference implementation lives in a distinct process.  Connect
# before importing candidate code, then scrub private-pipe credentials from the
# environment inside the client constructor.
_reference_client = connect_reference_worker(device=_device)

# ── Static source review before user-code import ────────────────────────────
_source_review = review_solution_sources(
    _solution,
    output_dtypes=_output_dtypes_torch,
)
if _source_review.blocked:
    _static_msg = _source_review.format_blocking_message()
    for _wl in workloads:
        _trace = Trace(
            definition=definition.name,
            solution=_solution_name,
            workload=_wl,
            evaluation=make_eval(
                EvaluationStatus.REWARD_HACK,
                _device,
                None,
                extra_msg=_static_msg,
            ),
        )
        emit_trace_jsonl(_trace, _real_stdout)
    _reference_client.close()
    sys.exit(0)

# ── Integrity snapshot (before user code import) ─────────────────────────────
# Capture id() of every function that affects measurement or correctness.
# Checked after user code import and after each user_fn() call.
_integrity_snapshot = snapshot_runtime_integrity(globals())
_check_integrity = check_runtime_integrity

# ── Evaluate each workload ────────────────────────────────────────────────────
# evaluate_workloads emits traces via emit_trace_jsonl, which uses allow_nan=False.
try:
    # Resolve candidate code only after the trusted reference channel exists.
    # The reference implementation itself is never imported in this process.
    user_fn = load_user_function(_solution, STAGING_DIR)
    evaluate_workloads(
        WorkloadEvaluationRequest(
            definition=definition,
            workloads=workloads,
            solution_name=_solution_name,
            device=_device,
            output_names=_output_names,
            output_dtypes_torch=_output_dtypes_torch,
            bench_config=bench_config,
            destination_passing_style=_dps,
            dependencies=EvaluationDependencies(
                reference_client=_reference_client,
                user_fn=user_fn,
                integrity_snapshot=_integrity_snapshot,
                check_integrity=_check_integrity,
                driver_globals=globals(),
                real_stdout=_real_stdout,
            ),
        ),
    )
finally:
    _reference_client.close()

# TorchInductor and ROCm runtimes can leave non-daemon worker threads alive after
# all benchmark traces have been emitted.  The driver is a one-shot subprocess,
# so flush the trace stream and terminate explicitly instead of letting teardown
# hang validation jobs. Profiler-backed timing runs need normal interpreter
# teardown so profiler finalizers can write trace artifacts.
try:
    _real_stdout.flush()
    sys.stderr.flush()
    sys.stdout.flush()
except Exception:
    pass
if os.environ.get("SOL_EXECBENCH_GRACEFUL_EXIT") == "1":
    sys.exit(0)
os._exit(0)
