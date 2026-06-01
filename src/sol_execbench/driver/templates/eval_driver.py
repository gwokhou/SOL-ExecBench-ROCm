#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gc
import json
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional

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

# ── Imports from sol_execbench.core ─────────────────────────────────────────────
from sol_execbench.core.bench.clock_lock import are_clocks_locked  # noqa: E402
from sol_execbench.core.bench.config import BenchmarkConfig  # noqa: E402
from sol_execbench.core.bench.correctness import (  # noqa: E402
    check_output_shape_dtype,
    compute_error_stats,
    set_seed,
)
from sol_execbench.core.bench.io import (  # noqa: E402
    allocate_outputs,
    gen_inputs,
    load_safetensors,
)
from sol_execbench.core.bench.reward_hack import (  # noqa: E402
    RewardHackDetected,
    check_eval_integrity,
    check_lazy_outputs,
    check_monkey_patch,
    check_thread_injection,
    review_solution_sources,
    snapshot_critical_functions,
)
from sol_execbench.core.bench.eval_runtime import (  # noqa: E402
    emit_trace_jsonl,
    load_reference_function,
    load_staged_problem,
    load_user_function,
    measure_latency,
    measure_reference_latency,
    run_reward_hack_check,
)
from sol_execbench.core.bench.utils import (  # noqa: E402
    call_and_collect_outputs,
    make_eval,
)
from sol_execbench.core import (  # noqa: E402
    Correctness,
    Definition,
    EvaluationStatus,
    Performance,
    Solution,
    Trace,
    Workload,
)
from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype  # noqa: E402

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
    sys.exit(0)

# ── Exec reference code ───────────────────────────────────────────────────────
_ref_module, ref_fn = load_reference_function(STAGING_DIR, definition.reference)
ref_namespace = vars(_ref_module)

# ── Integrity snapshot (before user code import) ─────────────────────────────
# Capture id() of every function that affects measurement or correctness.
# Checked after user code import and after each user_fn() call.
_CRITICAL_NAMES = [
    "measure_latency",
    "emit_trace_jsonl",
    "run_reward_hack_check",
    "compute_error_stats",
    "check_monkey_patch",
    "check_lazy_outputs",
    "check_thread_injection",
    "check_eval_integrity",
    "call_and_collect_outputs",
    "gen_inputs",
    "allocate_outputs",
    "make_eval",
]
_integrity_snapshot = snapshot_critical_functions(globals(), _CRITICAL_NAMES)
# Keep a local reference so that patching the name in globals() is ineffective.
_check_integrity = check_eval_integrity

# ── Resolve user function ─────────────────────────────────────────────────────
user_fn = load_user_function(_solution, STAGING_DIR)

# ── Safetensors blob roots ────────────────────────────────────────────────────
# Priority: 1) staging dir (client-inlined blobs), 2) flashinfer-trace directory.
_safetensors_roots = [STAGING_DIR]
_benchmark_dir = os.environ.get("FLASHINFER_TRACE_DIR", None)
if _benchmark_dir:
    _safetensors_roots.append(Path(_benchmark_dir))


# ── Output helper ─────────────────────────────────────────────────────────────
def _emit(trace: Trace) -> None:
    """Write one Trace as strictly valid JSON to the real stdout.

    Uses allow_nan=False so that any unexpected NaN/Inf in the trace dict
    raises ValueError immediately instead of producing the non-standard
    JavaScript literal ``NaN`` which strict JSON parsers silently reject.
    """
    emit_trace_jsonl(trace, _real_stdout)


def _reward_hack_check(workload, check_fn, *args, suppress_errors=False):
    """Run a reward-hack check; emit REWARD_HACK trace and return True if detected."""
    message = run_reward_hack_check(
        check_fn,
        *args,
        suppress_errors=suppress_errors,
    )
    if message is not None:
        _emit(
            Trace(
                definition=definition.name,
                solution=_solution_name,
                workload=workload,
                evaluation=_make_eval(
                    EvaluationStatus.REWARD_HACK, _device, None, extra_msg=message
                ),
            )
        )
        return True
    return False


# ── Evaluate each workload ────────────────────────────────────────────────────
_device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""

# Check whether clocks are locked (set by Docker entrypoint / server startup).
_clocks_locked = are_clocks_locked()

# Only include clock status in trace logs when the user explicitly requested
# clock locking (lock_clocks=True).  When lock_clocks=False (default), the
# status is noise and misleads users into thinking failures are clock-related.
_clock_status_msg: Optional[str] = None
if bench_config.lock_clocks:
    _clock_status_msg = "Clocks locked: yes" if _clocks_locked else "Clocks locked: no"


def _make_eval(
    status, device, log_path, *, correctness=None, performance=None, extra_msg=None
):
    parts = [p for p in (_clock_status_msg, extra_msg) if p]
    return make_eval(
        status,
        device,
        log_path,
        correctness=correctness,
        performance=performance,
        extra_msg="\n".join(parts) or None,
    )


# ── Integrity check after user code import ───────────────────────────────────
# Catches patches applied at import time (e.g. via sys.modules['__main__']).
# Runs once before any workload processing so patched correctness or timing
# functions are never called.
try:
    _check_integrity(_integrity_snapshot, globals())
except RewardHackDetected as _integrity_err:
    for _wl in workloads:
        _emit(
            Trace(
                definition=definition.name,
                solution=_solution_name,
                workload=_wl,
                evaluation=_make_eval(
                    EvaluationStatus.REWARD_HACK,
                    _device,
                    None,
                    extra_msg=str(_integrity_err),
                ),
            )
        )
    sys.exit(0)

# ── Validate clock lock requirement ───────────────────────────────────────────
# If lock_clocks=True but clocks are not locked, reject all workloads.
if bench_config.lock_clocks and not _clocks_locked:
    _reject_msg = "lock_clocks=True but GPU clocks are not locked on this server"
    for _wl in workloads:
        _emit(
            Trace(
                definition=definition.name,
                solution=_solution_name,
                workload=_wl,
                evaluation=_make_eval(
                    EvaluationStatus.RUNTIME_ERROR,
                    _device,
                    None,
                    extra_msg=_reject_msg,
                ),
            )
        )
    sys.exit(0)

set_seed(bench_config.seed)
_inputs = None
_ref_outputs = None
_user_outputs = None
_timing_outputs = None
for _workload in workloads:
    # Free GPU memory held by previous workload's tensors.
    # PyTorch's CUDA caching allocator retains freed blocks.
    # Without this cleanup, memory accumulates across workloads causing OOM.
    _inputs = None
    _ref_outputs = None
    _user_outputs = None
    _timing_outputs = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _resolved_axes = definition.get_resolved_axes_values(_workload.axes)

    # -- Safetensors inputs --
    _safe_tensors: dict = {}
    if any(v.type == "safetensors" for v in _workload.inputs.values()):
        try:
            _safe_tensors = load_safetensors(definition, _workload, _safetensors_roots)
        except Exception as _e:
            _emit(
                Trace(
                    definition=definition.name,
                    solution=_solution_name,
                    workload=_workload,
                    evaluation=_make_eval(
                        EvaluationStatus.RUNTIME_ERROR,
                        _device,
                        None,
                        extra_msg=f"Failed to load safetensors: {_e}",
                    ),
                )
            )
            continue

    # -- Correctness checking (10 rounds with fresh inputs) --
    # Running multiple rounds catches non-deterministic bugs and input-dependent
    # correctness issues.  Structural checks (lazy outputs, shape/dtype) and
    # defense snapshots run on round 0 only; numerical checks run every round.
    _custom_inputs_fn = (
        ref_namespace.get(definition.custom_inputs_entrypoint)
        if definition.custom_inputs_entrypoint
        else None
    )
    _correctness_failed = False
    _threads_before = None
    _correctness = Correctness()

    for _round in range(10):
        # -- Generate inputs --
        try:
            _inputs = gen_inputs(
                definition,
                _workload,
                device=_device,
                safe_tensors=_safe_tensors or None,
                custom_inputs_fn=_custom_inputs_fn,
            )
        except Exception as _e:
            _emit(
                Trace(
                    definition=definition.name,
                    solution=_solution_name,
                    workload=_workload,
                    evaluation=_make_eval(
                        EvaluationStatus.RUNTIME_ERROR,
                        _device,
                        None,
                        extra_msg=f"gen_inputs failed: {_e}",
                    ),
                )
            )
            _correctness_failed = True
            break

        # -- Run reference (always return-value style) to get ground-truth --
        try:
            _ref_outputs = call_and_collect_outputs(
                ref_fn,
                _inputs,
                destination_passing_style=False,
                definition=definition,
                resolved_axes=_resolved_axes,
                device=_device,
                output_names=_output_names,
                output_dtypes=_output_dtypes_torch,
            )
        except Exception as _e:
            _emit(
                Trace(
                    definition=definition.name,
                    solution=_solution_name,
                    workload=_workload,
                    evaluation=_make_eval(
                        EvaluationStatus.INVALID_REFERENCE,
                        _device,
                        None,
                        extra_msg=f"Reference run() failed: {_e}\n{traceback.format_exc()}",
                    ),
                )
            )
            _correctness_failed = True
            break

        # -- Run user function (DPS=True or False per solution spec) --
        # Round 0 also serves as a warmup call: JIT compilers (torch.compile,
        # Triton, CuTe DSL) spawn infrastructure threads on first call.
        try:
            _user_outputs = call_and_collect_outputs(
                user_fn,
                _inputs,
                destination_passing_style=_dps,
                definition=definition,
                resolved_axes=_resolved_axes,
                device=_device,
                output_names=_output_names,
                output_dtypes=_output_dtypes_torch,
            )
        except Exception as _e:
            _emit(
                Trace(
                    definition=definition.name,
                    solution=_solution_name,
                    workload=_workload,
                    evaluation=_make_eval(
                        EvaluationStatus.RUNTIME_ERROR,
                        _device,
                        None,
                        extra_msg=f"User function failed: {_e}\n{traceback.format_exc()}",
                    ),
                )
            )
            _correctness_failed = True
            break

        # -- Round 0 only: integrity, thread snapshot, lazy, shape/dtype --
        if _round == 0:
            if _reward_hack_check(
                _workload, _check_integrity, _integrity_snapshot, globals()
            ):
                _correctness_failed = True
                break

            # Capture stable post-JIT thread count; the check after
            # measure_latency will catch threads injected during timing.
            _threads_before = threading.active_count()

            if _reward_hack_check(_workload, check_lazy_outputs, _user_outputs):
                _correctness_failed = True
                break

            _shape_dtype_issue = check_output_shape_dtype(_ref_outputs, _user_outputs)

            if _shape_dtype_issue is not None:
                _emit(
                    Trace(
                        definition=definition.name,
                        solution=_solution_name,
                        workload=_workload,
                        evaluation=_make_eval(_shape_dtype_issue, _device, None),
                    )
                )
                _correctness_failed = True
                break

        # -- Numerical correctness check (every round) --
        _numerically_wrong = False
        for _ref_out, _usr_out in zip(_ref_outputs, _user_outputs):
            _c, _exceeds = compute_error_stats(_usr_out, _ref_out, _workload.tolerance)
            if _c.max_absolute_error > _correctness.max_absolute_error:
                _correctness = _c
            if _c.has_nan:
                _correctness = _c
            elif _c.has_inf and not _correctness.has_nan:
                _correctness = _c
            if _exceeds:
                _numerically_wrong = True
                break

        if _numerically_wrong:
            _emit(
                Trace(
                    definition=definition.name,
                    solution=_solution_name,
                    workload=_workload,
                    evaluation=_make_eval(
                        EvaluationStatus.INCORRECT_NUMERICAL,
                        _device,
                        None,
                        correctness=_correctness,
                    ),
                )
            )
            _correctness_failed = True
            break

    if _correctness_failed:
        continue

    # -- Monkey-patch defense before timing --
    if _reward_hack_check(_workload, check_monkey_patch):
        continue

    assert _inputs is not None

    # -- User latency measurement --
    try:
        _timing_outputs = (
            allocate_outputs(definition, _resolved_axes, _device) if _dps else []
        )
        _sol_timing = measure_latency(
            user_fn,
            _inputs,
            _timing_outputs,
            _device,
            warmup=bench_config.warmup_runs,
            rep=bench_config.iterations,
        )
        if _sol_timing.failure is not None:
            raise RuntimeError(_sol_timing.failure)
        _sol_latency_ms = _sol_timing.latency_ms
    except Exception as _e:
        _emit(
            Trace(
                definition=definition.name,
                solution=_solution_name,
                workload=_workload,
                evaluation=_make_eval(
                    EvaluationStatus.RUNTIME_ERROR,
                    _device,
                    None,
                    extra_msg=f"Timing failed: {_e}",
                ),
            )
        )
        continue

    # -- Thread injection defense check --
    if _reward_hack_check(
        _workload, check_thread_injection, _threads_before, threading.active_count()
    ):
        continue

    # -- Reference latency (for speedup factor) —always return-value style --
    # Inputs are cloned (not regenerated) since the reference cannot cheat.
    _ref_latency_ms = 0.0
    _ref_timing_failure = None
    if bench_config.benchmark_reference:
        _ref_timing = measure_reference_latency(
            ref_fn,
            _inputs,
            _device,
            warmup=bench_config.warmup_runs,
            rep=bench_config.iterations,
        )
        _ref_latency_ms = _ref_timing.latency_ms
        _ref_timing_failure = _ref_timing.failure

    _speedup = _ref_latency_ms / _sol_latency_ms if _sol_latency_ms > 0 else 0.0

    # -- Emit PASSED trace --
    _emit(
        Trace(
            definition=definition.name,
            solution=_solution_name,
            workload=_workload,
            evaluation=_make_eval(
                EvaluationStatus.PASSED,
                _device,
                None,
                correctness=_correctness,
                performance=Performance(
                    latency_ms=_sol_latency_ms,
                    reference_latency_ms=_ref_latency_ms,
                    speedup_factor=_speedup,
                ),
                extra_msg=_ref_timing_failure,
            ),
        )
    )
