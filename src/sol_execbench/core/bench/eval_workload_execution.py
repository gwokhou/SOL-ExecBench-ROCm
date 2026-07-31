# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Correctness and timing stages for one generated-driver workload."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import torch

from sol_execbench.core.bench.eval_correctness import (
    emit_reward_hack_if_detected,
    run_correctness_rounds,
)
from sol_execbench.core.bench.eval_timing import (
    SolutionTimingResult,
    measure_solution_latency,
    replay_marker_ranges,
)
from sol_execbench.core.bench.eval_trace_helpers import WorkloadTraceEmitter
from sol_execbench.core.bench.evaluation_requests import (
    WorkloadEvaluationRequest,
)
from sol_execbench.core.bench.reference_protocol import (
    ReferenceExecutionError,
    ReferenceFailureKind,
    ReferenceProtocolError,
    ReferenceTimingCase,
)
from sol_execbench.core.bench.reward_hack import (
    RewardHackError,
    ThreadInjectionMonitor,
    check_monkey_patch,
    check_thread_injection_from_monitor,
)
from sol_execbench.core.data.trace import (
    CacheClearEvidence,
    EvaluationStatus,
    Performance,
)
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.process.environment import (
    ENV_SOL_EXECBENCH_COUNTER_REPLAY,
    ENV_SOL_EXECBENCH_REPLAY_PASS_INDEX,
)


def evaluate_one_workload(
    request: WorkloadEvaluationRequest,
    emitter: WorkloadTraceEmitter,
    row_index: int,
    workload: Workload,
) -> None:
    """Run load, correctness, integrity, and timing stages for one workload."""
    _release_device_cache()
    resolved_axes = request.definition.get_resolved_axes_values(workload.axes)
    correctness_result = run_correctness_rounds(
        request=request,
        workload=workload,
        row_index=row_index,
        emitter=emitter,
    )
    if correctness_result.failed or emit_reward_hack_if_detected(
        emitter=emitter,
        workload=workload,
        check_fn=check_monkey_patch,
    ):
        return
    timing_case = _load_timing_case(request, emitter, workload, row_index)
    if timing_case is None:
        return
    _measure_and_emit(
        request,
        emitter,
        workload,
        resolved_axes,
        timing_case,
        correctness_result.correctness,
    )


def _load_timing_case(
    request: WorkloadEvaluationRequest,
    emitter: WorkloadTraceEmitter,
    workload: Workload,
    row_index: int,
) -> ReferenceTimingCase | None:
    try:
        return request.dependencies.reference_client.timing_case(
            workload_uuid=workload.uuid,
            row_index=row_index,
            round_index=9,
        )
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
    except ReferenceProtocolError as exc:
        emitter.emit_status(
            workload,
            EvaluationStatus.RUNTIME_ERROR,
            extra_msg=f"Trusted reference IPC failed: {exc}",
        )
    return None


def _measure_and_emit(
    request: WorkloadEvaluationRequest,
    emitter: WorkloadTraceEmitter,
    workload: Workload,
    resolved_axes: dict[str, int],
    timing_case: ReferenceTimingCase,
    correctness: Any,
) -> None:
    _release_device_cache()
    inputs = timing_case.inputs
    # Event guards catch every standard Python thread start during timing,
    # including workers shorter than the sampling interval. Concurrent count
    # sampling remains defense in depth for alternate entry points.
    thread_monitor = ThreadInjectionMonitor()
    with thread_monitor:
        try:
            solution_timing = measure_solution_latency(
                request=request,
                workload=workload,
                resolved_axes=resolved_axes,
                inputs=inputs,
                expected_outputs=timing_case.outputs,
            )
        except RewardHackError as exc:
            emitter.emit_status(
                workload,
                EvaluationStatus.REWARD_HACK,
                extra_msg=str(exc),
            )
            return
        except Exception as exc:  # noqa: BLE001 -- candidate timing boundary
            emitter.emit_status(
                workload,
                EvaluationStatus.RUNTIME_ERROR,
                extra_msg=f"Timing failed: {exc}",
            )
            return
    try:
        request.dependencies.check_integrity(
            request.dependencies.integrity_snapshot,
            request.dependencies.driver_globals,
        )
    except RewardHackError as exc:
        emitter.emit_status(
            workload,
            EvaluationStatus.REWARD_HACK,
            extra_msg=str(exc),
        )
        return
    if emit_reward_hack_if_detected(
        emitter=emitter,
        workload=workload,
        check_fn=check_thread_injection_from_monitor,
        args=(thread_monitor,),
    ):
        return
    _record_timing_evidence(
        request,
        workload,
        solution_timing,
        input_sha256=timing_case.input_sha256,
        access_patterns=[
            pattern.model_dump(mode="json")
            for pattern in timing_case.access_patterns
        ],
    )
    emitter.emit_status(
        workload,
        EvaluationStatus.PASSED,
        correctness=correctness,
        performance=_performance_evidence(
            request,
            timing_case,
            solution_timing,
        ),
        extra_msg=timing_case.timing_failure,
    )


def _record_timing_evidence(
    request: WorkloadEvaluationRequest,
    workload: Workload,
    timing: SolutionTimingResult,
    *,
    input_sha256: str,
    access_patterns: list[dict[str, object]],
) -> None:
    recorder = getattr(request.dependencies, "timing_recorder", None)
    if recorder is None:
        return
    if os.environ.get(ENV_SOL_EXECBENCH_COUNTER_REPLAY) == "1":
        recorder(_replay_record(workload, timing, input_sha256))
        return
    recorder(
        {
            "workload_uuid": workload.uuid,
            "input_sha256": input_sha256,
            "latency_ms": timing.latency_ms,
            "trial_samples_ms": [
                list(samples) for samples in timing.trial_samples_ms
            ],
            "warmup_runs": request.bench_config.warmup_runs,
            "timing_protocol": request.bench_config.timing_protocol,
            "access_patterns": access_patterns,
        },
    )


def _replay_record(
    workload: Workload,
    timing: SolutionTimingResult,
    input_sha256: str,
) -> dict[str, object]:
    cache = timing.cache_clear_policy
    cache_identity = stable_json_checksum(
        {
            "detected_l2_bytes": cache.detected_l2_bytes if cache else None,
            "clear_buffer_bytes": cache.clear_buffer_bytes if cache else None,
            "source": cache.source if cache else None,
            "fallback_reason": cache.fallback_reason if cache else None,
        },
    )
    return {
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "process_executable_sha256": sha256_file(
            Path("/proc/self/exe").resolve()
        ),
        "pass_index": int(
            os.environ.get(ENV_SOL_EXECBENCH_REPLAY_PASS_INDEX, "0")
        ),
        "workload_uuid": workload.uuid,
        "input_sha256": input_sha256,
        "cache_identity_sha256": cache_identity,
        "marker_ranges": replay_marker_ranges(workload.uuid),
    }


def _performance_evidence(
    request: WorkloadEvaluationRequest,
    timing_case: ReferenceTimingCase,
    solution_timing: SolutionTimingResult,
) -> Performance:
    cache_policy = solution_timing.cache_clear_policy
    cache_clear = (
        CacheClearEvidence(
            detected_l2_bytes=cache_policy.detected_l2_bytes,
            clear_buffer_bytes=cache_policy.clear_buffer_bytes,
            source=cache_policy.source,
            fallback_reason=cache_policy.fallback_reason,
        )
        if cache_policy is not None
        else None
    )
    return Performance(
        latency_ms=solution_timing.latency_ms,
        reference_latency_ms=timing_case.reference_latency_ms,
        speedup_factor=0.0,
        warmup_runs=request.bench_config.warmup_runs,
        timed_iterations=solution_timing.uniform_timed_iterations,
        timed_iterations_per_trial=list(
            solution_timing.timed_iterations_per_trial,
        ),
        trials=request.bench_config.trials,
        statistic="mean",
        timed_outputs_validated=True,
        cache_clear=cache_clear,
    )


def _release_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = ["evaluate_one_workload"]
