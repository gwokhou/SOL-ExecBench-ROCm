# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Replace RDNA4 fallback timing sidecars with profiler-backed rocprofv3 timing."""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

from sol_execbench.core import BenchmarkConfig, Definition, Solution, Workload
from sol_execbench.core.bench.rocm_profiler import (
    ROCPROFV3_EXECUTABLE,
    ProfilerRunner,
    Rocprofv3CollectionRequest,
    collect_rocprofv3_timing,
)
from sol_execbench.core.bench.timing_policy import (
    TimingActivityDomain,
    TimingBackend,
    TimingPolicy,
    TimingSourceType,
)
from sol_execbench.core.dataset.profiler_timing_coverage import (
    ProfilerTimingCoverageReport,
    ProfilerTimingProblemCoverage,
    build_profiler_timing_coverage_report,
)
from sol_execbench.core.dataset.inventory import build_dataset_inventory
from sol_execbench.core.dataset.readiness import classify_rocm_readiness
from sol_execbench.core.dataset.solutions import build_reference_solution
from sol_execbench.core.data.json_utils import load_json_dict
from sol_execbench.core.platform.runtime import (
    resolve_rocm_tool,
    resolve_rocm_tool_command,
)
from sol_execbench.core.process import run_in_process_group
from sol_execbench.driver import ProblemPackager
from sol_execbench.core.bench.pid_lock import (
    acquire_pid_lock,
    read_pid_lock_contention_marker,
)
from sol_execbench.core.bench.timing_isolation import (
    clear_gpu_cache_between_subprocesses,
    collect_timing_environment_snapshot,
    detect_concurrent_gpu_processes,
    validate_gpu_device_isolation,
    verify_clock_state_with_warning,
)
from sol_execbench.core.text_utils import subprocess_text as _subprocess_text

logger = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT = Path("data/SOL-ExecBench/benchmark")
DEFAULT_SOURCE_TIMING_DIR = Path("out/rdna4-timing-evidence/timing")
DEFAULT_OUTPUT_DIR = Path("out/rdna4-profiler-backed-timing")
DEFAULT_TEMP_ROOT = Path("tmp")
DEFAULT_EXPECTED_PROBLEM_DENOMINATOR = 235
DEFAULT_TOOL_VERSION = "rocprofv3"
DEFAULT_GPU_ARCHITECTURE = "gfx1200"
WORKLOAD_MANIFEST_SCHEMA_VERSION = "sol_execbench.rdna4_workload_profiler_manifest.v1"
WORKLOAD_AGGREGATE_SCHEMA_VERSION = "sol_execbench.rdna4_workload_profiler_aggregate.v1"
CLAIM_BOUNDARY = (
    "Profiler-backed RDNA4 timing replacement evidence only; not score "
    "authority, paper parity, leaderboard result, or broader hardware "
    "validation."
)
AUTO_TIMING_INPUT_CAP_FRACTION = 0.70
RDNA4_REFERENCE_OVERRIDES: dict[str, str] = {
    "L2/035_convnextv2_block_with_grn": """import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    x: torch.Tensor,
    dwconv_weight: torch.Tensor,
    dwconv_bias: torch.Tensor,
    layernorm_weight: torch.Tensor,
    layernorm_bias: torch.Tensor,
    pwconv1_weight: torch.Tensor,
    pwconv1_bias: torch.Tensor,
    grn_weight: torch.Tensor,
    grn_bias: torch.Tensor,
    pwconv2_weight: torch.Tensor,
    pwconv2_bias: torch.Tensor,
    eps: float,
    layer_norm_eps: float,
):
    residual = x
    _, channels, _, _ = x.shape

    out = F.conv2d(x, dwconv_weight, dwconv_bias, padding=3, groups=channels)
    out = out.permute(0, 2, 3, 1)
    out = F.layer_norm(
        out,
        (channels,),
        layernorm_weight,
        layernorm_bias,
        eps=layer_norm_eps,
    )
    out = out.permute(0, 3, 1, 2).contiguous()

    out = F.conv2d(out, pwconv1_weight[:, :, None, None], pwconv1_bias)
    out = F.gelu(out)

    global_features = torch.linalg.vector_norm(
        out,
        ord=2,
        dim=(2, 3),
        keepdim=True,
    )
    norm_features = global_features / (
        global_features.mean(dim=1, keepdim=True) + eps
    )
    grn_weight_nchw = grn_weight.permute(0, 3, 1, 2).contiguous()
    grn_bias_nchw = grn_bias.permute(0, 3, 1, 2).contiguous()
    out = grn_weight_nchw * (out * norm_features) + grn_bias_nchw + out

    out = F.conv2d(out, pwconv2_weight[:, :, None, None], pwconv2_bias)
    return residual + out
""",
}
RDNA4_REFERENCE_OVERRIDE_METADATA: dict[str, dict[str, Any]] = {
    "L2/035_convnextv2_block_with_grn": {
        "override_type": "equivalent_reference_implementation",
        "reason": (
            "Use NCHW 1x1 conv pointwise projections for RDNA4 profiler timing; "
            "semantic output is equivalent to the original NHWC matmul reference, "
            "but backend dispatch is not original-reference dispatch timing."
        ),
        "claim_boundary": (
            "Profiler timing collected with this override must be reported as "
            "reference-implementation override evidence, not unmodified benchmark "
            "reference dispatch timing."
        ),
    }
}


def _is_gfx1200_patched_profiler(executable: str) -> bool:
    """Return whether *executable* is the explicit patched clock owner.

    This is deliberately name-based rather than a PATH lookup: clock ownership
    is an operator-selected batch contract, and implicit same-name shadows are
    not accepted as evidence of that contract.
    """

    return Path(executable).name == "rocprofv3-gfx1200-patched"


def _partition_targets_by_index(
    targets: list[ProfilerTimingProblemCoverage],
    max_workers: int,
) -> list[list[ProfilerTimingProblemCoverage]]:
    """Pre-partition targets by index for exclusive worker ownership (PRFL-03)."""
    if not targets:
        return []
    chunk_size = (len(targets) + max_workers - 1) // max_workers
    chunks = []
    for i in range(0, len(targets), chunk_size):
        chunks.append(targets[i : i + chunk_size])
    return chunks


def _process_target_chunk(
    chunk: list[ProfilerTimingProblemCoverage],
    *,
    dataset_root: Path,
    replacement_timing_dir: Path,
    output_dir: Path,
    workload_limit: int | None = None,
    workload_offset: int = 0,
    workload_sharded: bool = False,
    workload_slice_timing_dirs: Sequence[Path] = (),
    workload_sharded_import_only: bool = False,
    timeout: int = 900,
    subprocess_memory_limit_gib: float | None = None,
    max_estimated_timing_input_gib: float | None = None,
    estimated_timing_input_cap_source: str | None = None,
    temp_dir: Path | None = None,
    tool_version: str = DEFAULT_TOOL_VERSION,
    gpu_architecture: str = DEFAULT_GPU_ARCHITECTURE,
    clock_locked: bool = True,
    include_hip_runtime_trace: bool = False,
    rocprofv3_available: bool,
    runner: ProfilerRunner | None,
    marked_blocked_ids: set[str],
    global_start_index: int = 0,
    concurrent_gpu_processes: list[dict[str, Any]] | None = None,
    calibration_path: Path | None = None,
    pid_lock_contention: bool = False,
    compact_workload_slices: bool = True,
    keep_staging: bool = False,
    keep_rocprofv3_csv: bool = False,
    profiler_executable: str = ROCPROFV3_EXECUTABLE,
) -> list[dict[str, Any]]:
    """Process a chunk of targets with CPU-parallel staging + serial GPU profiling (PRFL-01, PRFL-02)."""
    chunk_results = []
    contention_detected = pid_lock_contention
    for local_idx, target in enumerate(chunk, 1):
        global_target_index = global_start_index + local_idx
        if target.problem_id in marked_blocked_ids:
            continue

        # Re-verify clock state periodically (every 10 problems globally)
        if global_target_index % 10 == 0:
            verify_clock_state_with_warning(context=f"problem_{global_target_index}")
            # Also check for PID lock contention marker from rejected instances
            if not contention_detected:
                contention_detected = read_pid_lock_contention_marker(output_dir)

        # Process target (CPU staging + GPU profiling)
        if workload_sharded:
            result = _profile_target_workload_sharded(
                target,
                dataset_root=dataset_root,
                replacement_timing_dir=replacement_timing_dir,
                output_dir=output_dir,
                workload_limit=workload_limit,
                workload_offset=workload_offset,
                workload_slice_timing_dirs=workload_slice_timing_dirs,
                workload_sharded_import_only=workload_sharded_import_only,
                timeout=timeout,
                subprocess_memory_limit_gib=subprocess_memory_limit_gib,
                max_estimated_timing_input_gib=max_estimated_timing_input_gib,
                estimated_timing_input_cap_source=estimated_timing_input_cap_source,
                temp_dir=temp_dir,
                tool_version=tool_version,
                gpu_architecture=gpu_architecture,
                clock_locked=clock_locked,
                include_hip_runtime_trace=include_hip_runtime_trace,
                rocprofv3_available=rocprofv3_available,
                runner=runner,
                concurrent_gpu_processes=concurrent_gpu_processes,
                calibration_path=calibration_path,
                pid_lock_contention=contention_detected,
                compact_workload_slices=compact_workload_slices,
                keep_staging=keep_staging,
                keep_rocprofv3_csv=keep_rocprofv3_csv,
                profiler_executable=profiler_executable,
            )
        else:
            result = _profile_target(
                target,
                dataset_root=dataset_root,
                replacement_timing_dir=replacement_timing_dir,
                output_dir=output_dir,
                workload_limit=workload_limit,
                workload_offset=workload_offset,
                timeout=timeout,
                subprocess_memory_limit_gib=subprocess_memory_limit_gib,
                max_estimated_timing_input_gib=max_estimated_timing_input_gib,
                estimated_timing_input_cap_source=estimated_timing_input_cap_source,
                temp_dir=temp_dir,
                tool_version=tool_version,
                gpu_architecture=gpu_architecture,
                clock_locked=clock_locked,
                include_hip_runtime_trace=include_hip_runtime_trace,
                rocprofv3_available=rocprofv3_available,
                runner=runner,
                concurrent_gpu_processes=concurrent_gpu_processes,
                calibration_path=calibration_path,
                pid_lock_contention=contention_detected,
                keep_staging=keep_staging,
                keep_rocprofv3_csv=keep_rocprofv3_csv,
                profiler_executable=profiler_executable,
            )

        chunk_results.append(result)

        # Clear GPU cache at subprocess boundary
        clear_gpu_cache_between_subprocesses()

    return chunk_results


def run_batch(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    source_timing_dirs: Sequence[Path] = (DEFAULT_SOURCE_TIMING_DIR,),
    replacement_timing_dir: Path | None = None,
    limit: int | None = None,
    only_problem: Sequence[str] = (),
    skip_problem: Sequence[str] = (),
    mark_blocked_problem: Sequence[str] = (),
    mark_blocked_only: bool = False,
    workload_limit: int | None = None,
    workload_offset: int = 0,
    workload_sharded: bool = False,
    workload_slice_timing_dirs: Sequence[Path] = (),
    workload_sharded_import_only: bool = False,
    timeout: int = 900,
    subprocess_memory_limit_gib: float | None = None,
    max_estimated_timing_input_gib: float | None = None,
    auto_estimated_timing_input_cap: bool = True,
    temp_dir: Path | None = None,
    resume: bool = True,
    tool_version: str = DEFAULT_TOOL_VERSION,
    gpu_architecture: str = DEFAULT_GPU_ARCHITECTURE,
    clock_locked: bool = True,
    include_hip_runtime_trace: bool = False,
    rocprofv3_available: bool | None = None,
    runner: ProfilerRunner | None = None,
    max_workers: int = 4,
    strict_isolation: bool = False,
    gpu_device: int | None = None,
    calibration_path: Path | None = None,
    compact_workload_slices: bool = True,
    keep_staging: bool = False,
    keep_rocprofv3_csv: bool = False,
    profiler_executable: str | None = None,
) -> int:
    """Run fallback replacement batch and return a process-style status code."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    if workload_limit is not None and workload_limit <= 0:
        raise ValueError("workload_limit must be positive when provided")
    if workload_offset < 0:
        raise ValueError("workload_offset must be non-negative")
    if workload_sharded and workload_limit is not None:
        raise ValueError("workload_sharded mode controls workload_limit internally")
    if workload_sharded and workload_offset:
        raise ValueError("workload_sharded mode profiles all manifest-missing offsets")
    if strict_isolation and max_workers != 1:
        logger.info(
            "STRICT ISOLATION: reducing profiler batch workers from %d to 1 "
            "to avoid concurrent GPU profiling",
            max_workers,
        )
        max_workers = 1
    (
        resolved_estimated_timing_input_gib,
        estimated_timing_input_cap_source,
    ) = _resolve_estimated_timing_input_cap(
        max_estimated_timing_input_gib=max_estimated_timing_input_gib,
        auto_estimated_timing_input_cap=auto_estimated_timing_input_cap,
        subprocess_memory_limit_gib=subprocess_memory_limit_gib,
    )
    if resolved_estimated_timing_input_gib is not None:
        logger.info(
            "Using estimated timing input cap %.2f GiB (%s)",
            resolved_estimated_timing_input_gib,
            estimated_timing_input_cap_source,
        )

    replacement_root = replacement_timing_dir or output_dir / "timing"
    coverage = _build_coverage(
        dataset_root=dataset_root,
        timing_dirs=(replacement_root, *tuple(source_timing_dirs)),
    )
    targets = select_fallback_targets(
        coverage,
        replacement_timing_dir=replacement_root,
        limit=limit,
        only_problem=only_problem,
        skip_problem=skip_problem,
        resume=resume,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_profiler = (
        str(Path(profiler_executable).resolve())
        if profiler_executable and Path(profiler_executable).is_file()
        else profiler_executable or resolve_rocm_tool_command(ROCPROFV3_EXECUTABLE)
    )
    available = (
        (
            Path(selected_profiler).is_file()
            if profiler_executable
            else resolve_rocm_tool(ROCPROFV3_EXECUTABLE) is not None
        )
        if rocprofv3_available is None
        else rocprofv3_available
    )
    logger.info("Using rocprofv3 executable: %s", selected_profiler)

    # Pre-flight timing isolation audit
    logger.info("Running timing isolation pre-flight audit...")
    concurrent_processes = detect_concurrent_gpu_processes()
    if concurrent_processes:
        if strict_isolation:
            logger.error(
                "STRICT ISOLATION: Detected %d concurrent GPU process(es), aborting: %s",
                len(concurrent_processes),
                concurrent_processes,
            )
            return 1
        logger.warning(
            "Detected %d concurrent GPU process(es). This may introduce timing variability: %s",
            len(concurrent_processes),
            concurrent_processes,
        )
    # A requested patched wrapper owns the AUTO -> STABLE_PEAK transition.
    # Strict isolation must not reject AUTO before that owner has a chance to
    # acquire its lease.  Normal profiler runs still require a pre-locked GPU.
    profiler_manages_clocks = _is_gfx1200_patched_profiler(selected_profiler)
    clock_state_verified = verify_clock_state_with_warning(context="batch_start")
    if not clock_state_verified:
        if strict_isolation and not profiler_manages_clocks:
            logger.error(
                "STRICT ISOLATION: Clock state verification failed at batch start, aborting"
            )
            return 1
        logger.warning(
            "Clock state verification failed at batch start%s",
            "; requested patched profiler will acquire STABLE_PEAK"
            if profiler_manages_clocks
            else "",
        )

    # GPU device isolation check (ENV-01)
    gpu_isolation = validate_gpu_device_isolation(gpu_device=gpu_device)
    if not gpu_isolation["isolated"]:
        if strict_isolation:
            logger.error(
                "STRICT ISOLATION: GPU device isolation check failed, aborting: %s",
                gpu_isolation["warnings"],
            )
            return 1
        for warn in gpu_isolation["warnings"]:
            logger.warning("GPU device isolation: %s", warn)

    if runner is None and not _torch_rocm_gpu_available():
        logger.error(
            "PyTorch ROCm cannot see a GPU in this process. "
            "Run this profiler batch outside sandbox/device isolation; "
            "otherwise eval_driver falls back to CPU and produces invalid "
            "RUNTIME_ERROR timing sidecars."
        )
        return 1

    results: list[dict[str, Any]] = []
    marked_blocked = _mark_blocked_targets(
        coverage,
        replacement_timing_dir=replacement_root,
        problem_ids=mark_blocked_problem,
        reason="manual profiler block classification",
    )
    results.extend(marked_blocked)
    marked_blocked_ids = {result["problem_id"] for result in marked_blocked}
    if not mark_blocked_only:
        # Pre-partition targets by index for CPU-parallel staging (PRFL-03)
        target_chunks = _partition_targets_by_index(targets, max_workers)

        # CPU-parallel staging + GPU-serial profiling (PRFL-01, PRFL-02)
        all_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for chunk_idx, chunk in enumerate(target_chunks):
                global_start_idx = sum(len(c) for c in target_chunks[:chunk_idx])
                future = executor.submit(
                    _process_target_chunk,
                    chunk=chunk,
                    dataset_root=dataset_root,
                    replacement_timing_dir=replacement_root,
                    output_dir=output_dir,
                    workload_limit=workload_limit,
                    workload_offset=workload_offset,
                    workload_sharded=workload_sharded,
                    workload_slice_timing_dirs=workload_slice_timing_dirs,
                    workload_sharded_import_only=workload_sharded_import_only,
                    timeout=timeout,
                    subprocess_memory_limit_gib=subprocess_memory_limit_gib,
                    max_estimated_timing_input_gib=resolved_estimated_timing_input_gib,
                    estimated_timing_input_cap_source=estimated_timing_input_cap_source,
                    temp_dir=temp_dir,
                    tool_version=tool_version,
                    gpu_architecture=gpu_architecture,
                    clock_locked=clock_locked,
                    include_hip_runtime_trace=include_hip_runtime_trace,
                    rocprofv3_available=available,
                    runner=runner,
                    marked_blocked_ids=marked_blocked_ids,
                    global_start_index=global_start_idx,
                    concurrent_gpu_processes=concurrent_processes or None,
                    calibration_path=calibration_path,
                    compact_workload_slices=compact_workload_slices,
                    keep_staging=keep_staging,
                    keep_rocprofv3_csv=keep_rocprofv3_csv,
                    profiler_executable=selected_profiler,
                )
                futures[future] = chunk_idx

            try:
                # Wait for all futures to complete
                for future in as_completed(futures.keys()):
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
            except KeyboardInterrupt:
                # Structured interrupt handling (PRFL-05)
                logger.info(
                    "Keyboard interrupt received, cancelling pending workers..."
                )
                for future in futures:
                    future.cancel()

                # Collect completed results
                partial_results = []
                for future in futures:
                    if future.done():
                        try:
                            chunk_results = future.result()
                            partial_results.extend(chunk_results)
                        except Exception as exc:
                            logger.error(f"Worker failed: {exc}")

                # Include marked_blocked in partial results
                partial_results.extend(marked_blocked)
                all_results = partial_results

                # Build partial-completion summary
                summary = _build_summary(
                    coverage=coverage,
                    selected_targets=targets,
                    results=all_results,
                    output_dir=output_dir,
                    replacement_timing_dir=replacement_root,
                    source_timing_dirs=source_timing_dirs,
                    rocprofv3_available=available,
                    interrupted=True,  # NEW: flag for partial completion
                )
                (output_dir / "batch-summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                (output_dir / "batch-summary.md").write_text(
                    _render_summary_markdown(summary),
                    encoding="utf-8",
                )
                return 130  # Standard exit code for interrupted batch

        # Include marked_blocked in final results
        all_results.extend(marked_blocked)

        # Sort results for deterministic output order (PRFL-06)
        all_results.sort(key=lambda r: r["problem_id"])
        results = all_results

    summary = _build_summary(
        coverage=coverage,
        selected_targets=targets,
        results=results,
        output_dir=output_dir,
        replacement_timing_dir=replacement_root,
        source_timing_dirs=source_timing_dirs,
        rocprofv3_available=available,
    )
    (output_dir / "batch-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "batch-summary.md").write_text(
        _render_summary_markdown(summary),
        encoding="utf-8",
    )
    failures = summary["failed"] + summary["fallback_or_missing"]
    return 0 if failures == 0 else 1


def select_fallback_targets(
    coverage: ProfilerTimingCoverageReport,
    *,
    replacement_timing_dir: Path,
    limit: int | None = None,
    only_problem: Sequence[str] = (),
    skip_problem: Sequence[str] = (),
    resume: bool = True,
) -> list[ProfilerTimingProblemCoverage]:
    """Select coverage rows whose fallback timing should be replaced."""
    only = set(only_problem)
    skip = set(skip_problem)
    targets: list[ProfilerTimingProblemCoverage] = []
    for problem in coverage.problems:
        if problem.status not in {
            "timing_fallback",
            "ready_missing_profiler_timing",
            "partial_profiler_backed",
            "profiler_blocked",
        }:
            continue
        if only and problem.problem_id not in only:
            continue
        if problem.problem_id in skip:
            continue
        replacement_path = _replacement_timing_path(
            replacement_timing_dir,
            problem.category,
            problem.problem_path,
        )
        if resume and _is_classified_replacement_sidecar(replacement_path):
            continue
        targets.append(problem)
        if limit is not None and len(targets) >= limit:
            break
    return targets


def forced_pytorch_kernel_activity_policy() -> TimingPolicy:
    """Return the explicit Phase 149 replacement timing policy."""
    return TimingPolicy(
        source_type=TimingSourceType.PYTORCH,
        backend=TimingBackend.ROCPROFV3,
        activity_domain=TimingActivityDomain.KERNEL_ACTIVITY,
        aggregation_rule=(
            "aggregate ROCm kernel activity rows emitted while the staged "
            "PyTorch reference eval_driver process executes correctness and "
            "timing workloads"
        ),
        interpretation=(
            "forced profiler-backed kernel activity timing for RDNA4 fallback "
            "replacement; this is not PyTorch operator attribution"
        ),
        fallback_applied=False,
        reason="Phase 149 explicitly replaces fallback timing with rocprofv3 kernel activity",
    )


def _build_coverage(
    *,
    dataset_root: Path,
    timing_dirs: Sequence[Path],
) -> ProfilerTimingCoverageReport:
    inventory = build_dataset_inventory(dataset_root)
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)
    return build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=tuple(timing_dirs),
        expected_problem_denominator=DEFAULT_EXPECTED_PROBLEM_DENOMINATOR,
    )


def _profile_target(
    target: ProfilerTimingProblemCoverage,
    *,
    dataset_root: Path,
    replacement_timing_dir: Path,
    output_dir: Path,
    workload_limit: int | None,
    workload_offset: int,
    timeout: int,
    subprocess_memory_limit_gib: float | None,
    max_estimated_timing_input_gib: float | None,
    estimated_timing_input_cap_source: str | None,
    temp_dir: Path | None,
    tool_version: str,
    gpu_architecture: str,
    clock_locked: bool,
    include_hip_runtime_trace: bool,
    rocprofv3_available: bool,
    runner: ProfilerRunner | None,
    concurrent_gpu_processes: list[dict[str, Any]] | None = None,
    calibration_path: Path | None = None,
    pid_lock_contention: bool = False,
    compact_workload_slices: bool = True,
    keep_staging: bool = False,
    keep_rocprofv3_csv: bool = False,
    profiler_executable: str = ROCPROFV3_EXECUTABLE,
) -> dict[str, Any]:
    problem_dir = dataset_root / target.problem_path
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix="sol_execbench_rdna4_prof_batch_",
            dir=temp_dir,
        )
    )
    problem_name = Path(target.problem_path).name
    workload_slice_requested = workload_offset != 0 or workload_limit is not None
    profile_dir = output_dir / "rocprofv3" / target.category / problem_name
    output_file = problem_name
    if workload_slice_requested:
        profile_dir = profile_dir / f"workload-{workload_offset:04d}"
        output_file = f"{problem_name}_workload_{workload_offset:04d}"
    replacement_path = _replacement_timing_path(
        replacement_timing_dir,
        target.category,
        target.problem_path,
    )
    reference_override: dict[str, Any] | None = None
    try:
        definition_payload = _load_json(problem_dir / "definition.json")
        reference_override = _apply_rdna4_reference_override(
            definition_payload,
            target.problem_id,
        )
        workloads = _load_workloads(
            problem_dir / "workload.jsonl",
            limit=workload_limit,
            offset=workload_offset,
        )
        preflight_reason = _estimated_timing_input_block_reason(
            definition_payload,
            workloads,
            max_estimated_timing_input_gib=max_estimated_timing_input_gib,
            cap_source=estimated_timing_input_cap_source,
        )
        if preflight_reason is not None:
            return _result_with_optional_staging_cleanup(
                _write_blocked_sidecar(
                    target,
                    replacement_path=replacement_path,
                    staging_dir=staging_dir,
                    reason=preflight_reason,
                    workload_offset=workload_offset,
                    workload_limit=workload_limit,
                    concurrent_gpu_processes=concurrent_gpu_processes,
                    pid_lock_contention=pid_lock_contention,
                    reference_override=reference_override,
                ),
                staging_dir=staging_dir,
                keep_staging=keep_staging,
            )
        solution = build_reference_solution(definition_payload)
        packager = ProblemPackager(
            definition=Definition(**definition_payload),
            workloads=workloads,
            solution=Solution(**solution),
            config=BenchmarkConfig(),
            output_dir=staging_dir,
            keep_output_dir=True,
        )
        command_parts = [str(part) for part in packager.execute()]
        if command_parts and command_parts[0] == "python":
            command_parts[0] = sys.executable
        command = tuple(command_parts)
        batch_runner = runner or _staging_runner(
            staging_dir,
            timeout=timeout,
            memory_limit_gib=subprocess_memory_limit_gib,
        )
        workload_slice_requested = workload_offset != 0 or workload_limit is not None
        workload_slice_applied = workload_offset != 0 or (
            workload_limit is not None and len(workloads) < target.workload_count
        )
        result = collect_rocprofv3_timing(
            Rocprofv3CollectionRequest(
                application_command=command,
                output_directory=profile_dir.resolve(),
                output_file=output_file,
                policy=forced_pytorch_kernel_activity_policy(),
                tool_version=tool_version,
                gpu_architecture=gpu_architecture,
                warmup_runs=BenchmarkConfig().warmup_runs,
                iterations=BenchmarkConfig().iterations,
                min_measurement_time_seconds=(
                    BenchmarkConfig().min_measurement_time_seconds
                ),
                trial_count=1,
                clock_locked=clock_locked,
                include_hip_runtime=include_hip_runtime_trace,
                compact_rows=True,
                executable=profiler_executable,
            ),
            rocprofv3_available=rocprofv3_available,
            runner=batch_runner,
            calibration_path=calibration_path,
        )
    except subprocess.TimeoutExpired as exc:
        return _result_with_optional_staging_cleanup(
            _write_blocked_sidecar(
                target,
                replacement_path=replacement_path,
                staging_dir=staging_dir,
                reason=f"rocprofv3 command timed out after {timeout} seconds",
                workload_offset=workload_offset,
                workload_limit=workload_limit,
                stdout=_subprocess_text(exc.stdout),
                stderr=_subprocess_text(exc.stderr),
                concurrent_gpu_processes=concurrent_gpu_processes,
                pid_lock_contention=pid_lock_contention,
                reference_override=reference_override,
            ),
            staging_dir=staging_dir,
            keep_staging=keep_staging,
        )
    except (OSError, ValueError) as exc:
        return _result_with_optional_staging_cleanup(
            _write_blocked_sidecar(
                target,
                replacement_path=replacement_path,
                staging_dir=staging_dir,
                reason=str(exc),
                workload_offset=workload_offset,
                workload_limit=workload_limit,
                concurrent_gpu_processes=concurrent_gpu_processes,
                pid_lock_contention=pid_lock_contention,
                reference_override=reference_override,
            ),
            staging_dir=staging_dir,
            keep_staging=keep_staging,
        )

    trace_status_counts = _trace_status_counts(result.stdout)
    all_workloads_passed = (
        not workload_slice_applied
        and len(workloads) >= target.workload_count
        and trace_status_counts.get("PASSED", 0) >= target.workload_count
        and sum(trace_status_counts.values()) >= target.workload_count
    )
    payload: dict[str, Any] = dict(result.to_dict())
    if not isinstance(payload.get("evidence"), dict):
        payload["evidence"] = {"backend": "rocprofv3", "parsed_rows": []}
    evidence: dict[str, Any] = cast(
        dict[str, Any],
        payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {},
    )
    kernel_activity_rows = _kernel_activity_rows(evidence)
    kernel_duration_ms = _float_value(evidence.get("kernel_duration_ms"))
    replacement_failure_reason = _replacement_failure_reason(
        profiler_collected=result.profiler_collected,
        all_workloads_passed=all_workloads_passed,
        selection_reason=result.selection.reason,
    )
    status = _replacement_status(
        profiler_collected=result.profiler_collected,
        full_workload_coverage=all_workloads_passed,
        kernel_activity_rows=kernel_activity_rows,
    )
    _compact_payload_parsed_rows(
        payload,
        reason=(
            "RDNA4 profiler batch retains kernel row and duration summaries; "
            "raw rocprofv3 CSV is retained only when --keep-rocprofv3-csv is set"
        ),
    )
    payload["replacement_metadata"] = {
        "schema_version": "sol_execbench.rdna4_profiler_timing_replacement.v1",
        "problem_id": target.problem_id,
        "replacement_status": status,
        "profiled_workload_count": len(workloads),
        "expected_workload_count": target.workload_count,
        "trace_status_counts": trace_status_counts,
        "all_workloads_passed": all_workloads_passed,
        "workload_limit_applied": workload_limit if workload_slice_applied else None,
        "workload_offset": workload_offset,
        "workload_slice_applied": workload_slice_applied,
        "workload_slice": {
            "offset": workload_offset,
            "limit": workload_limit,
            "selected_workload_count": len(workloads),
        },
        "full_workload_coverage": all_workloads_passed,
        "failure_reason": replacement_failure_reason,
        "reference_override": reference_override,
    }
    if result.profiler_collected or status == "profiler_blocked":
        if concurrent_gpu_processes:
            payload["concurrent_gpu_processes"] = concurrent_gpu_processes
        if pid_lock_contention:
            payload["pid_lock_contention"] = True
        replacement_path.parent.mkdir(parents=True, exist_ok=True)
        replacement_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not keep_rocprofv3_csv and result.csv_path is not None:
        _remove_rocprofv3_run_dir(result.csv_path)
    return _result_with_optional_staging_cleanup(
        {
            "problem_id": target.problem_id,
            "status": status,
            "replacement_path": str(replacement_path),
            "staging_dir": str(staging_dir),
            "profiler_collected": result.profiler_collected,
            "full_workload_coverage": all_workloads_passed,
            "workload_offset": workload_offset,
            "workload_slice_applied": workload_slice_applied,
            "trace_status_counts": trace_status_counts,
            "csv_path": str(result.csv_path) if result.csv_path is not None else None,
            "kernel_activity_rows": kernel_activity_rows,
            "kernel_duration_ms": kernel_duration_ms,
            "fallback_reason": replacement_failure_reason,
            "returncode": result.returncode,
        },
        staging_dir=staging_dir,
        keep_staging=keep_staging,
    )


def _profile_target_workload_sharded(
    target: ProfilerTimingProblemCoverage,
    *,
    dataset_root: Path,
    replacement_timing_dir: Path,
    output_dir: Path,
    workload_limit: int | None,
    workload_offset: int,
    workload_slice_timing_dirs: Sequence[Path],
    workload_sharded_import_only: bool,
    timeout: int,
    subprocess_memory_limit_gib: float | None,
    max_estimated_timing_input_gib: float | None,
    estimated_timing_input_cap_source: str | None,
    temp_dir: Path | None,
    tool_version: str,
    gpu_architecture: str,
    clock_locked: bool,
    include_hip_runtime_trace: bool,
    rocprofv3_available: bool,
    runner: ProfilerRunner | None,
    concurrent_gpu_processes: list[dict[str, Any]] | None = None,
    calibration_path: Path | None = None,
    pid_lock_contention: bool = False,
    compact_workload_slices: bool = True,
    keep_staging: bool = False,
    keep_rocprofv3_csv: bool = False,
    profiler_executable: str = ROCPROFV3_EXECUTABLE,
) -> dict[str, Any]:
    if workload_limit is not None or workload_offset:
        raise ValueError("workload-sharded mode owns workload slicing")
    problem_dir = dataset_root / target.problem_path
    workload_refs = _load_workload_refs(problem_dir / "workload.jsonl")
    manifest_path = _workload_manifest_path(output_dir, target)
    manifest = _load_or_create_workload_manifest(
        target=target,
        dataset_root=dataset_root,
        workload_refs=workload_refs,
        manifest_path=manifest_path,
        tool_version=tool_version,
        gpu_architecture=gpu_architecture,
        clock_locked=clock_locked,
    )
    _write_workload_manifest(manifest_path, manifest)
    entries_by_index = {
        entry.get("workload_index"): entry
        for entry in manifest.get("workloads", [])
        if isinstance(entry, dict)
    }
    imported_sidecars = _imported_workload_slice_sidecars(
        target,
        workload_slice_timing_dirs,
    )
    results: list[dict[str, Any]] = []
    for ref in workload_refs:
        index = ref["workload_index"]
        existing = entries_by_index.get(index)
        if _workload_manifest_entry_complete(existing):
            continue
        imported = imported_sidecars.get(index)
        if imported is not None:
            entry = _workload_manifest_entry(
                ref,
                result={
                    "status": "partial_profiler_backed",
                    "replacement_path": imported.as_posix(),
                    "csv_path": _csv_path_from_sidecar(imported),
                    "fallback_reason": None,
                },
                sidecar=_load_optional_json(imported),
            )
            _upsert_workload_manifest_entry(manifest, entry)
            _write_workload_manifest(manifest_path, manifest)
            if _workload_manifest_entry_complete(entry):
                continue
        if workload_sharded_import_only:
            _upsert_workload_manifest_entry(
                manifest,
                {
                    **ref,
                    "status": "missing_imported_profiler_slice",
                    "retryable": True,
                    "profiler_collected": False,
                    "backend": "rocprofv3",
                    "trace_status_counts": {},
                    "kernel_activity_rows": 0,
                    "kernel_duration_ms": 0.0,
                    "replacement_path": None,
                    "csv_path": None,
                    "failure_reason": (
                        "no complete imported workload slice sidecar was found"
                    ),
                },
            )
            _write_workload_manifest(manifest_path, manifest)
            continue
        slice_timing_root = (
            output_dir / "workload-slices" / f"workload-{index:04d}" / "timing"
        )
        result = _profile_target(
            target,
            dataset_root=dataset_root,
            replacement_timing_dir=slice_timing_root,
            output_dir=output_dir,
            workload_limit=1,
            workload_offset=index,
            timeout=timeout,
            subprocess_memory_limit_gib=subprocess_memory_limit_gib,
            max_estimated_timing_input_gib=max_estimated_timing_input_gib,
            estimated_timing_input_cap_source=estimated_timing_input_cap_source,
            temp_dir=temp_dir,
            tool_version=tool_version,
            gpu_architecture=gpu_architecture,
            clock_locked=clock_locked,
            include_hip_runtime_trace=include_hip_runtime_trace,
            rocprofv3_available=rocprofv3_available,
            runner=runner,
            concurrent_gpu_processes=concurrent_gpu_processes,
            calibration_path=calibration_path,
            keep_staging=keep_staging,
            keep_rocprofv3_csv=keep_rocprofv3_csv,
            profiler_executable=profiler_executable,
        )
        results.append(result)
        entry = _workload_manifest_entry(ref, result=result, sidecar=None)
        _upsert_workload_manifest_entry(manifest, entry)
        _write_workload_manifest(manifest_path, manifest)
        if compact_workload_slices and _workload_manifest_entry_complete(entry):
            _compact_workload_slice_artifacts(
                sidecar_path=Path(result["replacement_path"]),
                csv_path=result.get("csv_path"),
            )

    aggregate = _write_workload_aggregate_sidecar(
        target=target,
        manifest=manifest,
        replacement_path=_replacement_timing_path(
            replacement_timing_dir,
            target.category,
            target.problem_path,
        ),
        tool_version=tool_version,
        gpu_architecture=gpu_architecture,
        clock_locked=clock_locked,
    )
    return {
        "problem_id": target.problem_id,
        "status": aggregate["status"],
        "replacement_path": aggregate["replacement_path"],
        "manifest_path": str(manifest_path),
        "profiler_collected": aggregate["profiler_collected"],
        "full_workload_coverage": aggregate["full_workload_coverage"],
        "profiled_workload_count": aggregate["profiled_workload_count"],
        "expected_workload_count": aggregate["expected_workload_count"],
        "workload_sharded": True,
        "workload_results": results,
        "fallback_reason": aggregate["failure_reason"],
        "returncode": None,
    }


def _target_failure(
    target: ProfilerTimingProblemCoverage,
    *,
    replacement_path: Path,
    staging_dir: Path | None,
    reason: str,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    return {
        "problem_id": target.problem_id,
        "status": "profiler_blocked",
        "replacement_path": str(replacement_path),
        "staging_dir": str(staging_dir) if staging_dir is not None else None,
        "profiler_collected": False,
        "csv_path": None,
        "fallback_reason": reason,
        "stdout_tail": stdout[-4096:],
        "stderr_tail": stderr[-4096:],
        "returncode": None,
    }


def _result_with_optional_staging_cleanup(
    result: dict[str, Any],
    *,
    staging_dir: Path,
    keep_staging: bool,
) -> dict[str, Any]:
    if not keep_staging:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return result


def _write_blocked_sidecar(
    target: ProfilerTimingProblemCoverage,
    *,
    replacement_path: Path,
    staging_dir: Path | None,
    reason: str,
    workload_offset: int = 0,
    workload_limit: int | None = None,
    stdout: str = "",
    stderr: str = "",
    concurrent_gpu_processes: list[dict[str, Any]] | None = None,
    pid_lock_contention: bool = False,
    reference_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workload_slice_applied = workload_offset != 0 or workload_limit is not None
    payload: dict[str, Any] = {
        "profiler_collected": False,
        "csv_path": None,
        "selection": {
            "reason": reason,
            "policy": forced_pytorch_kernel_activity_policy().to_dict(),
        },
        "evidence": {
            "backend": "rocprofv3",
            "parsed_rows": [],
        },
        "replacement_metadata": {
            "schema_version": "sol_execbench.rdna4_profiler_timing_replacement.v1",
            "problem_id": target.problem_id,
            "replacement_status": "profiler_blocked",
            "profiled_workload_count": 0,
            "expected_workload_count": target.workload_count,
            "trace_status_counts": {},
            "all_workloads_passed": False,
            "workload_limit_applied": workload_limit,
            "workload_offset": workload_offset,
            "workload_slice_applied": workload_slice_applied,
            "workload_slice": {
                "offset": workload_offset,
                "limit": workload_limit,
                "selected_workload_count": 0,
            },
            "full_workload_coverage": False,
            "failure_reason": reason,
            "reference_override": reference_override,
        },
    }
    if concurrent_gpu_processes:
        payload["concurrent_gpu_processes"] = concurrent_gpu_processes
    if pid_lock_contention:
        payload["pid_lock_contention"] = True
    replacement_path.parent.mkdir(parents=True, exist_ok=True)
    replacement_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _target_failure(
        target,
        replacement_path=replacement_path,
        staging_dir=staging_dir,
        reason=reason,
        stdout=stdout,
        stderr=stderr,
    )


def _mark_blocked_targets(
    coverage: ProfilerTimingCoverageReport,
    *,
    replacement_timing_dir: Path,
    problem_ids: Sequence[str],
    reason: str,
) -> list[dict[str, Any]]:
    if not problem_ids:
        return []
    problems = {problem.problem_id: problem for problem in coverage.problems}
    results: list[dict[str, Any]] = []
    for problem_id in problem_ids:
        target = problems.get(problem_id)
        if target is None:
            raise ValueError(
                f"unknown problem for blocked classification: {problem_id}"
            )
        replacement_path = _replacement_timing_path(
            replacement_timing_dir,
            target.category,
            target.problem_path,
        )
        results.append(
            _write_blocked_sidecar(
                target,
                replacement_path=replacement_path,
                staging_dir=None,
                reason=reason,
            )
        )
    return results


def _replacement_timing_path(
    replacement_timing_dir: Path,
    category: str,
    problem_path: str,
) -> Path:
    return replacement_timing_dir / category / f"{Path(problem_path).name}.timing.json"


def _workload_manifest_path(
    output_dir: Path,
    target: ProfilerTimingProblemCoverage,
) -> Path:
    return (
        output_dir
        / "workload-manifests"
        / target.category
        / f"{Path(target.problem_path).name}.workload-profiler-manifest.json"
    )


def _load_workload_refs(path: Path) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected workload JSON object at row {index}: {path}")
        refs.append(
            {
                "workload_index": index,
                "row_index": index,
                "workload_uuid": (
                    str(payload["uuid"]) if payload.get("uuid") is not None else None
                ),
            }
        )
    if not refs:
        raise ValueError(f"workload file has no records: {path}")
    return refs


def _imported_workload_slice_sidecars(
    target: ProfilerTimingProblemCoverage,
    timing_dirs: Sequence[Path],
) -> dict[int, Path]:
    sidecars: dict[int, Path] = {}
    complete_by_offset: dict[int, bool] = {}
    for timing_dir in timing_dirs:
        path = _replacement_timing_path(
            timing_dir,
            target.category,
            target.problem_path,
        )
        payload = _load_optional_json(path)
        if payload is None:
            continue
        metadata: dict[str, Any] = cast(
            dict[str, Any],
            payload.get("replacement_metadata")
            if isinstance(payload.get("replacement_metadata"), dict)
            else {},
        )
        if metadata.get("workload_slice_applied") is not True:
            continue
        offset = _int_value(metadata.get("workload_offset"))
        complete = _workload_slice_sidecar_complete(payload)
        if offset not in sidecars or (complete and not complete_by_offset[offset]):
            sidecars[offset] = path
            complete_by_offset[offset] = complete
    return sidecars


def _apply_rdna4_reference_override(
    definition_payload: dict[str, Any],
    problem_id: str,
) -> dict[str, Any] | None:
    reference = RDNA4_REFERENCE_OVERRIDES.get(problem_id)
    if reference is None:
        return None
    definition_payload["reference"] = reference
    metadata = {
        "problem_id": problem_id,
        **RDNA4_REFERENCE_OVERRIDE_METADATA.get(problem_id, {}),
    }
    return metadata


def _workload_slice_sidecar_complete(payload: dict[str, Any]) -> bool:
    evidence: dict[str, Any] = cast(
        dict[str, Any],
        payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {},
    )
    metadata: dict[str, Any] = cast(
        dict[str, Any],
        payload.get("replacement_metadata")
        if isinstance(payload.get("replacement_metadata"), dict)
        else {},
    )
    trace_counts = _trace_status_counts_from_mapping(
        metadata.get("trace_status_counts")
    )
    return (
        payload.get("profiler_collected") is True
        and evidence.get("backend") == "rocprofv3"
        and _sidecar_kernel_activity_rows(evidence) > 0
        and trace_counts.get("PASSED", 0) >= 1
    )


def _csv_path_from_sidecar(path: Path) -> str | None:
    payload = _load_optional_json(path)
    if payload is None or payload.get("csv_path") is None:
        return None
    return str(payload["csv_path"])


def _load_or_create_workload_manifest(
    *,
    target: ProfilerTimingProblemCoverage,
    dataset_root: Path,
    workload_refs: Sequence[dict[str, Any]],
    manifest_path: Path,
    tool_version: str,
    gpu_architecture: str,
    clock_locked: bool,
) -> dict[str, Any]:
    if manifest_path.is_file():
        payload = _load_json(manifest_path)
        if payload.get("schema_version") != WORKLOAD_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unknown workload manifest schema: {manifest_path}")
        if payload.get("problem_id") != target.problem_id:
            raise ValueError(f"workload manifest problem mismatch: {manifest_path}")
        payload["manifest_path"] = manifest_path.as_posix()
        _normalize_workload_manifest(payload)
        return payload
    payload = {
        "schema_version": WORKLOAD_MANIFEST_SCHEMA_VERSION,
        "manifest_path": manifest_path.as_posix(),
        "problem_id": target.problem_id,
        "category": target.category,
        "problem_path": target.problem_path,
        "dataset_root": dataset_root.as_posix(),
        "expected_workload_count": len(workload_refs),
        "expected_workloads": list(workload_refs),
        "tool_version": tool_version,
        "gpu_architecture": gpu_architecture,
        "clock_locked": clock_locked,
        "workloads": [],
    }
    _normalize_workload_manifest(payload)
    return payload


def _normalize_workload_manifest(manifest: dict[str, Any]) -> None:
    entries = manifest.get("workloads")
    if not isinstance(entries, list):
        manifest["workloads"] = []
        return
    for entry in entries:
        if isinstance(entry, dict) and _workload_manifest_entry_complete(entry):
            entry["status"] = "profiler_backed"
            entry["retryable"] = False
            entry["failure_reason"] = None


def _write_workload_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _workload_manifest_entry_complete(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    payload = cast(dict[str, Any], entry)
    return (
        payload.get("status") == "profiler_backed"
        and payload.get("profiler_collected") is True
        and _int_value(payload.get("kernel_activity_rows")) > 0
        and _trace_status_counts_from_mapping(payload.get("trace_status_counts")).get(
            "PASSED", 0
        )
        >= 1
    )


def _workload_manifest_entry(
    ref: dict[str, Any],
    *,
    result: dict[str, Any],
    sidecar: dict[str, Any] | None,
) -> dict[str, Any]:
    sidecar = sidecar or {}
    evidence: dict[str, Any] = cast(
        dict[str, Any],
        sidecar.get("evidence") if isinstance(sidecar.get("evidence"), dict) else {},
    )
    metadata: dict[str, Any] = cast(
        dict[str, Any],
        sidecar.get("replacement_metadata")
        if isinstance(sidecar.get("replacement_metadata"), dict)
        else {},
    )
    trace_counts = _trace_status_counts_from_mapping(result.get("trace_status_counts"))
    if not trace_counts:
        trace_counts = _trace_status_counts_from_mapping(
            metadata.get("trace_status_counts")
        )
    kernel_rows = _int_value(result.get("kernel_activity_rows"))
    if kernel_rows <= 0:
        kernel_rows = _sidecar_kernel_activity_rows(evidence)
    kernel_duration_ms = _float_value(result.get("kernel_duration_ms"))
    if kernel_duration_ms <= 0:
        kernel_duration_ms = _float_value(evidence.get("kernel_duration_ms"))
    profiler_collected = (
        result.get("profiler_collected") is True
        or sidecar.get("profiler_collected") is True
    )
    backend = evidence.get("backend")
    if backend is None and profiler_collected:
        backend = "rocprofv3"
    passed = trace_counts.get("PASSED", 0) >= 1
    status = (
        "profiler_backed"
        if profiler_collected and backend == "rocprofv3" and kernel_rows > 0 and passed
        else result["status"]
    )
    failure_reason = (
        None
        if status == "profiler_backed"
        else (
            metadata.get("failure_reason")
            if metadata.get("failure_reason") is not None
            else result.get("fallback_reason")
        )
    )
    return {
        **ref,
        "status": status,
        "retryable": status != "profiler_backed",
        "profiler_collected": profiler_collected,
        "backend": backend,
        "trace_status_counts": trace_counts,
        "kernel_activity_rows": kernel_rows,
        "kernel_duration_ms": kernel_duration_ms,
        "replacement_path": result.get("replacement_path"),
        "csv_path": result.get("csv_path"),
        "failure_reason": failure_reason,
    }


def _upsert_workload_manifest_entry(
    manifest: dict[str, Any],
    entry: dict[str, Any],
) -> None:
    entries = [
        item
        for item in manifest.get("workloads", [])
        if isinstance(item, dict)
        and item.get("workload_index") != entry.get("workload_index")
    ]
    entries.append(entry)
    manifest["workloads"] = sorted(
        entries,
        key=lambda item: int(item.get("workload_index", -1)),
    )


def _write_workload_aggregate_sidecar(
    *,
    target: ProfilerTimingProblemCoverage,
    manifest: dict[str, Any],
    replacement_path: Path,
    tool_version: str,
    gpu_architecture: str,
    clock_locked: bool,
) -> dict[str, Any]:
    entries = [
        entry for entry in manifest.get("workloads", []) if isinstance(entry, dict)
    ]
    expected = _int_value(manifest.get("expected_workload_count"))
    complete_entries = [
        entry for entry in entries if _workload_manifest_entry_complete(entry)
    ]
    trace_counts: dict[str, int] = {}
    source_entries: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    kernel_duration_ms = 0.0
    for entry in entries:
        for status, count in _trace_status_counts_from_mapping(
            entry.get("trace_status_counts")
        ).items():
            trace_counts[status] = trace_counts.get(status, 0) + count
        kernel_duration_ms += _float_value(entry.get("kernel_duration_ms"))
        source_entries.append(
            {
                "workload_index": entry.get("workload_index"),
                "workload_uuid": entry.get("workload_uuid"),
                "status": entry.get("status"),
                "replacement_path": entry.get("replacement_path"),
                "csv_path": entry.get("csv_path"),
                "kernel_activity_rows": entry.get("kernel_activity_rows"),
                "kernel_duration_ms": entry.get("kernel_duration_ms"),
                "failure_reason": (
                    None
                    if _workload_manifest_entry_complete(entry)
                    else entry.get("failure_reason")
                ),
            }
        )
        parsed_rows.extend(_aggregate_rows_from_entry(entry))
    full_workload_coverage = expected > 0 and len(complete_entries) == expected
    profiler_collected = full_workload_coverage or any(
        entry.get("profiler_collected") is True for entry in entries
    )
    status = _replacement_status(
        profiler_collected=profiler_collected,
        full_workload_coverage=full_workload_coverage,
        kernel_activity_rows=len(parsed_rows),
    )
    failure_reason = None
    if not full_workload_coverage:
        failure_reason = (
            "workload-sharded profiler manifest is incomplete or has failed workloads"
        )
    payload = {
        "profiler_collected": profiler_collected,
        "csv_path": None,
        "selection": {
            "reason": (
                "complete workload-sharded rocprofv3 aggregation"
                if full_workload_coverage
                else failure_reason
            ),
            "policy": forced_pytorch_kernel_activity_policy().to_dict(),
        },
        "evidence": {
            "schema_version": WORKLOAD_AGGREGATE_SCHEMA_VERSION,
            "derived": True,
            "backend": "rocprofv3",
            "tool_version": tool_version,
            "gpu_architecture": gpu_architecture,
            "activity_domain": "kernel_activity",
            "aggregation_rule": (
                "sum kernel activity rows from independently profiled workloads"
            ),
            "interpretation": (
                "problem-level timing aggregated from complete workload-sharded "
                "rocprofv3 profiler evidence"
            ),
            "clock_locked": clock_locked,
            "fallback_applied": False,
            "kernel_duration_ms": kernel_duration_ms,
            "parsed_rows": parsed_rows,
            "source_manifest_path": manifest.get("manifest_path"),
            "source_workloads": source_entries,
        },
        "replacement_metadata": {
            "schema_version": "sol_execbench.rdna4_profiler_timing_replacement.v1",
            "aggregation_schema_version": WORKLOAD_AGGREGATE_SCHEMA_VERSION,
            "problem_id": target.problem_id,
            "replacement_status": status,
            "profiled_workload_count": len(complete_entries),
            "expected_workload_count": expected,
            "trace_status_counts": dict(sorted(trace_counts.items())),
            "all_workloads_passed": full_workload_coverage,
            "workload_limit_applied": None,
            "workload_offset": 0,
            "workload_slice_applied": False,
            "workload_sharded_aggregation": True,
            "manifest_path": manifest.get("manifest_path"),
            "full_workload_coverage": full_workload_coverage,
            "failure_reason": failure_reason,
        },
    }
    payload["evidence"]["source_manifest_path"] = manifest.get("manifest_path")
    payload["replacement_metadata"]["manifest_path"] = manifest.get("manifest_path")
    replacement_path.parent.mkdir(parents=True, exist_ok=True)
    replacement_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": status,
        "replacement_path": str(replacement_path),
        "profiler_collected": profiler_collected,
        "full_workload_coverage": full_workload_coverage,
        "profiled_workload_count": len(complete_entries),
        "expected_workload_count": expected,
        "failure_reason": failure_reason,
    }


def _aggregate_rows_from_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    count = _int_value(entry.get("kernel_activity_rows"))
    duration = _float_value(entry.get("kernel_duration_ms"))
    if count <= 0 or duration <= 0:
        rows = _load_sidecar_rows(entry.get("replacement_path"))
        if rows:
            for row in rows:
                row["workload_index"] = entry.get("workload_index")
                row["workload_uuid"] = entry.get("workload_uuid")
            return rows
        return []
    return [
        {
            "name": "workload_sharded_kernel_activity",
            "domain": "KERNEL_DISPATCH",
            "duration_ns": duration * 1_000_000.0,
            "duration_ms": duration,
            "is_kernel_activity": True,
            "raw": {},
            "workload_index": entry.get("workload_index"),
            "workload_uuid": entry.get("workload_uuid"),
        }
    ]


def _compact_payload_parsed_rows(payload: dict[str, Any], *, reason: str) -> None:
    evidence = (
        payload.get("evidence") if isinstance(payload.get("evidence"), dict) else None
    )
    if evidence is None or not isinstance(evidence.get("parsed_rows"), list):
        return
    evidence["parsed_rows"] = []
    evidence["parsed_rows_compacted"] = True
    evidence["compaction_reason"] = reason


def _compact_workload_slice_artifacts(
    *,
    sidecar_path: Path,
    csv_path: object,
) -> None:
    payload = _load_optional_json(sidecar_path)
    if payload is not None:
        _compact_payload_parsed_rows(
            payload,
            reason="workload manifest retained kernel row and duration summaries",
        )
        sidecar_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if isinstance(csv_path, str) and csv_path:
        _remove_rocprofv3_run_dir(Path(csv_path))


def _remove_rocprofv3_run_dir(csv_path: Path) -> None:
    try:
        run_dir = csv_path.resolve().parent
    except OSError:
        run_dir = csv_path.parent
    marker = f"{os.sep}rocprofv3{os.sep}"
    if marker not in str(run_dir):
        return
    shutil.rmtree(run_dir, ignore_errors=True)


def _load_sidecar_rows(path_value: object) -> list[dict[str, Any]]:
    if not isinstance(path_value, str):
        return []
    payload = _load_optional_json(Path(path_value))
    if payload is None:
        return []
    evidence: dict[str, Any] = cast(
        dict[str, Any],
        payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {},
    )
    rows = evidence.get("parsed_rows")
    if not isinstance(rows, list):
        return []
    return [
        dict(row)
        for row in rows
        if isinstance(row, dict) and row.get("is_kernel_activity") is True
    ]


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _is_classified_replacement_sidecar(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("backend") != "rocprofv3":
        return False
    metadata = payload.get("replacement_metadata")
    if not isinstance(metadata, dict):
        if payload.get("profiler_collected") is not True:
            return False
        return _kernel_activity_rows(evidence) > 0
    if metadata.get("workload_limit_applied") is not None:
        return False
    if metadata.get("workload_slice_applied") is True:
        return False
    return (
        metadata.get("replacement_status") == "profiler_backed"
        and metadata.get("full_workload_coverage") is True
    )


def _full_workload_coverage(payload: dict[str, Any]) -> bool:
    metadata = payload.get("replacement_metadata")
    if not isinstance(metadata, dict):
        return True
    return metadata.get("full_workload_coverage") is True


def _kernel_activity_rows(evidence: dict[str, Any]) -> int:
    parsed_rows = evidence.get("parsed_rows")
    if isinstance(parsed_rows, list):
        return sum(
            1
            for row in parsed_rows
            if isinstance(row, dict) and row.get("is_kernel_activity") is True
        )
    try:
        return 1 if float(evidence.get("kernel_duration_ms") or 0) > 0 else 0
    except (TypeError, ValueError):
        return 0


def _sidecar_kernel_activity_rows(evidence: dict[str, Any]) -> int:
    kernel_rows = _kernel_activity_rows(evidence)
    if kernel_rows > 0:
        return kernel_rows
    if (
        evidence.get("parsed_rows_compacted") is True
        and _float_value(evidence.get("kernel_duration_ms")) > 0
    ):
        return 1
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    return load_json_dict(path)


def _load_workloads(
    path: Path, *, limit: int | None, offset: int = 0
) -> list[Workload]:
    workloads: list[Workload] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        if index < offset:
            continue
        workloads.append(Workload.model_validate_json(line))
        if limit is not None and len(workloads) >= limit:
            break
    if not workloads:
        raise ValueError(f"workload file has no records: {path}")
    return workloads


_DTYPE_NBYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "float4_e2m1fn_x2": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}


def _resolve_estimated_timing_input_cap(
    *,
    max_estimated_timing_input_gib: float | None,
    auto_estimated_timing_input_cap: bool,
    subprocess_memory_limit_gib: float | None,
) -> tuple[float | None, str | None]:
    if max_estimated_timing_input_gib is not None:
        if max_estimated_timing_input_gib <= 0:
            raise ValueError("max estimated timing input GiB must be positive")
        return max_estimated_timing_input_gib, "manual"
    if not auto_estimated_timing_input_cap:
        return None, None
    available_bytes = _dynamic_available_memory_bytes(
        subprocess_memory_limit_gib=subprocess_memory_limit_gib,
    )
    if available_bytes is None or available_bytes <= 0:
        return None, None
    cap_bytes = int(available_bytes * AUTO_TIMING_INPUT_CAP_FRACTION)
    return cap_bytes / (1024 * 1024 * 1024), (
        f"dynamic_available_memory:{AUTO_TIMING_INPUT_CAP_FRACTION:.0%}"
    )


def _dynamic_available_memory_bytes(
    *,
    subprocess_memory_limit_gib: float | None,
) -> int | None:
    candidates: list[int] = []
    mem_available = _proc_mem_available_bytes()
    if mem_available is not None:
        candidates.append(mem_available)
    cgroup_available = _cgroup_available_memory_bytes()
    if cgroup_available is not None:
        candidates.append(cgroup_available)
    if subprocess_memory_limit_gib is not None:
        if subprocess_memory_limit_gib <= 0:
            raise ValueError("subprocess memory limit must be positive")
        candidates.append(int(subprocess_memory_limit_gib * 1024 * 1024 * 1024))
    return min(candidates) if candidates else None


def _proc_mem_available_bytes(path: Path = Path("/proc/meminfo")) -> int | None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("MemAvailable:"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _cgroup_available_memory_bytes(root: Path = Path("/sys/fs/cgroup")) -> int | None:
    max_path = root / "memory.max"
    current_path = root / "memory.current"
    try:
        raw_max = max_path.read_text(encoding="utf-8").strip()
        if raw_max == "max":
            return None
        memory_max = int(raw_max)
        memory_current = int(current_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return max(memory_max - memory_current, 0)


def _estimated_timing_input_block_reason(
    definition_payload: dict[str, Any],
    workloads: Sequence[Workload],
    *,
    max_estimated_timing_input_gib: float | None,
    cap_source: str | None = None,
) -> str | None:
    if max_estimated_timing_input_gib is None:
        return None
    if max_estimated_timing_input_gib <= 0:
        raise ValueError("max estimated timing input GiB must be positive")
    max_estimated_bytes = int(max_estimated_timing_input_gib * 1024 * 1024 * 1024)
    peak_input_bytes = 0
    peak_uuid: str | None = None
    for workload in workloads:
        input_bytes = _estimate_workload_input_bytes(definition_payload, workload)
        if input_bytes > peak_input_bytes:
            peak_input_bytes = input_bytes
            peak_uuid = workload.uuid

    # time_runnable builds a shifting pool for every input tensor, so large
    # tensor inputs are held once by the generated workload and once by the
    # timing pool before any kernel intermediates are allocated.
    estimated_timing_input_bytes = peak_input_bytes * 2
    if estimated_timing_input_bytes <= max_estimated_bytes:
        return None
    return (
        "estimated timing input footprint exceeds preflight cap: "
        f"input={_format_gib(peak_input_bytes)} GiB, "
        f"timing_pool_peak={_format_gib(estimated_timing_input_bytes)} GiB, "
        f"cap={_format_gib(max_estimated_bytes)} GiB, "
        f"cap_source={cap_source or 'manual'}, "
        f"workload_uuid={peak_uuid or '<unknown>'}"
    )


def _estimate_workload_input_bytes(
    definition_payload: dict[str, Any],
    workload: Workload,
) -> int:
    axes = dict(workload.axes)
    resolved_axes = _resolve_definition_axes(definition_payload, axes)
    total = 0
    inputs = definition_payload.get("inputs")
    if not isinstance(inputs, dict):
        return 0
    for spec in inputs.values():
        if not isinstance(spec, dict):
            continue
        shape = spec.get("shape")
        dtype = spec.get("dtype")
        if not isinstance(shape, list) or not isinstance(dtype, str):
            continue
        elem_count = _resolve_shape_numel(shape, resolved_axes)
        dtype_nbytes = _DTYPE_NBYTES.get(dtype)
        if elem_count is None or dtype_nbytes is None:
            continue
        total += elem_count * dtype_nbytes
    return total


def _resolve_definition_axes(
    definition_payload: dict[str, Any],
    workload_axes: dict[str, Any],
) -> dict[str, int]:
    resolved: dict[str, int] = {}
    axes = definition_payload.get("axes")
    if isinstance(axes, dict):
        for name, spec in axes.items():
            if not isinstance(name, str) or not isinstance(spec, dict):
                continue
            if spec.get("type") == "const" and spec.get("value") is not None:
                try:
                    resolved[name] = int(spec["value"])
                except (TypeError, ValueError):
                    pass
    for name, value in workload_axes.items():
        try:
            resolved[str(name)] = int(value)
        except (TypeError, ValueError):
            continue
    return resolved


def _resolve_shape_numel(
    shape: Sequence[Any],
    axes: dict[str, int],
) -> int | None:
    numel = 1
    for dim in shape:
        if isinstance(dim, int):
            value = dim
        elif isinstance(dim, str):
            if dim not in axes:
                return None
            value = axes[dim]
        else:
            return None
        if value < 0:
            return None
        numel *= value
    return numel


def _format_gib(value: int) -> str:
    return f"{value / (1024 * 1024 * 1024):.2f}"


def _torch_rocm_gpu_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def _memory_limit_preexec(memory_limit_gib: float | None):
    if memory_limit_gib is None:
        return None
    if memory_limit_gib <= 0:
        raise ValueError("subprocess memory limit must be positive")
    limit_bytes = int(memory_limit_gib * 1024 * 1024 * 1024)

    def apply_limit() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    return apply_limit


def _staging_runner(
    staging_dir: Path,
    *,
    timeout: int,
    memory_limit_gib: float | None = None,
) -> ProfilerRunner:
    def run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "PYTHONPATH": _pythonpath_with_src(),
            "SOL_EXECBENCH_GRACEFUL_EXIT": "1",
            "TMPDIR": str(staging_dir.parent.resolve()),
        }
        preexec_fn = _memory_limit_preexec(memory_limit_gib)
        return run_in_process_group(
            command,
            cwd=staging_dir,
            env=env,
            timeout=timeout,
            preexec_fn=preexec_fn,
        )

    return run


def _pythonpath_with_src() -> str:
    src = str(Path(__file__).resolve().parents[3] / "src")
    existing = os.environ.get("PYTHONPATH")
    return src if not existing else f"{src}{os.pathsep}{existing}"


def _trace_status_counts(stdout: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        evaluation = payload.get("evaluation")
        if not isinstance(evaluation, dict):
            continue
        status = evaluation.get("status")
        if isinstance(status, str):
            counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _trace_status_counts_from_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, count_value in value.items():
        if isinstance(key, str):
            count = _int_value(count_value)
            if count:
                counts[key] = count
    return dict(sorted(counts.items()))


def _int_value(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _float_value(value: object) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _replacement_failure_reason(
    *,
    profiler_collected: bool,
    all_workloads_passed: bool,
    selection_reason: str,
) -> str | None:
    if profiler_collected and all_workloads_passed:
        return None
    if profiler_collected:
        return "replacement did not produce PASSED traces for every workload"
    return selection_reason


def _replacement_status(
    *,
    profiler_collected: bool,
    full_workload_coverage: bool,
    kernel_activity_rows: int,
) -> str:
    if profiler_collected and full_workload_coverage and kernel_activity_rows > 0:
        return "profiler_backed"
    if profiler_collected and kernel_activity_rows > 0:
        return "partial_profiler_backed"
    return "profiler_blocked"


def _build_summary(
    *,
    coverage: ProfilerTimingCoverageReport,
    selected_targets: Sequence[ProfilerTimingProblemCoverage],
    results: Sequence[dict[str, Any]],
    output_dir: Path,
    replacement_timing_dir: Path,
    source_timing_dirs: Sequence[Path],
    rocprofv3_available: bool,
    interrupted: bool = False,
) -> dict[str, Any]:
    succeeded = sum(1 for result in results if result["status"] == "profiler_backed")
    partial = sum(
        1 for result in results if result["status"] == "partial_profiler_backed"
    )
    profiler_blocked = sum(
        1 for result in results if result["status"] == "profiler_blocked"
    )
    failed = profiler_blocked
    fallback_or_missing = sum(
        1
        for result in results
        if result["status"]
        not in {"profiler_backed", "partial_profiler_backed", "profiler_blocked"}
    )
    return {
        "schema_version": "sol_execbench.rdna4_profiler_timing_batch.v1",
        "coverage_problem_denominator": coverage.totals.problem_denominator,
        "coverage_fallback_timing_problems": coverage.totals.fallback_timing_problems,
        "selected_targets": len(selected_targets),
        "succeeded": succeeded,
        "partial_profiler_backed": partial,
        "profiler_blocked": profiler_blocked,
        "failed": failed,
        "fallback_or_missing": fallback_or_missing,
        "rocprofv3_available": rocprofv3_available,
        "output_dir": str(output_dir),
        "replacement_timing_dir": str(replacement_timing_dir),
        "source_timing_dirs": [str(path) for path in source_timing_dirs],
        "claim_boundary": CLAIM_BOUNDARY,
        "results": list(results),
        "timing_isolation_snapshot": collect_timing_environment_snapshot(),
        "interrupted": interrupted,
    }


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# RDNA4 Profiler-Backed Timing Batch",
            "",
            f"- Selected targets: `{summary['selected_targets']}`",
            f"- Succeeded: `{summary['succeeded']}`",
            f"- Partial profiler-backed: `{summary['partial_profiler_backed']}`",
            f"- Profiler-blocked: `{summary['profiler_blocked']}`",
            f"- Failed: `{summary['failed']}`",
            f"- Fallback or missing: `{summary['fallback_or_missing']}`",
            f"- Replacement timing dir: `{summary['replacement_timing_dir']}`",
            "",
            "## Claim Boundary",
            "",
            summary["claim_boundary"],
            "",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-timing-dir",
        action="append",
        type=Path,
        default=None,
        help="Existing fallback timing root; may be provided more than once.",
    )
    parser.add_argument("--replacement-timing-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-problem", action="append", default=[])
    parser.add_argument("--only-problem-file", type=Path, default=None)
    parser.add_argument("--skip-problem", action="append", default=[])
    parser.add_argument("--skip-problem-file", type=Path, default=None)
    parser.add_argument("--mark-blocked-problem", action="append", default=[])
    parser.add_argument("--mark-blocked-only", action="store_true")
    parser.add_argument("--workload-limit", type=int, default=None)
    parser.add_argument("--workload-offset", type=int, default=0)
    parser.add_argument(
        "--workload-sharded",
        action="store_true",
        help=(
            "Profile each workload independently and aggregate only complete "
            "profiler-backed workload manifests into problem-level timing."
        ),
    )
    parser.add_argument(
        "--workload-slice-timing-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "Existing timing root containing diagnostic workload slice sidecars "
            "to import before profiling missing workloads; may be repeated."
        ),
    )
    parser.add_argument(
        "--workload-sharded-import-only",
        action="store_true",
        help=(
            "In workload-sharded mode, aggregate imported slice sidecars only "
            "and do not profile missing workloads."
        ),
    )
    parser.add_argument(
        "--compact-workload-slices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "In workload-sharded mode, compact completed slice timing sidecars "
            "and remove raw rocprofv3 run directories after manifest summaries "
            "have been recorded."
        ),
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--subprocess-memory-limit-gib",
        type=float,
        default=None,
        help=(
            "Optional address-space limit for staged profiler subprocesses. "
            "Use this for large RDNA4 closure targets so one workload cannot "
            "trigger a system-wide OOM kill."
        ),
    )
    parser.add_argument(
        "--max-estimated-timing-input-gib",
        type=float,
        default=None,
        help=(
            "Manual preflight cap for estimated timing input footprint. "
            "The estimate accounts for the generated input tensors and the "
            "timing allocator's input pool copy; workloads above the cap are "
            "classified as profiler-blocked before launching the subprocess. "
            "When omitted, the batch uses a dynamic cap derived from current "
            "host/cgroup availability unless --no-auto-estimated-timing-input-cap "
            "is set."
        ),
    )
    parser.add_argument(
        "--auto-estimated-timing-input-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Dynamically cap estimated timing input footprint from current "
            "MemAvailable, cgroup remaining memory, and subprocess memory limit. "
            "Enabled by default; use --no-auto-estimated-timing-input-cap to "
            "disable preflight unless a manual cap is provided."
        ),
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help=(
            "Directory for staged eval_driver packages. Defaults to "
            "tmp/<output-dir-name> so validation temporaries stay under the "
            "project tmp directory."
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help=(
            "Maximum number of target-level workers. Use --max-workers 1 for "
            "resource-sensitive GPU profiling runs that must avoid concurrent "
            "eval_driver processes."
        ),
    )
    parser.add_argument("--timing-tool-version", default=DEFAULT_TOOL_VERSION)
    parser.add_argument(
        "--profiler-executable",
        type=str,
        default=None,
        help=(
            "Explicit rocprofv3 executable or gfx1200 patched wrapper. The exact "
            "value is recorded in every profiler command; when omitted, resolve "
            "rocprofv3 through the active ROCm root."
        ),
    )
    parser.add_argument("--gpu-architecture", default=DEFAULT_GPU_ARCHITECTURE)
    parser.add_argument(
        "--clock-locked",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hip-runtime-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include rocprofv3 HIP runtime API tracing alongside kernel tracing. "
            "Disabled by default because HIP API CSVs can grow to GiB-scale; "
            "enable only for profiler debugging."
        ),
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        default=False,
        help=(
            "Keep per-target staging directories after each profiler subprocess. "
            "Disabled by default to avoid accumulating temporary files."
        ),
    )
    parser.add_argument(
        "--keep-rocprofv3-csv",
        action="store_true",
        default=False,
        help=(
            "Keep raw rocprofv3 CSV run directories after compact timing sidecars "
            "are written. Disabled by default to avoid large trace artifacts."
        ),
    )
    parser.add_argument(
        "--strict-isolation",
        action="store_true",
        default=False,
        help=(
            "Abort on any timing isolation check failure (concurrent GPU processes, "
            "clock state, GPU device isolation) instead of warning. Recommended for "
            "data-measurement-sensitive profiling runs."
        ),
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=None,
        help=(
            "Set ROCR_VISIBLE_DEVICES to this device index for GPU device isolation. "
            "Recommended on multi-GPU systems to prevent cross-device interference."
        ),
    )
    parser.add_argument(
        "--calibration-path",
        type=Path,
        default=None,
        help=(
            "Path to a rocprofv3 overhead calibration JSON sidecar produced by "
            "run_rdna4_profiler_overhead_calibration.py. When provided, the profiler "
            "overhead value is included in timing evidence payloads."
        ),
    )
    return parser.parse_args(argv)


def _load_problem_id_file(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    problem_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        problem_ids.append(stripped)
    return tuple(problem_ids)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_timing_dirs = (
        tuple(args.source_timing_dir)
        if args.source_timing_dir
        else (DEFAULT_SOURCE_TIMING_DIR,)
    )
    skip_problem = tuple(args.skip_problem) + _load_problem_id_file(
        args.skip_problem_file
    )
    only_problem = tuple(args.only_problem) + _load_problem_id_file(
        args.only_problem_file
    )
    temp_dir = args.temp_dir or DEFAULT_TEMP_ROOT / args.output_dir.name
    try:
        with acquire_pid_lock(args.output_dir):
            return run_batch(
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                source_timing_dirs=source_timing_dirs,
                replacement_timing_dir=args.replacement_timing_dir,
                limit=args.limit,
                only_problem=only_problem,
                skip_problem=skip_problem,
                mark_blocked_problem=tuple(args.mark_blocked_problem),
                mark_blocked_only=args.mark_blocked_only,
                workload_limit=args.workload_limit,
                workload_offset=args.workload_offset,
                workload_sharded=args.workload_sharded,
                workload_slice_timing_dirs=tuple(args.workload_slice_timing_dir),
                workload_sharded_import_only=args.workload_sharded_import_only,
                timeout=args.timeout,
                subprocess_memory_limit_gib=args.subprocess_memory_limit_gib,
                max_estimated_timing_input_gib=args.max_estimated_timing_input_gib,
                auto_estimated_timing_input_cap=args.auto_estimated_timing_input_cap,
                temp_dir=temp_dir,
                resume=args.resume,
                tool_version=args.timing_tool_version,
                gpu_architecture=args.gpu_architecture,
                clock_locked=args.clock_locked,
                include_hip_runtime_trace=args.hip_runtime_trace,
                strict_isolation=args.strict_isolation,
                gpu_device=args.gpu_device,
                calibration_path=args.calibration_path,
                compact_workload_slices=args.compact_workload_slices,
                keep_staging=args.keep_staging,
                keep_rocprofv3_csv=args.keep_rocprofv3_csv,
                profiler_executable=args.profiler_executable,
                max_workers=args.max_workers,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
