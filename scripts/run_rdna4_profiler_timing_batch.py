# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Replace RDNA4 fallback timing sidecars with profiler-backed rocprofv3 timing."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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
from sol_execbench.core.dataset import (
    ProfilerTimingCoverageReport,
    build_dataset_inventory,
    build_profiler_timing_coverage_report,
    classify_rocm_readiness,
)
from sol_execbench.core.dataset.profiler_timing_coverage import (
    ProfilerTimingProblemCoverage,
)
from sol_execbench.core.dataset.runner import build_reference_solution
from sol_execbench.driver import ProblemPackager
from sol_execbench.core.bench.pid_lock import acquire_pid_lock
from sol_execbench.core.bench.timing_isolation import (
    clear_gpu_cache_between_subprocesses,
    collect_timing_environment_snapshot,
    detect_concurrent_gpu_processes,
    verify_clock_state_with_warning,
)

logger = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT = Path("data/SOL-ExecBench/benchmark")
DEFAULT_SOURCE_TIMING_DIR = Path("out/rdna4-timing-evidence/timing")
DEFAULT_OUTPUT_DIR = Path("out/rdna4-profiler-backed-timing")
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
    temp_dir: Path | None = None,
    tool_version: str = DEFAULT_TOOL_VERSION,
    gpu_architecture: str = DEFAULT_GPU_ARCHITECTURE,
    clock_locked: bool = True,
    rocprofv3_available: bool,
    runner: ProfilerRunner,
    marked_blocked_ids: set[str],
    global_start_index: int = 0,
    concurrent_gpu_processes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Process a chunk of targets with CPU-parallel staging + serial GPU profiling (PRFL-01, PRFL-02)."""
    chunk_results = []
    for local_idx, target in enumerate(chunk, 1):
        global_target_index = global_start_index + local_idx
        if target.problem_id in marked_blocked_ids:
            continue

        # Re-verify clock state periodically (every 10 problems globally)
        if global_target_index % 10 == 0:
            verify_clock_state_with_warning(context=f"problem_{global_target_index}")

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
                temp_dir=temp_dir,
                tool_version=tool_version,
                gpu_architecture=gpu_architecture,
                clock_locked=clock_locked,
                rocprofv3_available=rocprofv3_available,
                runner=runner,
                concurrent_gpu_processes=concurrent_gpu_processes,
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
                temp_dir=temp_dir,
                tool_version=tool_version,
                gpu_architecture=gpu_architecture,
                clock_locked=clock_locked,
                rocprofv3_available=rocprofv3_available,
                runner=runner,
                concurrent_gpu_processes=concurrent_gpu_processes,
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
    temp_dir: Path | None = None,
    resume: bool = True,
    tool_version: str = DEFAULT_TOOL_VERSION,
    gpu_architecture: str = DEFAULT_GPU_ARCHITECTURE,
    clock_locked: bool = True,
    rocprofv3_available: bool | None = None,
    runner: ProfilerRunner | None = None,
    max_workers: int = 4,
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
    available = (
        shutil.which(ROCPROFV3_EXECUTABLE) is not None
        if rocprofv3_available is None
        else rocprofv3_available
    )

    # Pre-flight timing isolation audit
    logger.info("Running timing isolation pre-flight audit...")
    concurrent_processes = detect_concurrent_gpu_processes()
    if concurrent_processes:
        logger.warning(
            "Detected %d concurrent GPU process(es). This may introduce timing variability: %s",
            len(concurrent_processes),
            concurrent_processes,
        )
    clock_state_verified = verify_clock_state_with_warning(context="batch_start")
    if not clock_state_verified:
        logger.warning("Clock state verification failed at batch start")

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
                    temp_dir=temp_dir,
                    tool_version=tool_version,
                    gpu_architecture=gpu_architecture,
                    clock_locked=clock_locked,
                    rocprofv3_available=available,
                    runner=runner,
                    marked_blocked_ids=marked_blocked_ids,
                    global_start_index=global_start_idx,
                    concurrent_gpu_processes=concurrent_processes or None,
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
    temp_dir: Path | None,
    tool_version: str,
    gpu_architecture: str,
    clock_locked: bool,
    rocprofv3_available: bool,
    runner: ProfilerRunner | None,
    concurrent_gpu_processes: list[dict[str, Any]] | None = None,
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
    try:
        definition_payload = _load_json(problem_dir / "definition.json")
        workloads = _load_workloads(
            problem_dir / "workload.jsonl",
            limit=workload_limit,
            offset=workload_offset,
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
        command = tuple(str(part) for part in packager.execute())
        batch_runner = runner or _staging_runner(staging_dir, timeout=timeout)
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
                trial_count=1,
                clock_locked=clock_locked,
            ),
            rocprofv3_available=rocprofv3_available,
            runner=batch_runner,
        )
    except subprocess.TimeoutExpired as exc:
        return _write_blocked_sidecar(
            target,
            replacement_path=replacement_path,
            staging_dir=staging_dir,
            reason=f"rocprofv3 command timed out after {timeout} seconds",
            workload_offset=workload_offset,
            workload_limit=workload_limit,
            stdout=_subprocess_text(exc.stdout),
            stderr=_subprocess_text(exc.stderr),
            concurrent_gpu_processes=concurrent_gpu_processes,
        )
    except (OSError, ValueError) as exc:
        return _write_blocked_sidecar(
            target,
            replacement_path=replacement_path,
            staging_dir=staging_dir,
            reason=str(exc),
            workload_offset=workload_offset,
            workload_limit=workload_limit,
            concurrent_gpu_processes=concurrent_gpu_processes,
        )

    workload_slice_applied = workload_offset != 0 or (
        workload_limit is not None and len(workloads) < target.workload_count
    )
    trace_status_counts = _trace_status_counts(result.stdout)
    all_workloads_passed = (
        not workload_slice_applied
        and len(workloads) >= target.workload_count
        and trace_status_counts.get("PASSED", 0) >= target.workload_count
        and sum(trace_status_counts.values()) >= target.workload_count
    )
    payload = result.to_dict()
    if not isinstance(payload.get("evidence"), dict):
        payload["evidence"] = {"backend": "rocprofv3", "parsed_rows": []}
    replacement_failure_reason = _replacement_failure_reason(
        profiler_collected=result.profiler_collected,
        all_workloads_passed=all_workloads_passed,
        selection_reason=result.selection.reason,
    )
    status = _replacement_status(
        profiler_collected=result.profiler_collected,
        full_workload_coverage=all_workloads_passed,
        kernel_activity_rows=_kernel_activity_rows(payload.get("evidence") or {}),
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
    }
    if result.profiler_collected or status == "profiler_blocked":
        if concurrent_gpu_processes:
            payload["concurrent_gpu_processes"] = concurrent_gpu_processes
        replacement_path.parent.mkdir(parents=True, exist_ok=True)
        replacement_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {
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
        "fallback_reason": replacement_failure_reason,
        "returncode": result.returncode,
    }


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
    temp_dir: Path | None,
    tool_version: str,
    gpu_architecture: str,
    clock_locked: bool,
    rocprofv3_available: bool,
    runner: ProfilerRunner | None,
    concurrent_gpu_processes: list[dict[str, Any]] | None = None,
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
            temp_dir=temp_dir,
            tool_version=tool_version,
            gpu_architecture=gpu_architecture,
            clock_locked=clock_locked,
            rocprofv3_available=rocprofv3_available,
            runner=runner,
            concurrent_gpu_processes=concurrent_gpu_processes,
        )
        results.append(result)
        sidecar = _load_optional_json(Path(result["replacement_path"]))
        entry = _workload_manifest_entry(ref, result=result, sidecar=sidecar)
        _upsert_workload_manifest_entry(manifest, entry)
        _write_workload_manifest(manifest_path, manifest)

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
) -> dict[str, Any]:
    workload_slice_applied = workload_offset != 0 or workload_limit is not None
    payload = {
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
        },
    }
    if concurrent_gpu_processes:
        payload["concurrent_gpu_processes"] = concurrent_gpu_processes
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
    for timing_dir in timing_dirs:
        path = _replacement_timing_path(
            timing_dir,
            target.category,
            target.problem_path,
        )
        payload = _load_optional_json(path)
        if payload is None:
            continue
        metadata = (
            payload.get("replacement_metadata")
            if isinstance(payload.get("replacement_metadata"), dict)
            else {}
        )
        if metadata.get("workload_slice_applied") is not True:
            continue
        offset = _int_value(metadata.get("workload_offset"))
        if offset not in sidecars:
            sidecars[offset] = path
    return sidecars


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
    return (
        isinstance(entry, dict)
        and entry.get("status") == "profiler_backed"
        and entry.get("profiler_collected") is True
        and _int_value(entry.get("kernel_activity_rows")) > 0
        and _trace_status_counts_from_mapping(entry.get("trace_status_counts")).get(
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
    evidence = (
        sidecar.get("evidence") if isinstance(sidecar.get("evidence"), dict) else {}
    )
    metadata = (
        sidecar.get("replacement_metadata")
        if isinstance(sidecar.get("replacement_metadata"), dict)
        else {}
    )
    trace_counts = _trace_status_counts_from_mapping(
        metadata.get("trace_status_counts")
    )
    kernel_rows = _kernel_activity_rows(evidence)
    profiler_collected = sidecar.get("profiler_collected") is True
    passed = trace_counts.get("PASSED", 0) >= 1
    status = (
        "profiler_backed"
        if profiler_collected
        and evidence.get("backend") == "rocprofv3"
        and kernel_rows > 0
        and passed
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
        "backend": evidence.get("backend"),
        "trace_status_counts": trace_counts,
        "kernel_activity_rows": kernel_rows,
        "kernel_duration_ms": _float_value(evidence.get("kernel_duration_ms")),
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
    rows = _load_sidecar_rows(entry.get("replacement_path"))
    if rows:
        for row in rows:
            row["workload_index"] = entry.get("workload_index")
            row["workload_uuid"] = entry.get("workload_uuid")
        return rows
    count = _int_value(entry.get("kernel_activity_rows"))
    duration = _float_value(entry.get("kernel_duration_ms"))
    if count <= 0 or duration <= 0:
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


def _load_sidecar_rows(path_value: object) -> list[dict[str, Any]]:
    if not isinstance(path_value, str):
        return []
    payload = _load_optional_json(Path(path_value))
    evidence = (
        payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
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
    return metadata.get("replacement_status") in {
        "profiler_backed",
        "partial_profiler_backed",
        "profiler_blocked",
    }


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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _load_workloads(
    path: Path, *, limit: int | None, offset: int = 0
) -> list[Workload]:
    workloads: list[Workload] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        if index < offset:
            continue
        workloads.append(Workload(**json.loads(line)))
        if limit is not None and len(workloads) >= limit:
            break
    if not workloads:
        raise ValueError(f"workload file has no records: {path}")
    return workloads


def _staging_runner(staging_dir: Path, *, timeout: int) -> ProfilerRunner:
    def run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "PYTHONPATH": _pythonpath_with_src(),
            "SOL_EXECBENCH_GRACEFUL_EXIT": "1",
        }
        return subprocess.run(
            list(command),
            cwd=staging_dir,
            env=env,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

    return run


def _pythonpath_with_src() -> str:
    src = str(Path(__file__).resolve().parents[1] / "src")
    existing = os.environ.get("PYTHONPATH")
    return src if not existing else f"{src}{os.pathsep}{existing}"


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


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
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: object) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
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
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--temp-dir", type=Path, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timing-tool-version", default=DEFAULT_TOOL_VERSION)
    parser.add_argument("--gpu-architecture", default=DEFAULT_GPU_ARCHITECTURE)
    parser.add_argument(
        "--clock-locked",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    try:
        with acquire_pid_lock(args.output_dir):
            return run_batch(
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                source_timing_dirs=source_timing_dirs,
                replacement_timing_dir=args.replacement_timing_dir,
                limit=args.limit,
                only_problem=tuple(args.only_problem),
                skip_problem=skip_problem,
                mark_blocked_problem=tuple(args.mark_blocked_problem),
                mark_blocked_only=args.mark_blocked_only,
                workload_limit=args.workload_limit,
                workload_offset=args.workload_offset,
                workload_sharded=args.workload_sharded,
                workload_slice_timing_dirs=tuple(args.workload_slice_timing_dir),
                workload_sharded_import_only=args.workload_sharded_import_only,
                timeout=args.timeout,
                temp_dir=args.temp_dir,
                resume=args.resume,
                tool_version=args.timing_tool_version,
                gpu_architecture=args.gpu_architecture,
                clock_locked=args.clock_locked,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
