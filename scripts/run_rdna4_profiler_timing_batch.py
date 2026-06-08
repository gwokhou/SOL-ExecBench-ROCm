# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Replace RDNA4 fallback timing sidecars with profiler-backed rocprofv3 timing."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
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

DEFAULT_DATASET_ROOT = Path("data/SOL-ExecBench/benchmark")
DEFAULT_SOURCE_TIMING_DIR = Path("out/rdna4-timing-evidence/timing")
DEFAULT_OUTPUT_DIR = Path("out/rdna4-profiler-backed-timing")
DEFAULT_EXPECTED_PROBLEM_DENOMINATOR = 235
DEFAULT_TOOL_VERSION = "rocprofv3"
DEFAULT_GPU_ARCHITECTURE = "gfx1200"
CLAIM_BOUNDARY = (
    "Profiler-backed RDNA4 timing replacement evidence only; not score "
    "authority, paper parity, leaderboard result, or broader hardware "
    "validation."
)


def run_batch(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    source_timing_dirs: Sequence[Path] = (DEFAULT_SOURCE_TIMING_DIR,),
    replacement_timing_dir: Path | None = None,
    limit: int | None = None,
    only_problem: Sequence[str] = (),
    workload_limit: int | None = None,
    timeout: int = 900,
    resume: bool = True,
    tool_version: str = DEFAULT_TOOL_VERSION,
    gpu_architecture: str = DEFAULT_GPU_ARCHITECTURE,
    clock_locked: bool = True,
    rocprofv3_available: bool | None = None,
    runner: ProfilerRunner | None = None,
) -> int:
    """Run fallback replacement batch and return a process-style status code."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    if workload_limit is not None and workload_limit <= 0:
        raise ValueError("workload_limit must be positive when provided")

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
        resume=resume,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    available = (
        shutil.which(ROCPROFV3_EXECUTABLE) is not None
        if rocprofv3_available is None
        else rocprofv3_available
    )

    results: list[dict[str, Any]] = []
    for target in targets:
        results.append(
            _profile_target(
                target,
                dataset_root=dataset_root,
                replacement_timing_dir=replacement_root,
                output_dir=output_dir,
                workload_limit=workload_limit,
                timeout=timeout,
                tool_version=tool_version,
                gpu_architecture=gpu_architecture,
                clock_locked=clock_locked,
                rocprofv3_available=available,
                runner=runner,
            )
        )

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
    resume: bool = True,
) -> list[ProfilerTimingProblemCoverage]:
    """Select coverage rows whose fallback timing should be replaced."""
    only = set(only_problem)
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
    timeout: int,
    tool_version: str,
    gpu_architecture: str,
    clock_locked: bool,
    rocprofv3_available: bool,
    runner: ProfilerRunner | None,
) -> dict[str, Any]:
    problem_dir = dataset_root / target.problem_path
    staging_dir = Path(tempfile.mkdtemp(prefix="sol_execbench_rdna4_prof_batch_"))
    profile_dir = (
        output_dir / "rocprofv3" / target.category / Path(target.problem_path).name
    )
    output_file = Path(target.problem_path).name
    replacement_path = _replacement_timing_path(
        replacement_timing_dir,
        target.category,
        target.problem_path,
    )
    try:
        definition_payload = _load_json(problem_dir / "definition.json")
        workloads = _load_workloads(problem_dir / "workload.jsonl", workload_limit)
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
        return _target_failure(
            target,
            replacement_path=replacement_path,
            staging_dir=staging_dir,
            reason=f"rocprofv3 command timed out after {timeout} seconds",
            stdout=_subprocess_text(exc.stdout),
            stderr=_subprocess_text(exc.stderr),
        )
    except (OSError, ValueError) as exc:
        return _target_failure(
            target,
            replacement_path=replacement_path,
            staging_dir=staging_dir,
            reason=str(exc),
        )

    trace_status_counts = _trace_status_counts(result.stdout)
    all_workloads_passed = (
        len(workloads) >= target.workload_count
        and trace_status_counts.get("PASSED", 0) >= target.workload_count
        and sum(trace_status_counts.values()) >= target.workload_count
    )
    payload = result.to_dict()
    replacement_failure_reason = _replacement_failure_reason(
        profiler_collected=result.profiler_collected,
        all_workloads_passed=all_workloads_passed,
        selection_reason=result.selection.reason,
    )
    payload["replacement_metadata"] = {
        "schema_version": "sol_execbench.rdna4_profiler_timing_replacement.v1",
        "problem_id": target.problem_id,
        "replacement_status": _replacement_status(
            profiler_collected=result.profiler_collected,
            full_workload_coverage=all_workloads_passed,
            kernel_activity_rows=_kernel_activity_rows(payload.get("evidence") or {}),
        ),
        "profiled_workload_count": len(workloads),
        "expected_workload_count": target.workload_count,
        "trace_status_counts": trace_status_counts,
        "all_workloads_passed": all_workloads_passed,
        "workload_limit_applied": workload_limit,
        "full_workload_coverage": all_workloads_passed,
        "failure_reason": replacement_failure_reason,
    }
    if result.profiler_collected:
        replacement_path.parent.mkdir(parents=True, exist_ok=True)
        replacement_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    status = _replacement_status(
        profiler_collected=result.profiler_collected,
        full_workload_coverage=all_workloads_passed,
        kernel_activity_rows=_kernel_activity_rows(payload.get("evidence") or {}),
    )
    return {
        "problem_id": target.problem_id,
        "status": status,
        "replacement_path": str(replacement_path),
        "staging_dir": str(staging_dir),
        "profiler_collected": result.profiler_collected,
        "full_workload_coverage": all_workloads_passed,
        "trace_status_counts": trace_status_counts,
        "csv_path": str(result.csv_path) if result.csv_path is not None else None,
        "fallback_reason": replacement_failure_reason,
        "returncode": result.returncode,
    }


def _target_failure(
    target: ProfilerTimingProblemCoverage,
    *,
    replacement_path: Path,
    staging_dir: Path,
    reason: str,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    return {
        "problem_id": target.problem_id,
        "status": "profiler_blocked",
        "replacement_path": str(replacement_path),
        "staging_dir": str(staging_dir),
        "profiler_collected": False,
        "csv_path": None,
        "fallback_reason": reason,
        "stdout_tail": stdout[-4096:],
        "stderr_tail": stderr[-4096:],
        "returncode": None,
    }


def _replacement_timing_path(
    replacement_timing_dir: Path,
    category: str,
    problem_path: str,
) -> Path:
    return replacement_timing_dir / category / f"{Path(problem_path).name}.timing.json"


def _is_classified_replacement_sidecar(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or payload.get("profiler_collected") is not True:
        return False
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("backend") != "rocprofv3":
        return False
    metadata = payload.get("replacement_metadata")
    if not isinstance(metadata, dict):
        return _kernel_activity_rows(evidence) > 0
    if metadata.get("workload_limit_applied") is not None:
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


def _load_workloads(path: Path, limit: int | None) -> list[Workload]:
    workloads: list[Workload] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
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
    parser.add_argument("--workload-limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timing-tool-version", default=DEFAULT_TOOL_VERSION)
    parser.add_argument("--gpu-architecture", default=DEFAULT_GPU_ARCHITECTURE)
    parser.add_argument(
        "--clock-locked",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_timing_dirs = (
        tuple(args.source_timing_dir)
        if args.source_timing_dir
        else (DEFAULT_SOURCE_TIMING_DIR,)
    )
    try:
        return run_batch(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            source_timing_dirs=source_timing_dirs,
            replacement_timing_dir=args.replacement_timing_dir,
            limit=args.limit,
            only_problem=tuple(args.only_problem),
            workload_limit=args.workload_limit,
            timeout=args.timeout,
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
