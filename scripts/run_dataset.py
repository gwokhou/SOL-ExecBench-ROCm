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

"""Run SOL ExecBench problems using their reference implementation as the solution.

The positional ``problems_dir`` argument is auto-detected:
  - If it contains ``definition.json`` + ``workload.jsonl`` it is treated as a
    single problem directory (optionally with ``solution.py``).
  - Otherwise it is treated as a dataset root with category sub-directories
    (e.g. L1/, L2/).

Usage:
    # Single problem
    uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark/L1/my_problem [-o ./results]

    # Dataset with categories
    uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark [--category L1 L2] [--limit 5] [-o ./results]

    # Use a custom solution filename
    uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --solution-name solution.json
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.dataset.evidence_refs import (
    relative_ref as _relative_ref,
    safe_sidecar_stem as _safe_sidecar_stem,
)
from sol_execbench.core.dataset.run_closure import (
    closure_record as _build_closure_record,
    closure_totals as _build_closure_totals,
    dataset_reuse_decision as _build_dataset_reuse_decision,
    derived_evidence_for_workload as _build_derived_evidence_for_workload,
    prior_closure_provenance as _load_prior_closure_provenance,
    selected_workload_closure_record as _build_selected_workload_closure_record,
    stale_provenance_mismatch as _build_stale_provenance_mismatch,
    utc_timestamp as _build_utc_timestamp,
    write_execution_closure as _write_execution_closure_report,
)
from sol_execbench.core.dataset.runner import (
    CLI_LOG_LIMIT as _CLI_LOG_LIMIT,
    _cli_failure_notes,
    _save_cli_log,
    _save_cli_timeout_log,
    _extend_derived_reports_for_problem,
    build_cli_command,
    build_custom_solution,
    build_reference_solution,
    build_solution_for_problem,
    build_amd_score_reports_for_problem,
    collect_timing_evidence_for_problem,
    inspect_traces,
    print_summary,
    run_cli as _runner_run_cli,
    write_amd_score_report,
    write_summary_report,
)
from sol_execbench.core.dataset.run_state import (
    closure_status_for_trace as _run_state_closure_status_for_trace,
    closure_status_with_evidence as _run_state_closure_status_with_evidence,
    discover_problems as _discover_problems,
    read_workload_rows as _read_run_workload_rows,
    readiness_workload_map as _build_readiness_workload_map,
    ready_problem_map as _build_ready_problem_map,
    requested_evidence_requirements as _build_requested_evidence_requirements,
    selected_workload_rows as _select_run_workload_rows,
    trace_map as _build_trace_map,
    trace_status as _run_state_trace_status,
    workload_key as _run_workload_key,
)
from sol_execbench.core.scoring.amd_score import (
    AmdNativeScore,
)
from sol_execbench.core.scoring.baseline_artifact import (
    load_scoring_baseline_artifact,
)

ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = {"L1", "L2", "FlashInfer-Bench", "Quant"}

_SCRIPT_COMPAT_EXPORTS = (
    _CLI_LOG_LIMIT,
    _safe_sidecar_stem,
    _save_cli_log,
    _save_cli_timeout_log,
    build_cli_command,
    build_custom_solution,
    build_reference_solution,
    build_amd_score_reports_for_problem,
)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def discover_problems(
    benchmark_dir: Path, categories: list[str] | None = None
) -> list[Path]:
    """Return a sorted list of problem directories under *benchmark_dir*.

    Each problem directory must contain definition.json and workload.jsonl.
    If *categories* is given (e.g. ["L1", "L2"]), only those sub-directories are searched.
    """
    return _discover_problems(
        benchmark_dir,
        categories,
        known_categories=CATEGORIES,
    )


def run_cli(
    definition_path: Path,
    workload_path: Path,
    solution_path: Path,
    output_dir: Path,
    job_name: str,
    timeout: int,
    config_path: Path | None = None,
    keep_staging: bool = False,
    verbose: bool = False,
) -> list[dict] | None:
    """Compatibility wrapper for callers importing this script module."""
    return _runner_run_cli(
        definition_path=definition_path,
        workload_path=workload_path,
        solution_path=solution_path,
        output_dir=output_dir,
        job_name=job_name,
        timeout=timeout,
        config_path=config_path,
        keep_staging=keep_staging,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Ready-subset execution closure
# ---------------------------------------------------------------------------


def _utc_timestamp() -> str:
    return _build_utc_timestamp()


def _load_json_sidecar(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _problem_id_for(benchmark_root: Path, problem_dir: Path) -> str:
    try:
        return problem_dir.relative_to(benchmark_root).as_posix()
    except ValueError:
        return f"{problem_dir.parent.name}/{problem_dir.name}"


def _workload_key(uuid: str | None, row_index: int | None) -> tuple[str, str | int]:
    return _run_workload_key(uuid, row_index)


def _read_workload_rows(workload_path: Path) -> list[tuple[int, dict, str]]:
    return _read_run_workload_rows(workload_path)


def _ready_problem_map(ready_subset: dict | None) -> dict[str, dict]:
    return _build_ready_problem_map(ready_subset)


def _readiness_workload_map(readiness: dict | None) -> dict[tuple[str, tuple[str, str | int]], dict]:
    return _build_readiness_workload_map(readiness)


def _manifest_checksum(manifest: dict | None) -> str | None:
    if manifest is None:
        return None
    checksum = manifest.get("manifest_checksum")
    if isinstance(checksum, dict):
        return checksum.get("value")
    if isinstance(checksum, str):
        return checksum
    return None


def _sidecar_checksum(payload: dict | None, key: str) -> str | None:
    if payload is None:
        return None
    value = payload.get(key)
    if isinstance(value, dict):
        return value.get("value")
    if isinstance(value, str):
        return value
    return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _first_relative_ref(path: Path, *bases: Path) -> str:
    for base in bases:
        ref = _relative_ref(path, base)
        if ref != path.resolve().name:
            return ref
    return path.resolve().name


def _normalized_command_args(args: argparse.Namespace) -> list[str]:
    normalized = [args.problems_dir.name]
    if args.category:
        normalized.extend(["--category", ",".join(args.category)])
    if args.limit is not None:
        normalized.extend(["--limit", str(args.limit)])
    if args.max_workloads is not None:
        normalized.extend(["--max-workloads", str(args.max_workloads)])
    if args.timeout != 120:
        normalized.extend(["--timeout", str(args.timeout)])
    if args.solution_name:
        normalized.extend(["--solution-name", args.solution_name])
    if args.rerun:
        normalized.append("--rerun")
    if args.keep_staging:
        normalized.append("--keep-staging")
    if args.verbose:
        normalized.append("--verbose")
    if args.lock_clocks:
        normalized.append("--lock-clocks")
    for flag, value in (
        ("--output", args.output),
        ("--ready-subset", args.ready_subset),
        ("--readiness", args.readiness),
        ("--execution-closure", args.execution_closure),
        ("--dataset-manifest", args.dataset_manifest),
        ("--amd-score-report", args.amd_score_report),
        ("--amd-sol-bound-dir", args.amd_sol_bound_dir),
        ("--solar-derivation", args.solar_derivation),
        ("--timing-evidence-dir", args.timing_evidence_dir),
        ("--scoring-baseline", args.scoring_baseline),
    ):
        if value is not None:
            normalized.extend([flag, Path(value).name])
    return normalized


def _closure_record(
    *,
    category: str,
    problem_id: str,
    problem_path: str,
    workload_uuid: str | None,
    row_index: int,
    closure_status: str,
    readiness: dict | None = None,
    filter_reasons: list[str] | None = None,
    trace_ref: str | None = None,
    summary_ref: str | None = None,
    cli_log_ref: str | None = None,
    solution_ref: str | None = None,
    evidence_refs: dict[str, str] | None = None,
    evidence_gaps: list[str] | None = None,
    trace_status: str | None = None,
    notes: list[str] | None = None,
) -> dict:
    return _build_closure_record(
        category=category,
        problem_id=problem_id,
        problem_path=problem_path,
        workload_uuid=workload_uuid,
        row_index=row_index,
        closure_status=closure_status,
        readiness=readiness,
        filter_reasons=filter_reasons or [],
        trace_status=trace_status,
        trace_ref=trace_ref,
        summary_ref=summary_ref,
        cli_log_ref=cli_log_ref,
        solution_ref=solution_ref,
        evidence_refs=evidence_refs or {},
        evidence_gaps=evidence_gaps or [],
        notes=notes,
    )


def _selected_workload_rows(
    workload_path: Path,
    workload_refs: list[dict],
    *,
    max_workloads: int | None,
) -> tuple[list[str], list[dict], list[dict], list[dict]]:
    return _select_run_workload_rows(
        workload_path,
        workload_refs,
        max_workloads=max_workloads,
    )


def _trace_map(traces: list[dict]) -> dict[tuple[str, str | int], dict]:
    return _build_trace_map(traces)


def _trace_status(trace: dict | None) -> str | None:
    return _run_state_trace_status(trace)


def _closure_status_for_trace(trace: dict | None, *, skipped: bool = False) -> str:
    return _run_state_closure_status_for_trace(trace, skipped=skipped)


def _derived_evidence_for_workload(
    *,
    definition_name: str,
    workload_uuid: str | None,
    problem_output_dir: Path,
    output_dir: Path,
    amd_score_report: Path | None,
    sol_bound_artifact_dir: Path | None,
    solar_derivation_dir: Path | None,
    timing_evidence_dir: Path | None,
    category: str,
) -> tuple[dict[str, str], list[str]]:
    return _build_derived_evidence_for_workload(
        definition_name=definition_name,
        workload_uuid=workload_uuid,
        problem_output_dir=problem_output_dir,
        output_dir=output_dir,
        amd_score_report=amd_score_report,
        sol_bound_artifact_dir=sol_bound_artifact_dir,
        solar_derivation_dir=solar_derivation_dir,
        timing_evidence_dir=timing_evidence_dir,
        category=category,
    )


def _closure_status_with_evidence(
    status: str,
    evidence_gaps: list[str],
) -> str:
    return _run_state_closure_status_with_evidence(status, evidence_gaps)


def _requested_evidence_requirements(args: argparse.Namespace) -> list[str]:
    return _build_requested_evidence_requirements(
        amd_score_report=args.amd_score_report,
        amd_sol_bound_dir=args.amd_sol_bound_dir,
        solar_derivation=args.solar_derivation,
        timing_evidence_dir=args.timing_evidence_dir,
    )


def _stale_provenance_mismatch(
    *,
    observed: str | None,
) -> dict[str, object]:
    return _build_stale_provenance_mismatch(observed=observed)


def _prior_closure_provenance(path: Path) -> tuple[dict | None, dict[str, object] | None]:
    return _load_prior_closure_provenance(path)


def _dataset_reuse_decision(
    *,
    rerun: bool,
    traces_path: Path,
    failed_count: int,
    execution_closure_path: Path | None,
    provenance: dict,
):
    return _build_dataset_reuse_decision(
        rerun=rerun,
        traces_path=traces_path,
        failed_count=failed_count,
        execution_closure_path=execution_closure_path,
        provenance=provenance,
    )


def _closure_totals(records: list[dict]) -> dict[str, int]:
    return _build_closure_totals(records)


def _selected_workload_closure_record(
    *,
    category: str,
    problem_id: str,
    problem_path: str,
    workload_uuid: str | None,
    row_index: int,
    readiness: dict | None,
    trace: dict | None,
    skipped: bool,
    traces_path: Path,
    summary_ref: str,
    solution_path: Path | None,
    output_dir: Path,
    definition_name: str,
    problem_output_dir: Path,
    amd_score_report: Path | None,
    sol_bound_artifact_dir: Path | None,
    solar_derivation_dir: Path | None,
    timing_evidence_dir: Path | None,
) -> dict:
    return _build_selected_workload_closure_record(
        category=category,
        problem_id=problem_id,
        problem_path=problem_path,
        workload_uuid=workload_uuid,
        row_index=row_index,
        readiness=readiness,
        trace=trace,
        skipped=skipped,
        traces_path=traces_path,
        summary_ref=summary_ref,
        solution_path=solution_path,
        output_dir=output_dir,
        definition_name=definition_name,
        problem_output_dir=problem_output_dir,
        amd_score_report=amd_score_report,
        sol_bound_artifact_dir=sol_bound_artifact_dir,
        solar_derivation_dir=solar_derivation_dir,
        timing_evidence_dir=timing_evidence_dir,
    )


def _write_execution_closure(
    *,
    path: Path,
    records: list[dict],
    provenance: dict,
    filters: dict,
    provenance_mismatches: list[dict] | None = None,
) -> None:
    _write_execution_closure_report(
        path=path,
        records=records,
        provenance=provenance,
        filters=filters,
        provenance_mismatches=provenance_mismatches,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Run SOL ExecBench problems using reference implementations.",
    )
    ap.add_argument(
        "problems_dir",
        type=Path,
        help="Path to a single problem directory (contains definition.json + workload.jsonl) "
        "or a dataset root with category sub-directories (e.g. L1/, L2/).",
    )
    ap.add_argument(
        "--category",
        type=str,
        nargs="+",
        default=None,
        choices=sorted(CATEGORIES),
        help="Restrict to one or more categories (e.g. --category L1 L2).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of problems to evaluate.",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "out",
        help="Output directory for traces and summary. Defaults to <repo_root>/out.",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-problem GPU evaluation timeout in seconds (default: 300).",
    )
    ap.add_argument(
        "--max-workloads",
        type=int,
        default=None,
        help="Max number of workloads per problem. Truncates the workload file if exceeded.",
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of timing iterations per workload (default: 50, from BenchmarkConfig).",
    )
    ap.add_argument(
        "--warmup-runs",
        type=int,
        default=None,
        help="Number of warmup executions before timing (default: 10, from BenchmarkConfig).",
    )
    ap.add_argument(
        "--lock-clocks",
        action="store_true",
        help="Require locked GPU clocks for benchmark and timing evidence.",
    )
    ap.add_argument(
        "--solution-name",
        type=str,
        default=None,
        help="Filename to look for in each problem directory as the solution "
        "(e.g. solution.py, solution.json). "
        ".py files are wrapped into a solution JSON automatically; "
        ".json files are loaded directly. "
        "Defaults to None (uses definition.reference).",
    )
    ap.add_argument(
        "--rerun",
        action="store_true",
        help="Re-evaluate problems that already have results. By default, existing results are skipped.",
    )
    ap.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep CLI staging directories after execution (useful for debugging).",
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Pass --verbose to the sol-execbench CLI.",
    )
    ap.add_argument(
        "--amd-score-report",
        type=Path,
        default=None,
        help="Optional path for a derived AMD-native suite score JSON report.",
    )
    ap.add_argument(
        "--scoring-baseline",
        type=Path,
        default=None,
        help=(
            "Optional release-defined scoring baseline artifact JSON for "
            "--amd-score-report. Without it, reference latency is used as a "
            "provisional fallback."
        ),
    )
    ap.add_argument(
        "--amd-sol-bound-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for derived AMD SOL bound v2 sidecars when "
            "--amd-score-report is enabled."
        ),
    )
    ap.add_argument(
        "--solar-derivation",
        type=Path,
        default=None,
        help=(
            "Optional directory for generated SOLAR derivation sidecars. "
            "Sidecars are built from definition and workload inputs only."
        ),
    )
    ap.add_argument(
        "--timing-evidence-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for per-problem source-specific ROCm timing "
            "evidence JSON. Profiler-backed policies invoke rocprofv3."
        ),
    )
    ap.add_argument(
        "--timing-tool-version",
        type=str,
        default="rocprofv3",
        help="Tool version string recorded in timing evidence.",
    )
    ap.add_argument(
        "--gpu-architecture",
        type=str,
        default="unknown",
        help="GPU architecture string recorded in timing evidence, e.g. gfx942.",
    )
    ap.add_argument(
        "--ready-subset",
        type=Path,
        default=None,
        help="Phase 54 ready_subset.json used to bound dataset execution.",
    )
    ap.add_argument(
        "--readiness",
        type=Path,
        default=None,
        help="Optional Phase 54 readiness.json used to enrich closure blockers.",
    )
    ap.add_argument(
        "--execution-closure",
        type=Path,
        default=None,
        help=(
            "Path for execution_closure.json. Defaults to "
            "<output>/execution_closure.json when --ready-subset is supplied."
        ),
    )
    ap.add_argument(
        "--dataset-manifest",
        type=Path,
        default=None,
        help="Optional dataset manifest JSON used for closure provenance.",
    )
    args = ap.parse_args()

    problems_dir = args.problems_dir.resolve()
    if not problems_dir.is_dir():
        print(f"Error: {problems_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_closure_path = (
        args.execution_closure.resolve()
        if args.execution_closure is not None
        else output_dir / "execution_closure.json"
        if args.ready_subset is not None
        else None
    )

    ready_subset = (
        _load_json_sidecar(args.ready_subset.resolve())
        if args.ready_subset is not None
        else None
    )
    readiness = (
        _load_json_sidecar(args.readiness.resolve())
        if args.readiness is not None
        else None
    )
    dataset_manifest = (
        _load_json_sidecar(args.dataset_manifest.resolve())
        if args.dataset_manifest is not None
        else None
    )
    ready_problems = _ready_problem_map(ready_subset)
    readiness_by_workload = _readiness_workload_map(readiness)
    closure_records: list[dict] = []
    provenance_mismatches: list[dict] = []

    # Auto-detect: single problem dir vs. dataset root
    is_single_problem = (problems_dir / "definition.json").exists() and (
        problems_dir / "workload.jsonl"
    ).exists()

    if is_single_problem:
        problems = [problems_dir]
        print(f"Single problem: {problems_dir.name}")
    else:
        problems = discover_problems(problems_dir, args.category)
        if ready_subset is not None:
            discovered_by_id = {
                _problem_id_for(problems_dir, problem_dir): problem_dir
                for problem_dir in problems
            }
            category_filter = set(args.category or [])
            selected_ids: list[str] = []
            for problem_id, problem_ref in ready_problems.items():
                reason: list[str] = []
                if category_filter and problem_ref.get("category") not in category_filter:
                    reason.append("category_filter")
                elif problem_id not in discovered_by_id:
                    reason.append("problem_not_discovered")
                else:
                    selected_ids.append(problem_id)
                    continue
                for workload_ref in problem_ref.get("workloads", []):
                    readiness_record = readiness_by_workload.get(
                        (problem_id, _workload_key(workload_ref.get("uuid"), workload_ref.get("row_index")))
                    )
                    closure_records.append(
                        _closure_record(
                            category=str(problem_ref.get("category")),
                            problem_id=problem_id,
                            problem_path=str(problem_ref.get("problem_path")),
                            workload_uuid=workload_ref.get("uuid"),
                            row_index=int(workload_ref.get("row_index", 0)),
                            closure_status="filtered",
                            readiness=readiness_record,
                            filter_reasons=reason,
                        )
                    )
            limited_ids = selected_ids
            if args.limit:
                limited_ids = selected_ids[: args.limit]
                filtered_ids = set(selected_ids[args.limit :])
                for problem_id in selected_ids[args.limit :]:
                    problem_ref = ready_problems[problem_id]
                    for workload_ref in problem_ref.get("workloads", []):
                        readiness_record = readiness_by_workload.get(
                            (problem_id, _workload_key(workload_ref.get("uuid"), workload_ref.get("row_index")))
                        )
                        closure_records.append(
                            _closure_record(
                                category=str(problem_ref.get("category")),
                                problem_id=problem_id,
                                problem_path=str(problem_ref.get("problem_path")),
                                workload_uuid=workload_ref.get("uuid"),
                                row_index=int(workload_ref.get("row_index", 0)),
                                closure_status="filtered",
                                readiness=readiness_record,
                                filter_reasons=["problem_limit"],
                            )
                        )
                problems = [
                    discovered_by_id[problem_id]
                    for problem_id in limited_ids
                    if problem_id not in filtered_ids
                ]
            else:
                problems = [discovered_by_id[problem_id] for problem_id in limited_ids]
        elif args.limit:
            problems = problems[: args.limit]

        print(f"Discovered {len(problems)} problems under {problems_dir}")
        if not problems and ready_subset is None:
            print("No problems found. Check the directory path.")
            sys.exit(1)

    benchmark_config = BenchmarkConfig(
        warmup_runs=(
            args.warmup_runs
            if args.warmup_runs is not None
            else BenchmarkConfig().warmup_runs
        ),
        iterations=(
            args.iterations
            if args.iterations is not None
            else BenchmarkConfig().iterations
        ),
        lock_clocks=args.lock_clocks,
    )

    # Build benchmark config if non-default timing settings were requested
    config_path = None
    if args.iterations is not None or args.warmup_runs is not None or args.lock_clocks:
        config_dict = {
            "warmup_runs": benchmark_config.warmup_runs,
            "iterations": benchmark_config.iterations,
            "lock_clocks": benchmark_config.lock_clocks,
        }
        config_path = output_dir / "config.json"
        config_path.write_text(json.dumps(config_dict, indent=2))
        print(f"Using benchmark config: {config_dict}")

    summaries = []
    amd_scores: list[AmdNativeScore] = []
    scoring_baseline = (
        load_scoring_baseline_artifact(args.scoring_baseline)
        if args.scoring_baseline is not None
        else None
    )
    provenance = {
        "command_args": _normalized_command_args(args),
        "dataset_root": _first_relative_ref(problems_dir, ROOT),
        "selected_categories": args.category,
        "limit": args.limit,
        "max_workloads": args.max_workloads,
        "timeout": args.timeout,
        "warmup_runs": benchmark_config.warmup_runs,
        "iterations": benchmark_config.iterations,
        "lock_clocks": benchmark_config.lock_clocks,
        "rerun": args.rerun,
        "keep_staging": args.keep_staging,
        "verbose": args.verbose,
        "solution_mode": "named" if args.solution_name else "reference",
        "solution_name": args.solution_name,
        "output_dir": _first_relative_ref(output_dir, ROOT),
        "summary_path": _relative_ref(output_dir / "summary.json", output_dir),
        "ready_subset_path": (
            _first_relative_ref(args.ready_subset, ROOT, problems_dir, output_dir)
            if args.ready_subset
            else None
        ),
        "ready_subset_checksum": _sidecar_checksum(ready_subset, "ready_subset_checksum"),
        "readiness_path": (
            _first_relative_ref(args.readiness, ROOT, problems_dir, output_dir)
            if args.readiness
            else None
        ),
        "readiness_checksum": (
            _sidecar_checksum(readiness, "readiness_checksum")
            or (ready_subset or {}).get("readiness_checksum")
        ),
        "dataset_manifest_path": (
            _first_relative_ref(args.dataset_manifest, ROOT, problems_dir, output_dir)
            if args.dataset_manifest
            else None
        ),
        "dataset_manifest_checksum": _manifest_checksum(dataset_manifest),
        "requested_evidence_requirements": _requested_evidence_requirements(args),
        "git_commit": _git_commit(),
        "config_path": _relative_ref(config_path, output_dir) if config_path else None,
        "benchmark_config": {
            "warmup_runs": benchmark_config.warmup_runs,
            "iterations": benchmark_config.iterations,
            "lock_clocks": benchmark_config.lock_clocks,
        },
        "derived_evidence": {
            "amd_score_report": (
                _first_relative_ref(args.amd_score_report, output_dir, ROOT)
                if args.amd_score_report
                else None
            ),
            "amd_sol_bound_dir": (
                _first_relative_ref(args.amd_sol_bound_dir, output_dir, ROOT)
                if args.amd_sol_bound_dir
                else None
            ),
            "solar_derivation": (
                _first_relative_ref(args.solar_derivation, output_dir, ROOT)
                if args.solar_derivation
                else None
            ),
            "timing_evidence_dir": (
                _first_relative_ref(args.timing_evidence_dir, output_dir, ROOT)
                if args.timing_evidence_dir
                else None
            ),
        },
    }

    if ready_subset is not None and not problems:
        _write_execution_closure(
            path=execution_closure_path or (output_dir / "execution_closure.json"),
            records=closure_records,
            provenance=provenance,
            filters={
                "ready_subset": True,
                "category": args.category,
                "limit": args.limit,
                "max_workloads": args.max_workloads,
            },
            provenance_mismatches=provenance_mismatches,
        )
        print(f"Execution closure saved to {execution_closure_path}")
        return

    attempted_ready_keys: set[tuple[str, tuple[str, str | int]]] = set()
    # ROCm execution stays serial here; runner helpers provide the future scheduling seam.
    for i, problem_dir in enumerate(problems):
        problem_name = problem_dir.name
        category = problem_dir.parent.name
        problem_id = _problem_id_for(problems_dir, problem_dir)
        problem_ref = ready_problems.get(problem_id)
        print(f"\n[{i + 1}/{len(problems)}] {category}/{problem_name}")

        definition_path = problem_dir / "definition.json"
        workload_path = problem_dir / "workload.jsonl"

        problem_output_dir = output_dir / category / problem_name
        traces_path = problem_output_dir / "traces.json"
        definition_payload = json.loads(definition_path.read_text())
        selected_workload_refs: list[dict] | None = None
        if problem_ref is not None:
            problem_output_dir.mkdir(parents=True, exist_ok=True)
            selected_lines, selected_workload_refs, cap_filtered, missing_refs = (
                _selected_workload_rows(
                    workload_path,
                    list(problem_ref.get("workloads", [])),
                    max_workloads=args.max_workloads,
                )
            )
            filtered_workload_path = problem_output_dir / "workload.jsonl"
            filtered_workload_path.write_text("\n".join(selected_lines) + ("\n" if selected_lines else ""))
            workload_path = filtered_workload_path
            for workload_ref in cap_filtered:
                key = _workload_key(workload_ref.get("uuid"), workload_ref.get("row_index"))
                readiness_record = readiness_by_workload.get((problem_id, key))
                closure_records.append(
                    _closure_record(
                        category=category,
                        problem_id=problem_id,
                        problem_path=str(problem_ref.get("problem_path")),
                        workload_uuid=workload_ref.get("uuid"),
                        row_index=int(workload_ref.get("row_index", 0)),
                        closure_status="filtered",
                        readiness=readiness_record,
                        filter_reasons=["max_workloads_cap"],
                    )
                )
            for workload_ref in missing_refs:
                key = _workload_key(workload_ref.get("uuid"), workload_ref.get("row_index"))
                readiness_record = readiness_by_workload.get((problem_id, key))
                closure_records.append(
                    _closure_record(
                        category=category,
                        problem_id=problem_id,
                        problem_path=str(problem_ref.get("problem_path")),
                        workload_uuid=workload_ref.get("uuid"),
                        row_index=int(workload_ref.get("row_index", 0)),
                        closure_status="filtered",
                        readiness=readiness_record,
                        filter_reasons=["workload_not_found"],
                    )
                )
            if not selected_workload_refs:
                continue

        # Preserve ordinary resume behavior unless closure provenance is part of the contract.
        if traces_path.exists():
            traces = json.loads(traces_path.read_text())
            summary = inspect_traces(traces, f"{category}/{problem_name}")
            reuse_decision = _dataset_reuse_decision(
                rerun=args.rerun,
                traces_path=traces_path,
                failed_count=int(summary["failed"]),
                execution_closure_path=execution_closure_path,
                provenance=provenance,
            )
            if reuse_decision.should_reuse:
                if (
                    args.amd_score_report is not None
                    or args.amd_sol_bound_dir is not None
                    or args.solar_derivation is not None
                ):
                    _extend_derived_reports_for_problem(
                        amd_scores=amd_scores,
                        definition_path=definition_path,
                        workload_path=workload_path,
                        traces_path=traces_path,
                        traces_payload=traces,
                        output_dir=output_dir,
                        baseline_artifact=scoring_baseline,
                        sol_bound_artifact_dir=args.amd_sol_bound_dir,
                        solar_derivation_dir=args.solar_derivation,
                    )
                print("  Skipping (already passed). Use --rerun to re-evaluate.")
                summaries.append(summary)
                if selected_workload_refs is not None:
                    trace_by_key = _trace_map(traces)
                    for workload_ref in selected_workload_refs:
                        key = _workload_key(
                            workload_ref.get("uuid"), workload_ref.get("row_index")
                        )
                        attempted_ready_keys.add((problem_id, key))
                        trace = trace_by_key.get(key)
                        closure_records.append(
                            _selected_workload_closure_record(
                                category=category,
                                problem_id=problem_id,
                                problem_path=str(problem_ref.get("problem_path")),
                                workload_uuid=workload_ref.get("uuid"),
                                row_index=int(workload_ref.get("row_index", 0)),
                                readiness=readiness_by_workload.get((problem_id, key)),
                                trace=trace,
                                skipped=True,
                                traces_path=traces_path,
                                summary_ref="summary.json",
                                solution_path=problem_output_dir / "solution.json",
                                output_dir=output_dir,
                                definition_name=str(definition_payload["name"]),
                                problem_output_dir=problem_output_dir,
                                amd_score_report=args.amd_score_report,
                                sol_bound_artifact_dir=args.amd_sol_bound_dir,
                                solar_derivation_dir=args.solar_derivation,
                                timing_evidence_dir=args.timing_evidence_dir,
                            )
                        )
                continue
            if reuse_decision.provenance_mismatches:
                provenance_mismatches.extend(
                    list(reuse_decision.provenance_mismatches)
                )
                print("  Re-running (previous pass has stale closure provenance).")
            elif reuse_decision.reason == "previous_failed":
                print(f"  Re-running (previous run had {summary['failed']} failures).")
            elif reuse_decision.reason == "rerun_requested":
                print("  Re-running (--rerun requested).")

        # Clear previous run output
        if problem_output_dir.exists():
            shutil.rmtree(problem_output_dir)
        problem_output_dir.mkdir(parents=True, exist_ok=True)
        if selected_workload_refs is not None:
            workload_path = problem_output_dir / "workload.jsonl"
            workload_path.write_text(
                "\n".join(selected_lines) + ("\n" if selected_lines else "")
            )

        # Truncate workloads if --max-workloads is set outside ready-subset mode.
        if args.max_workloads is not None and selected_workload_refs is None:
            lines = workload_path.read_text().splitlines()
            if len(lines) > args.max_workloads:
                truncated_path = problem_output_dir / "workload.jsonl"
                truncated_path.write_text("\n".join(lines[: args.max_workloads]))
                workload_path = truncated_path

        # Load definition to build the reference solution
        definition = definition_payload

        # Use named solution file if present, otherwise fall back to reference
        if args.solution_name:
            solution_file = problem_dir / args.solution_name
            if not solution_file.exists():
                print(f"  Skipping: {args.solution_name} not found")
                continue
        else:
            if "reference" not in definition or not definition["reference"].strip():
                print("  Skipping: no reference code")
                continue

        solution = build_solution_for_problem(
            definition, problem_dir, args.solution_name
        )

        solution_path = problem_output_dir / "solution.json"
        solution_path.write_text(json.dumps(solution, indent=2))

        # Call sol-execbench CLI
        job_name = f"ref_{problem_name[:40]}"
        traces = run_cli(
            definition_path=definition_path,
            workload_path=workload_path,
            solution_path=solution_path,
            output_dir=problem_output_dir,
            job_name=job_name,
            timeout=args.timeout,
            config_path=config_path,
            keep_staging=args.keep_staging,
            verbose=args.verbose,
        )

        if traces is None:
            print("  ERROR: CLI returned no traces")
            summaries.append(
                {
                    "problem": f"{category}/{problem_name}",
                    "total": 0,
                    "passed": 0,
                    "failed": 1,
                    "latencies_ms": [],
                    "failure_reasons": ["CLI returned no output"],
                }
            )
            if selected_workload_refs is not None:
                cli_log = problem_output_dir / f"{job_name}_cli.log"
                for workload_ref in selected_workload_refs:
                    key = _workload_key(
                        workload_ref.get("uuid"), workload_ref.get("row_index")
                    )
                    attempted_ready_keys.add((problem_id, key))
                    closure_records.append(
                        _closure_record(
                            category=category,
                            problem_id=problem_id,
                            problem_path=str(problem_ref.get("problem_path")),
                            workload_uuid=workload_ref.get("uuid"),
                            row_index=int(workload_ref.get("row_index", 0)),
                            closure_status="attempted_failed",
                            readiness=readiness_by_workload.get((problem_id, key)),
                            summary_ref="summary.json",
                            cli_log_ref=_relative_ref(cli_log, output_dir)
                            if cli_log.exists()
                            else None,
                            solution_ref=_relative_ref(solution_path, output_dir),
                            notes=_cli_failure_notes(cli_log),
                        )
                    )
            continue

        # Save raw traces
        traces_path = problem_output_dir / "traces.json"
        traces_path.write_text(json.dumps(traces, indent=2))

        if (
            args.amd_score_report is not None
            or args.amd_sol_bound_dir is not None
            or args.solar_derivation is not None
        ):
            _extend_derived_reports_for_problem(
                amd_scores=amd_scores,
                definition_path=definition_path,
                workload_path=workload_path,
                traces_path=traces_path,
                traces_payload=traces,
                output_dir=output_dir,
                baseline_artifact=scoring_baseline,
                sol_bound_artifact_dir=args.amd_sol_bound_dir,
                solar_derivation_dir=args.solar_derivation,
            )

        if args.timing_evidence_dir is not None:
            timing_root = args.timing_evidence_dir.resolve()
            collect_timing_evidence_for_problem(
                definition_path=definition_path,
                workload_path=workload_path,
                solution_path=solution_path,
                output_dir=problem_output_dir,
                timing_evidence_root=timing_root / category,
                job_name=job_name,
                solution=solution,
                benchmark_config=benchmark_config,
                timeout=args.timeout,
                config_path=config_path,
                keep_staging=args.keep_staging,
                verbose=args.verbose,
                tool_version=args.timing_tool_version,
                gpu_architecture=args.gpu_architecture,
            )
            print(f"  Timing evidence saved under {timing_root / category}")

        if selected_workload_refs is not None:
            trace_by_key = _trace_map(traces)
            for workload_ref in selected_workload_refs:
                key = _workload_key(workload_ref.get("uuid"), workload_ref.get("row_index"))
                attempted_ready_keys.add((problem_id, key))
                trace = trace_by_key.get(key)
                closure_records.append(
                    _selected_workload_closure_record(
                        category=category,
                        problem_id=problem_id,
                        problem_path=str(problem_ref.get("problem_path")),
                        workload_uuid=workload_ref.get("uuid"),
                        row_index=int(workload_ref.get("row_index", 0)),
                        readiness=readiness_by_workload.get((problem_id, key)),
                        trace=trace,
                        skipped=False,
                        traces_path=traces_path,
                        summary_ref="summary.json",
                        solution_path=solution_path,
                        output_dir=output_dir,
                        definition_name=str(definition_payload["name"]),
                        problem_output_dir=problem_output_dir,
                        amd_score_report=args.amd_score_report,
                        sol_bound_artifact_dir=args.amd_sol_bound_dir,
                        solar_derivation_dir=args.solar_derivation,
                        timing_evidence_dir=args.timing_evidence_dir,
                    )
                )

        # Inspect
        summary = inspect_traces(traces, f"{category}/{problem_name}")
        summaries.append(summary)

        status = "OK" if summary["failed"] == 0 else "FAIL"
        print(f"  {status}: {summary['passed']}/{summary['total']} passed")

        if summary["failure_reasons"]:
            for reason in summary["failure_reasons"][:3]:
                print(f"  {reason}")

    # Print overall summary
    print_summary(summaries)

    # Save summary JSON
    summary_path = write_summary_report(output_dir, summaries)
    print(f"\nSummary saved to {summary_path}")
    print(f"Per-problem traces saved under {output_dir}")

    if args.amd_score_report is not None:
        report_path = args.amd_score_report.resolve()
        write_amd_score_report(
            report_path,
            amd_scores,
            problem_count=len(summaries),
            baseline_entry_count=len(scoring_baseline.entries)
            if scoring_baseline
            else 0,
        )
        print(f"AMD-native score report saved to {report_path}")

    if execution_closure_path is not None:
        if readiness is not None:
            for workload in readiness.get("workloads", []):
                key = _workload_key(
                    workload.get("workload_uuid"), workload.get("row_index")
                )
                problem_id = str(workload.get("problem_id"))
                if workload.get("status") == "ready" or (
                    problem_id,
                    key,
                ) in attempted_ready_keys:
                    continue
                closure_records.append(
                    _closure_record(
                        category=str(workload.get("category")),
                        problem_id=problem_id,
                        problem_path=str(workload.get("problem_path")),
                        workload_uuid=workload.get("workload_uuid"),
                        row_index=int(workload.get("row_index", 0)),
                        closure_status="not_attempted",
                        readiness=workload,
                        filter_reasons=["readiness_blocked"],
                    )
                )
        _write_execution_closure(
            path=execution_closure_path,
            records=closure_records,
            provenance=provenance,
            filters={
                "ready_subset": args.ready_subset is not None,
                "category": args.category,
                "limit": args.limit,
                "max_workloads": args.max_workloads,
            },
            provenance_mismatches=provenance_mismatches,
        )
        print(f"Execution closure saved to {execution_closure_path}")


if __name__ == "__main__":
    main()
