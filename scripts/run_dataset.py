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
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.dataset.evidence_refs import (
    relative_ref as _relative_ref,
    safe_sidecar_stem as _safe_sidecar_stem,
)
from sol_execbench.core.dataset.low_precision import (
    cdna4_low_precision_skip_reason as _cdna4_low_precision_skip_reason,
    should_skip_cdna4_low_precision_on_arch as _should_skip_cdna4_low_precision_on_arch,
)
from sol_execbench.core.dataset.long_tail_exclusions import (
    LONG_TAIL_EXCLUSION_STATUS,
    LongTailExclusionSidecar,
    exclusion_closure_metadata as _long_tail_exclusion_closure_metadata,
    load_long_tail_exclusions as _load_long_tail_exclusions,
    split_excluded_workloads as _split_long_tail_excluded_workloads_core,
)
from sol_execbench.core.dataset.cli_execution import (
    CLI_LOG_LIMIT as _CLI_LOG_LIMIT,
    build_cli_command,
    cli_failure_notes as _cli_failure_notes,
    run_cli as _core_run_cli,
    save_cli_log as _save_cli_log,
    save_cli_timeout_log as _save_cli_timeout_log,
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
    collect_timing_evidence_for_problem,
    inspect_traces,
    print_summary,
    write_summary_report,
)
from sol_execbench.core.dataset.runner_scoring import (
    _extend_derived_reports_for_problem,
    build_amd_score_reports_for_problem,
    scoring_baseline_coverage_report,
    write_amd_score_report,
    write_official_score_report,
)
from sol_execbench.core.dataset.solutions import (
    build_custom_solution,
    build_reference_solution,
    build_solution_for_problem,
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
from sol_execbench.core.dataset.sharding import (
    workload_prefix_lines as _core_workload_prefix_lines,
    workload_shard_paths as _core_workload_shard_paths,
)
from sol_execbench.core.scoring.amd_score import (
    AmdNativeScore,
)
from sol_execbench.core.scoring.baseline_artifact import (
    BASELINE_ARTIFACT_SCHEMA_VERSION,
    load_scoring_baseline_artifact,
)
from sol_execbench.core.scoring.official_score import OFFICIAL_AGGREGATION_POLICY

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


def _skipped_problem_summary(problem: str, reason: str) -> dict:
    return {
        "problem": problem,
        "total": 0,
        "passed": 0,
        "failed": 0,
        "latencies_ms": [],
        "failure_reasons": [],
        "skipped": 1,
        "skip_reasons": [reason],
    }


def _effective_gpu_architecture(gpu_architecture: str | None) -> str:
    if gpu_architecture and gpu_architecture != "unknown":
        return gpu_architecture
    for env_name in (
        "SOL_EXECBENCH_RUNTIME_GFX_ARCHITECTURE",
        "PYTORCH_ROCM_ARCH",
    ):
        value = os.environ.get(env_name)
        if value:
            return value.split(";", maxsplit=1)[0]
    return gpu_architecture or "unknown"


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
    """Compatibility wrapper for tests and callers importing this script module."""
    return _core_run_cli(
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


def _workload_prefix_lines(workload_path: Path, limit: int) -> tuple[list[str], bool]:
    return _core_workload_prefix_lines(workload_path, limit)


def _workload_shard_paths(
    workload_path: Path,
    *,
    shard_size: int | None,
    output_dir: Path,
) -> list[Path]:
    """Return workload JSONL paths, optionally split into fixed-size shards."""
    return _core_workload_shard_paths(
        workload_path,
        shard_size=shard_size,
        output_dir=output_dir,
    )


def _timeout_traces_for_workload_file(
    *,
    definition_name: str,
    workload_path: Path,
    solution_name: str,
    timeout_seconds: int,
    log: str,
) -> list[dict]:
    traces: list[dict] = []
    for line in workload_path.read_text().splitlines():
        if not line.strip():
            continue
        traces.append(
            {
                "definition": definition_name,
                "workload": json.loads(line),
                "solution": solution_name,
                "evaluation": {
                    "status": "TIMEOUT",
                    "environment": {"hardware": "AMD ROCm", "libs": {}},
                    "timestamp": _build_utc_timestamp(),
                    "log": f"timed out after {timeout_seconds} seconds"
                    + (f": {log}" if log else ""),
                    "correctness": None,
                    "performance": None,
                },
            }
        )
    return traces


def _safetensors_refs_for_workload_file(workload_path: Path) -> list[str]:
    refs: list[str] = []
    for line in workload_path.read_text().splitlines():
        if not line.strip():
            continue
        workload = json.loads(line)
        inputs = workload.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for input_spec in inputs.values():
            if not isinstance(input_spec, dict):
                continue
            if input_spec.get("type") != "safetensors":
                continue
            path = input_spec.get("path")
            if isinstance(path, str):
                refs.append(path)
    return refs


def _safetensors_ref_exists(
    raw_path: str, *, blob_index: "_SafetensorsBlobIndex | None" = None
) -> bool:
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        return False

    parts = path.parts
    if blob_index is not None:
        for start in range(len(parts)):
            if Path(*parts[start:]).as_posix() in blob_index.refs:
                return True
        return False

    roots: list[Path] = []
    env_root = os.environ.get("FLASHINFER_TRACE_DIR")
    if env_root:
        roots.append(Path(env_root))
    roots.append(ROOT)

    for root in roots:
        for start in range(len(parts)):
            if (root / Path(*parts[start:])).is_file():
                return True
    return False


def _missing_safetensors_refs(
    workload_path: Path, *, blob_index: "_SafetensorsBlobIndex | None" = None
) -> list[str]:
    return sorted(
        {
            ref
            for ref in _safetensors_refs_for_workload_file(workload_path)
            if not _safetensors_ref_exists(ref, blob_index=blob_index)
        }
    )


def _problems_have_safetensors_refs(problems: list[Path]) -> bool:
    for problem_dir in problems:
        workload_path = problem_dir / "workload.jsonl"
        if workload_path.exists() and _safetensors_refs_for_workload_file(
            workload_path
        ):
            return True
    return False


def _resolve_jobs(value: str, *, phase: str, problem_count: int) -> int:
    if value == "auto":
        requested = min(os.cpu_count() or 1, max(problem_count, 1), 8)
    else:
        try:
            requested = int(value)
        except ValueError as exc:
            raise SystemExit("--jobs must be a positive integer or 'auto'") from exc
    if requested < 1:
        raise SystemExit("--jobs must be a positive integer or 'auto'")
    if phase != "derived":
        if requested > 1 or value == "auto":
            print("Ignoring --jobs for GPU/profiler phases; execution remains serial.")
        return 1
    return min(requested, max(problem_count, 1))


def _ready_problem_map(ready_subset: dict | None) -> dict[str, dict]:
    return _build_ready_problem_map(ready_subset)


def _readiness_workload_map(
    readiness: dict | None,
) -> dict[tuple[str, tuple[str, str | int]], dict]:
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


def _license_boundary_metadata(manifest: dict | None) -> dict:
    if manifest is None:
        return {}
    boundary = manifest.get("license_boundary")
    if not isinstance(boundary, dict):
        return {}
    fields = (
        "source_boundary",
        "generated_artifact_source_id",
        "license",
        "redistribution_class",
        "repository_redistribution",
        "release_bundle_redistribution",
        "attribution",
    )
    return {field: boundary[field] for field in fields if field in boundary}


def _dataset_manifest_summary(
    manifest: dict | None, *, problems_dir: Path, output_dir: Path
) -> dict:
    if manifest is None:
        return {}
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    denominators = (
        manifest.get("denominators")
        if isinstance(manifest.get("denominators"), dict)
        else {}
    )
    source_root = source.get("source_root")
    source_root_ref = None
    if isinstance(source_root, str) and source_root:
        source_root_ref = _first_relative_ref(
            Path(source_root), ROOT, problems_dir, output_dir
        )
    return {
        "migration_kind": manifest.get("migration_kind"),
        "source_id": source.get("source_id"),
        "repo_id": source.get("repo_id"),
        "source_revision": source.get("revision"),
        "source_root": source_root_ref,
        "output_root": Path(str(manifest.get("output_root"))).name
        if manifest.get("output_root") is not None
        else None,
        "selected_categories": list(manifest.get("selected_categories") or []),
        "denominators": {
            key: denominators.get(key)
            for key in (
                "discovered_problems",
                "migrated_problems",
                "generated_artifacts",
                "blockers",
                "warnings",
            )
            if key in denominators
        },
        "blocker_codes": sorted(
            str(blocker.get("code"))
            for blocker in manifest.get("blockers", [])
            if isinstance(blocker, dict) and blocker.get("code") is not None
        ),
    }


def _ready_subset_summary(ready_subset: dict | None) -> dict:
    if ready_subset is None:
        return {}
    denominator = (
        ready_subset.get("denominator")
        if isinstance(ready_subset.get("denominator"), dict)
        else {}
    )
    exclusions = [
        exclusion
        for exclusion in ready_subset.get("exclusions", [])
        if isinstance(exclusion, dict)
    ]
    included_workloads = ready_subset.get("included_workloads")
    if included_workloads is None:
        included_workloads = denominator.get("included_workloads")
    excluded_workloads = ready_subset.get("excluded_workloads")
    if excluded_workloads is None:
        excluded_workloads = denominator.get("excluded_workloads")
    return {
        "dataset_root": ready_subset.get("dataset_root"),
        "selected_categories": list(ready_subset.get("selected_categories") or []),
        "included_workloads": included_workloads,
        "excluded_workloads": excluded_workloads,
        "denominator": {
            key: denominator.get(key)
            for key in (
                "total_workloads",
                "included_workloads",
                "excluded_workloads",
                "blocked_workloads",
            )
            if key in denominator
        },
        "exclusion_reason_codes": sorted(
            str(exclusion.get("reason_code"))
            for exclusion in exclusions
            if exclusion.get("reason_code") is not None
        ),
        "claim_boundary": ready_subset.get("claim_boundary") or {},
    }


def _readiness_summary(readiness: dict | None) -> dict:
    if readiness is None:
        return {}
    workloads = [
        workload
        for workload in readiness.get("workloads", [])
        if isinstance(workload, dict)
    ]
    status_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    blocker_type_counts: dict[str, int] = {}
    for workload in workloads:
        status = str(workload.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
        readiness_class = str(workload.get("readiness_class"))
        class_counts[readiness_class] = class_counts.get(readiness_class, 0) + 1
        for blocker in workload.get("blocker_reports", []):
            if isinstance(blocker, dict) and blocker.get("blocker_type") is not None:
                blocker_type = str(blocker["blocker_type"])
                blocker_type_counts[blocker_type] = (
                    blocker_type_counts.get(blocker_type, 0) + 1
                )
    return {
        "selected_categories": list(readiness.get("selected_categories") or []),
        "workloads": len(workloads),
        "status_counts": dict(sorted(status_counts.items())),
        "readiness_class_counts": dict(sorted(class_counts.items())),
        "blocker_type_counts": dict(sorted(blocker_type_counts.items())),
        "claim_boundary": readiness.get("claim_boundary") or {},
    }


def _source_refs(
    *,
    ready_subset_path: Path | None,
    readiness_path: Path | None,
    dataset_manifest_path: Path | None,
    long_tail_exclusions_path: Path | None,
    problems_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    refs: dict[str, str] = {}
    for key, path in (
        ("ready_subset", ready_subset_path),
        ("readiness", readiness_path),
        ("dataset_manifest", dataset_manifest_path),
        ("long_tail_exclusions", long_tail_exclusions_path),
    ):
        if path is not None:
            refs[key] = _first_relative_ref(path, ROOT, problems_dir, output_dir)
    return refs


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
    if args.phase != "all":
        normalized.extend(["--phase", args.phase])
    if args.category:
        normalized.extend(["--category", ",".join(args.category)])
    if args.limit is not None:
        normalized.extend(["--limit", str(args.limit)])
    if args.max_workloads is not None:
        normalized.extend(["--max-workloads", str(args.max_workloads)])
    if args.workload_shard_size is not None:
        normalized.extend(["--workload-shard-size", str(args.workload_shard_size)])
    if str(args.jobs) != "1":
        normalized.extend(["--jobs", str(args.jobs)])
    if args.execution_mode != "serial":
        normalized.extend(["--execution-mode", args.execution_mode])
    if str(args.prepare_jobs) != "auto":
        normalized.extend(["--prepare-jobs", str(args.prepare_jobs)])
    if args.gpu_jobs != 1:
        normalized.extend(["--gpu-jobs", str(args.gpu_jobs)])
    if args.timeout_policy != "record":
        normalized.extend(["--timeout-policy", args.timeout_policy])
    if args.timeout_overrides is not None:
        normalized.extend(["--timeout-overrides", Path(args.timeout_overrides).name])
    if args.long_tail_exclusions is not None:
        normalized.extend(
            ["--long-tail-exclusions", Path(args.long_tail_exclusions).name]
        )
    if args.blob_precheck != "fail":
        normalized.extend(["--blob-precheck", args.blob_precheck])
    if args.log_order != "completion":
        normalized.extend(["--log-order", args.log_order])
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
        ("--official-score-report", args.official_score_report),
        ("--amd-sol-bound-dir", args.amd_sol_bound_dir),
        ("--solar-derivation", args.solar_derivation),
        ("--timing-evidence-dir", args.timing_evidence_dir),
        ("--scoring-baseline", args.scoring_baseline),
    ):
        if value is not None:
            normalized.extend([flag, Path(value).name])
    if args.official_aggregation_policy is not None:
        normalized.extend(
            ["--official-aggregation-policy", args.official_aggregation_policy]
        )
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


def _split_long_tail_excluded_workloads(
    *,
    problem_id: str,
    workload_refs: list[dict],
    long_tail_exclusions: LongTailExclusionSidecar | None,
    workload_shard_size: int | None,
) -> tuple[list[dict], list[tuple[dict, object]]]:
    return _split_long_tail_excluded_workloads_core(
        problem_id=problem_id,
        workload_refs=workload_refs,
        exclusions=long_tail_exclusions.config if long_tail_exclusions else None,
        workload_shard_size=workload_shard_size,
    )


def _filter_long_tail_workload_file(
    *,
    workload_path: Path,
    filtered_workload_path: Path,
    category: str,
    problem_id: str,
    problem_path: str,
    long_tail_exclusions: LongTailExclusionSidecar | None,
    workload_shard_size: int | None,
) -> tuple[Path, list[dict], bool]:
    """Filter a plain workload.jsonl by long-tail config and record exclusions."""
    if long_tail_exclusions is None:
        return workload_path, [], False

    rows = _read_workload_rows(workload_path)
    selected_lines: list[str] = []
    closure_records: list[dict] = []
    excluded_any = False
    for row_index, payload, line in rows:
        workload_uuid = payload.get("uuid")
        entry = long_tail_exclusions.config.match_workload(
            problem_id=problem_id,
            workload_uuid=workload_uuid,
            row_index=row_index,
            workload_shard_size=workload_shard_size,
        )
        if entry is None:
            selected_lines.append(line)
            continue

        excluded_any = True
        metadata = _long_tail_exclusion_closure_metadata(entry)
        closure_records.append(
            _closure_record(
                category=category,
                problem_id=problem_id,
                problem_path=problem_path,
                workload_uuid=workload_uuid,
                row_index=row_index,
                closure_status=LONG_TAIL_EXCLUSION_STATUS,
                filter_reasons=metadata["filter_reasons"],
                evidence_refs=metadata["evidence_refs"],
                notes=metadata["notes"],
            )
        )

    if not excluded_any:
        return workload_path, [], False

    filtered_workload_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_workload_path.write_text(
        "\n".join(selected_lines) + ("\n" if selected_lines else "")
    )
    return filtered_workload_path, closure_records, len(selected_lines) == 0


def _derived_sidecar_exclusions_for_problem(
    *,
    problem_id: str,
    workload_path: Path,
    long_tail_exclusions: LongTailExclusionSidecar | None,
    workload_shard_size: int | None,
) -> dict[str, str]:
    """Return workload UUIDs whose missing derived sidecars should not be built."""
    if long_tail_exclusions is None:
        return {}
    problem_ids = {problem_id}
    inferred_problem_id = (
        f"{workload_path.parent.parent.name}/{workload_path.parent.name}"
    )
    problem_ids.add(inferred_problem_id)
    exclusions: dict[str, str] = {}
    for row_index, payload, _line in _read_workload_rows(workload_path):
        workload_uuid = payload.get("uuid")
        entry = None
        for candidate_problem_id in problem_ids:
            entry = long_tail_exclusions.config.match_workload(
                problem_id=candidate_problem_id,
                workload_uuid=workload_uuid,
                row_index=row_index,
                workload_shard_size=workload_shard_size,
            )
            if entry is not None:
                break
        if entry is not None and workload_uuid is not None:
            exclusions[str(workload_uuid)] = entry.closure_note()
    return exclusions


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
    amd_score_report = (
        args.amd_score_report if args.phase in {"all", "derived"} else None
    )
    amd_sol_bound_dir = (
        args.amd_sol_bound_dir if args.phase in {"all", "derived"} else None
    )
    solar_derivation = (
        args.solar_derivation if args.phase in {"all", "derived"} else None
    )
    timing_evidence_dir = (
        args.timing_evidence_dir if args.phase in {"all", "timing"} else None
    )
    return _build_requested_evidence_requirements(
        amd_score_report=amd_score_report,
        amd_sol_bound_dir=amd_sol_bound_dir,
        solar_derivation=solar_derivation,
        timing_evidence_dir=timing_evidence_dir,
    )


def _stale_provenance_mismatch(
    *,
    observed: str | None,
) -> dict[str, object]:
    return _build_stale_provenance_mismatch(observed=observed)


def _prior_closure_provenance(
    path: Path,
) -> tuple[dict | None, dict[str, object] | None]:
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
    source_refs: dict[str, str] | None = None,
) -> None:
    _write_execution_closure_report(
        path=path,
        records=records,
        provenance=provenance,
        filters=filters,
        provenance_mismatches=provenance_mismatches,
        source_refs=source_refs,
    )


def _run_existing_trace_derived_problem(
    *,
    problem_dir: Path,
    index: int,
    problem_count: int,
    problems_dir: Path,
    output_dir: Path,
    problem_ref: dict | None,
    readiness_by_workload: dict[tuple[str, tuple[str, str | int]], dict],
    max_workloads: int | None,
    amd_score_report: Path | None,
    amd_sol_bound_dir: Path | None,
    solar_derivation: Path | None,
    timing_evidence_dir: Path | None,
    scoring_baseline,
    derived_sidecar_exclusions: dict[str, str] | None = None,
) -> dict:
    problem_name = problem_dir.name
    category = problem_dir.parent.name
    problem_id = _problem_id_for(problems_dir, problem_dir)
    messages = [f"\n[{index + 1}/{problem_count}] {category}/{problem_name}"]
    closure_records: list[dict] = []
    attempted_ready_keys: set[tuple[str, tuple[str, str | int]]] = set()
    amd_scores: list[AmdNativeScore] = []

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
                max_workloads=max_workloads,
            )
        )
        filtered_workload_path = problem_output_dir / "workload.jsonl"
        filtered_workload_path.write_text(
            "\n".join(selected_lines) + ("\n" if selected_lines else "")
        )
        workload_path = filtered_workload_path
        for workload_ref in cap_filtered:
            key = _workload_key(workload_ref.get("uuid"), workload_ref.get("row_index"))
            closure_records.append(
                _closure_record(
                    category=category,
                    problem_id=problem_id,
                    problem_path=str(problem_ref.get("problem_path")),
                    workload_uuid=workload_ref.get("uuid"),
                    row_index=int(workload_ref.get("row_index", 0)),
                    closure_status="filtered",
                    readiness=readiness_by_workload.get((problem_id, key)),
                    filter_reasons=["max_workloads_cap"],
                )
            )
        for workload_ref in missing_refs:
            key = _workload_key(workload_ref.get("uuid"), workload_ref.get("row_index"))
            closure_records.append(
                _closure_record(
                    category=category,
                    problem_id=problem_id,
                    problem_path=str(problem_ref.get("problem_path")),
                    workload_uuid=workload_ref.get("uuid"),
                    row_index=int(workload_ref.get("row_index", 0)),
                    closure_status="filtered",
                    readiness=readiness_by_workload.get((problem_id, key)),
                    filter_reasons=["workload_not_found"],
                )
            )
        if not selected_workload_refs:
            return {
                "messages": messages,
                "summary": None,
                "amd_scores": amd_scores,
                "closure_records": closure_records,
                "attempted_ready_keys": attempted_ready_keys,
            }

    if not traces_path.exists():
        messages.append("  Skipping (--phase derived requires existing traces).")
        return {
            "messages": messages,
            "summary": None,
            "amd_scores": amd_scores,
            "closure_records": closure_records,
            "attempted_ready_keys": attempted_ready_keys,
        }
    try:
        traces = json.loads(traces_path.read_text())
    except (OSError, json.JSONDecodeError):
        messages.append("  Skipping (--phase derived found unreadable traces).")
        return {
            "messages": messages,
            "summary": None,
            "amd_scores": amd_scores,
            "closure_records": closure_records,
            "attempted_ready_keys": attempted_ready_keys,
        }

    summary = inspect_traces(traces, f"{category}/{problem_name}")
    if (
        amd_score_report is not None
        or amd_sol_bound_dir is not None
        or solar_derivation is not None
    ):
        _extend_derived_reports_for_problem(
            amd_scores=amd_scores,
            definition_path=definition_path,
            workload_path=workload_path,
            traces_path=traces_path,
            traces_payload=traces,
            output_dir=output_dir,
            baseline_artifact=scoring_baseline,
            sol_bound_artifact_dir=amd_sol_bound_dir,
            solar_derivation_dir=solar_derivation,
            derived_sidecar_exclusions=derived_sidecar_exclusions,
        )

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
                    skipped=True,
                    traces_path=traces_path,
                    summary_ref="summary.json",
                    solution_path=problem_output_dir / "solution.json",
                    output_dir=output_dir,
                    definition_name=str(definition_payload["name"]),
                    problem_output_dir=problem_output_dir,
                    amd_score_report=amd_score_report,
                    sol_bound_artifact_dir=amd_sol_bound_dir,
                    solar_derivation_dir=solar_derivation,
                    timing_evidence_dir=timing_evidence_dir,
                )
            )
    messages.append("  Phase derived: reused existing traces.")
    return {
        "messages": messages,
        "summary": summary,
        "amd_scores": amd_scores,
        "closure_records": closure_records,
        "attempted_ready_keys": attempted_ready_keys,
    }


@dataclass
class _PipelineTracePreparedProblem:
    index: int
    problem_count: int
    problem_dir: Path
    problem_id: str
    problem_name: str
    category: str
    definition_path: Path
    workload_path: Path
    problem_output_dir: Path
    traces_path: Path
    job_name: str
    definition: dict
    solution: dict | None
    solution_path: Path | None
    messages: list[str] = field(default_factory=list)
    summary: dict | None = None
    run_required: bool = True


@dataclass
class _PipelineTraceProblemResult:
    index: int
    messages: list[str]
    summary: dict | None


@dataclass(frozen=True)
class _SafetensorsBlobIndex:
    refs: frozenset[str]
    files: int = 0
    total_bytes: int = 0


@dataclass
class _TraceInvocationResult:
    traces: list[dict]
    shard_failure_reasons: list[str]
    messages: list[str]


def _load_timeout_overrides(path: Path | None) -> dict:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("--timeout-overrides must point to a JSON object")
    return payload


def _timeout_override_ref(path: Path | None, *, output_dir: Path) -> str | None:
    if path is None:
        return None
    return _first_relative_ref(path, output_dir, ROOT)


def _timeout_for_workload(
    *,
    workload: dict,
    problem_id: str,
    default_timeout: int,
    timeout_overrides: dict,
) -> int:
    timeout = timeout_overrides.get("default", default_timeout)
    problems = timeout_overrides.get("problems")
    if isinstance(problems, dict) and problem_id in problems:
        timeout = problems[problem_id]

    uuid = workload.get("uuid")
    workloads = timeout_overrides.get("workloads")
    if isinstance(workloads, dict):
        scoped_key = f"{problem_id}:{uuid}" if uuid is not None else None
        if scoped_key is not None and scoped_key in workloads:
            timeout = workloads[scoped_key]
        elif uuid is not None and str(uuid) in workloads:
            timeout = workloads[str(uuid)]

    try:
        timeout_int = int(timeout)
    except (TypeError, ValueError) as exc:
        raise SystemExit("timeout override values must be positive integers") from exc
    if timeout_int < 1:
        raise SystemExit("timeout override values must be positive integers")
    return timeout_int


def _timeout_for_workload_file(
    *,
    workload_path: Path,
    problem_id: str,
    default_timeout: int,
    timeout_overrides: dict,
) -> int:
    timeouts: list[int] = []
    for line in workload_path.read_text().splitlines():
        if not line.strip():
            continue
        timeouts.append(
            _timeout_for_workload(
                workload=json.loads(line),
                problem_id=problem_id,
                default_timeout=default_timeout,
                timeout_overrides=timeout_overrides,
            )
        )
    return max(timeouts) if timeouts else default_timeout


def _build_safetensors_blob_index() -> _SafetensorsBlobIndex:
    roots: list[Path] = []
    env_root = os.environ.get("FLASHINFER_TRACE_DIR")
    if env_root:
        roots.append(Path(env_root))
    roots.append(ROOT / "data" / "flashinfer-trace")

    refs: set[str] = set()
    total_bytes = 0
    files = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.safetensors"):
            if not path.is_file():
                continue
            files += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                rel = path.name
            refs.add(rel)
            parts = Path(rel).parts
            for start in range(len(parts)):
                refs.add(Path(*parts[start:]).as_posix())
    return _SafetensorsBlobIndex(
        refs=frozenset(refs),
        files=files,
        total_bytes=total_bytes,
    )


def _resolve_prepare_jobs(value: str, *, problem_count: int) -> int:
    if value == "auto":
        requested = min(os.cpu_count() or 1, max(problem_count, 1), 8)
    else:
        try:
            requested = int(value)
        except ValueError as exc:
            raise SystemExit(
                "--prepare-jobs must be a positive integer or 'auto'"
            ) from exc
    if requested < 1:
        raise SystemExit("--prepare-jobs must be a positive integer or 'auto'")
    return min(requested, max(problem_count, 1))


def _prepare_pipeline_trace_problem(
    *,
    problem_dir: Path,
    index: int,
    problem_count: int,
    problems_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    effective_gpu_architecture: str,
    provenance: dict,
    blob_index: _SafetensorsBlobIndex | None,
) -> _PipelineTracePreparedProblem:
    problem_name = problem_dir.name
    category = problem_dir.parent.name
    problem_id = _problem_id_for(problems_dir, problem_dir)
    definition_path = problem_dir / "definition.json"
    workload_path = problem_dir / "workload.jsonl"
    problem_output_dir = output_dir / category / problem_name
    traces_path = problem_output_dir / "traces.json"
    definition = json.loads(definition_path.read_text())
    messages = [f"\n[{index + 1}/{problem_count}] {category}/{problem_name}"]

    if _should_skip_cdna4_low_precision_on_arch(definition, effective_gpu_architecture):
        skip_reason = _cdna4_low_precision_skip_reason(effective_gpu_architecture)
        messages.append(f"  Skipping ({skip_reason}).")
        return _PipelineTracePreparedProblem(
            index=index,
            problem_count=problem_count,
            problem_dir=problem_dir,
            problem_id=problem_id,
            problem_name=problem_name,
            category=category,
            definition_path=definition_path,
            workload_path=workload_path,
            problem_output_dir=problem_output_dir,
            traces_path=traces_path,
            job_name=f"ref_{problem_name[:40]}",
            definition=definition,
            solution=None,
            solution_path=None,
            messages=messages,
            summary=_skipped_problem_summary(f"{category}/{problem_name}", skip_reason),
            run_required=False,
        )

    if traces_path.exists():
        try:
            traces = json.loads(traces_path.read_text())
            summary = inspect_traces(traces, f"{category}/{problem_name}")
            reuse_decision = _dataset_reuse_decision(
                rerun=args.rerun,
                traces_path=traces_path,
                failed_count=int(summary["failed"]),
                execution_closure_path=None,
                provenance=provenance,
            )
        except (OSError, json.JSONDecodeError):
            summary = None
            reuse_decision = None
            messages.append("  Re-running (previous trace output is unreadable).")
        if reuse_decision is not None and reuse_decision.should_reuse:
            messages.append("  Skipping (already passed). Use --rerun to re-evaluate.")
            return _PipelineTracePreparedProblem(
                index=index,
                problem_count=problem_count,
                problem_dir=problem_dir,
                problem_id=problem_id,
                problem_name=problem_name,
                category=category,
                definition_path=definition_path,
                workload_path=workload_path,
                problem_output_dir=problem_output_dir,
                traces_path=traces_path,
                job_name=f"ref_{problem_name[:40]}",
                definition=definition,
                solution=None,
                solution_path=None,
                messages=messages,
                summary=summary,
                run_required=False,
            )
        elif reuse_decision is not None and reuse_decision.reason == "previous_failed":
            failed = summary["failed"] if summary is not None else "unknown"
            messages.append(f"  Re-running (previous run had {failed} failures).")
        elif reuse_decision is not None and reuse_decision.reason == "rerun_requested":
            messages.append("  Re-running (--rerun requested).")

    if problem_output_dir.exists():
        shutil.rmtree(problem_output_dir)
    problem_output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_workloads is not None:
        lines, truncated = _workload_prefix_lines(workload_path, args.max_workloads)
        if truncated:
            workload_path = problem_output_dir / "workload.jsonl"
            workload_path.write_text("\n".join(lines[: args.max_workloads]))

    if args.blob_precheck != "off":
        missing_refs = _missing_safetensors_refs(workload_path, blob_index=blob_index)
        if missing_refs:
            message = (
                "missing safetensors blobs: "
                + ", ".join(missing_refs[:3])
                + (" ..." if len(missing_refs) > 3 else "")
            )
            messages.append(f"  Blob precheck: {message}")
            if args.blob_precheck == "fail":
                return _PipelineTracePreparedProblem(
                    index=index,
                    problem_count=problem_count,
                    problem_dir=problem_dir,
                    problem_id=problem_id,
                    problem_name=problem_name,
                    category=category,
                    definition_path=definition_path,
                    workload_path=workload_path,
                    problem_output_dir=problem_output_dir,
                    traces_path=traces_path,
                    job_name=f"ref_{problem_name[:40]}",
                    definition=definition,
                    solution=None,
                    solution_path=None,
                    messages=messages,
                    summary={
                        "problem": f"{category}/{problem_name}",
                        "total": 1,
                        "passed": 0,
                        "failed": 1,
                        "latencies_ms": [],
                        "failure_reasons": [f"  [MISSING_SAFETENSORS] {message}"],
                    },
                    run_required=False,
                )

    if args.solution_name:
        solution_file = problem_dir / args.solution_name
        if not solution_file.exists():
            messages.append(f"  Skipping: {args.solution_name} not found")
            return _PipelineTracePreparedProblem(
                index=index,
                problem_count=problem_count,
                problem_dir=problem_dir,
                problem_id=problem_id,
                problem_name=problem_name,
                category=category,
                definition_path=definition_path,
                workload_path=workload_path,
                problem_output_dir=problem_output_dir,
                traces_path=traces_path,
                job_name=f"ref_{problem_name[:40]}",
                definition=definition,
                solution=None,
                solution_path=None,
                messages=messages,
                run_required=False,
            )
    elif "reference" not in definition or not definition["reference"].strip():
        messages.append("  Skipping: no reference code")
        return _PipelineTracePreparedProblem(
            index=index,
            problem_count=problem_count,
            problem_dir=problem_dir,
            problem_id=problem_id,
            problem_name=problem_name,
            category=category,
            definition_path=definition_path,
            workload_path=workload_path,
            problem_output_dir=problem_output_dir,
            traces_path=traces_path,
            job_name=f"ref_{problem_name[:40]}",
            definition=definition,
            solution=None,
            solution_path=None,
            messages=messages,
            run_required=False,
        )

    solution = build_solution_for_problem(definition, problem_dir, args.solution_name)
    solution_path = problem_output_dir / "solution.json"
    solution_path.write_text(json.dumps(solution, indent=2))
    return _PipelineTracePreparedProblem(
        index=index,
        problem_count=problem_count,
        problem_dir=problem_dir,
        problem_id=problem_id,
        problem_name=problem_name,
        category=category,
        definition_path=definition_path,
        workload_path=workload_path,
        problem_output_dir=problem_output_dir,
        traces_path=traces_path,
        job_name=f"ref_{problem_name[:40]}",
        definition=definition,
        solution=solution,
        solution_path=solution_path,
        messages=messages,
    )


def _run_trace_invocations(
    *,
    definition_path: Path,
    workload_path: Path,
    solution_path: Path,
    output_dir: Path,
    job_name: str,
    definition: dict,
    solution: dict,
    problem_id: str,
    args: argparse.Namespace,
    config_path: Path | None,
) -> _TraceInvocationResult:
    workload_shards = _workload_shard_paths(
        workload_path,
        shard_size=args.workload_shard_size,
        output_dir=output_dir,
    )
    traces: list[dict] = []
    shard_failure_reasons: list[str] = []
    messages: list[str] = []
    if len(workload_shards) > 1:
        messages.append(
            f"  Running {len(workload_shards)} workload shards "
            f"(--workload-shard-size {args.workload_shard_size})."
        )
    for shard_index, shard_workload_path in enumerate(workload_shards, start=1):
        shard_job_name = (
            job_name
            if len(workload_shards) == 1
            else f"{job_name}_shard{shard_index:04d}"
        )
        shard_timeout = _timeout_for_workload_file(
            workload_path=shard_workload_path,
            problem_id=problem_id,
            default_timeout=args.timeout,
            timeout_overrides=args.timeout_override_rules,
        )
        shard_traces = run_cli(
            definition_path=definition_path,
            workload_path=shard_workload_path,
            solution_path=solution_path,
            output_dir=output_dir,
            job_name=shard_job_name,
            timeout=shard_timeout,
            config_path=config_path,
            keep_staging=args.keep_staging,
            verbose=args.verbose,
        )
        if shard_traces is None:
            cli_log = output_dir / f"{shard_job_name}_cli.log"
            notes = _cli_failure_notes(cli_log)
            timeout_notes = [
                note for note in notes if note.startswith("CLI timed out after ")
            ]
            if timeout_notes and args.timeout_policy == "record":
                traces.extend(
                    _timeout_traces_for_workload_file(
                        definition_name=str(definition["name"]),
                        workload_path=shard_workload_path,
                        solution_name=str(solution["name"]),
                        timeout_seconds=shard_timeout,
                        log="; ".join(timeout_notes),
                    )
                )
                continue
            if len(workload_shards) == 1:
                shard_failure_reasons.extend(notes)
            else:
                suffix = f" ({'; '.join(notes)})" if notes else ""
                shard_failure_reasons.append(
                    f"workload shard {shard_index}/{len(workload_shards)} "
                    f"returned no traces{suffix}"
                )
            continue
        traces.extend(shard_traces)

    return _TraceInvocationResult(
        traces=traces,
        shard_failure_reasons=shard_failure_reasons,
        messages=messages,
    )


def _run_pipeline_trace_problem(
    *,
    prepared: _PipelineTracePreparedProblem,
    args: argparse.Namespace,
    config_path: Path | None,
) -> _PipelineTraceProblemResult:
    messages = [*prepared.messages]
    if not prepared.run_required:
        return _PipelineTraceProblemResult(
            index=prepared.index,
            messages=messages,
            summary=prepared.summary,
        )

    assert prepared.solution is not None
    assert prepared.solution_path is not None

    invocation = _run_trace_invocations(
        definition_path=prepared.definition_path,
        workload_path=prepared.workload_path,
        solution_path=prepared.solution_path,
        output_dir=prepared.problem_output_dir,
        job_name=prepared.job_name,
        definition=prepared.definition,
        solution=prepared.solution,
        problem_id=prepared.problem_id,
        args=args,
        config_path=config_path,
    )
    traces = invocation.traces
    shard_failure_reasons = invocation.shard_failure_reasons
    messages.extend(invocation.messages)

    if not traces and shard_failure_reasons:
        messages.append("  ERROR: CLI returned no traces")
        return _PipelineTraceProblemResult(
            index=prepared.index,
            messages=messages,
            summary={
                "problem": f"{prepared.category}/{prepared.problem_name}",
                "total": len(shard_failure_reasons),
                "passed": 0,
                "failed": len(shard_failure_reasons),
                "latencies_ms": [],
                "failure_reasons": shard_failure_reasons,
            },
        )

    prepared.traces_path.write_text(json.dumps(traces, indent=2))
    summary = inspect_traces(traces, f"{prepared.category}/{prepared.problem_name}")
    if shard_failure_reasons:
        summary["total"] += len(shard_failure_reasons)
        summary["failed"] += len(shard_failure_reasons)
        summary["failure_reasons"].extend(shard_failure_reasons)

    status = "OK" if summary["failed"] == 0 else "FAIL"
    messages.append(f"  {status}: {summary['passed']}/{summary['total']} passed")
    for reason in summary["failure_reasons"][:3]:
        messages.append(f"  {reason}")
    return _PipelineTraceProblemResult(
        index=prepared.index,
        messages=messages,
        summary=summary,
    )


def _run_pipeline_post_trace_outputs(
    *,
    problems: list[Path],
    output_dir: Path,
    args: argparse.Namespace,
    scoring_baseline,
    amd_scores: list[AmdNativeScore],
    benchmark_config: BenchmarkConfig,
    config_path: Path | None,
) -> None:
    if not (
        args.amd_score_report is not None
        or args.amd_sol_bound_dir is not None
        or args.solar_derivation is not None
        or args.timing_evidence_dir is not None
    ):
        return

    for problem_dir in problems:
        problem_name = problem_dir.name
        category = problem_dir.parent.name
        definition_path = problem_dir / "definition.json"
        problem_output_dir = output_dir / category / problem_name
        prepared_workload_path = problem_output_dir / "workload.jsonl"
        workload_path = (
            prepared_workload_path
            if prepared_workload_path.exists()
            else problem_dir / "workload.jsonl"
        )
        traces_path = problem_output_dir / "traces.json"
        if not traces_path.exists():
            continue
        try:
            traces = json.loads(traces_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

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

        if args.timing_evidence_dir is None:
            continue
        solution_path = problem_output_dir / "solution.json"
        if not solution_path.exists():
            continue
        try:
            solution = json.loads(solution_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        timing_root = args.timing_evidence_dir.resolve()
        collect_timing_evidence_for_problem(
            definition_path=definition_path,
            workload_path=workload_path,
            solution_path=solution_path,
            output_dir=problem_output_dir,
            timing_evidence_root=timing_root / category,
            job_name=f"ref_{problem_name[:40]}",
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
        "--phase",
        choices=("all", "traces", "derived", "timing"),
        default="all",
        help=(
            "Select the dataset pass to run. Default 'all' preserves the legacy "
            "single-pass behavior; 'traces' runs GPU validation only, 'derived' "
            "builds AMD/SOLAR reports from existing traces, and 'timing' collects "
            "profiler timing evidence from existing traces."
        ),
    )
    ap.add_argument(
        "--jobs",
        default="1",
        help=(
            "Parallel workers for safe CPU/I/O-only phases. Use a positive integer "
            "or 'auto'. Only --phase derived uses more than one worker; GPU and "
            "profiler phases remain serial."
        ),
    )
    ap.add_argument(
        "--execution-mode",
        choices=("serial", "pipeline"),
        default="serial",
        help=(
            "Execution scheduler. Default 'serial' preserves legacy ordering. "
            "'pipeline' overlaps trace-stage CPU preparation with serial GPU "
            "evaluation; it currently supports --phase traces or all."
        ),
    )
    ap.add_argument(
        "--prepare-jobs",
        default="auto",
        help=(
            "CPU preparation workers for --execution-mode pipeline. Use a positive "
            "integer or 'auto'. GPU evaluation remains controlled by --gpu-jobs."
        ),
    )
    ap.add_argument(
        "--gpu-jobs",
        type=int,
        default=1,
        help=(
            "GPU evaluation workers for --execution-mode pipeline. Only 1 is "
            "currently supported to preserve benchmark isolation."
        ),
    )
    ap.add_argument(
        "--timeout-policy",
        choices=("record", "fail"),
        default="record",
        help=(
            "How dataset runs handle sol-execbench CLI timeouts. 'record' emits "
            "TIMEOUT traces for the affected workload shard; 'fail' preserves "
            "the no-trace failure."
        ),
    )
    ap.add_argument(
        "--timeout-overrides",
        type=Path,
        default=None,
        help=(
            "Optional JSON object with per-problem or per-workload timeout "
            "overrides. Supported keys: default, problems, workloads."
        ),
    )
    ap.add_argument(
        "--long-tail-exclusions",
        type=Path,
        default=None,
        help=(
            "Optional JSON config for temporarily excluding known long-tail "
            "problems, workloads, or workload shards from execution while "
            "preserving closure accounting."
        ),
    )
    ap.add_argument(
        "--blob-precheck",
        choices=("fail", "warn", "off"),
        default="fail",
        help=(
            "Precheck workload safetensors references before invoking GPU "
            "evaluation. 'fail' records a problem failure without running the CLI, "
            "'warn' logs missing refs and continues, and 'off' disables the check."
        ),
    )
    ap.add_argument(
        "--log-order",
        choices=("completion", "problem"),
        default="completion",
        help=(
            "Pipeline log ordering. 'completion' prints each problem as soon as "
            "its GPU evaluation finishes; 'problem' buffers logs until earlier "
            "problem indexes are available."
        ),
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
        "--workload-shard-size",
        type=int,
        default=None,
        help=(
            "Optional number of workloads per sol-execbench invocation. "
            "When set, a problem workload file is split into temporary shards "
            "and traces are merged back under the original problem."
        ),
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
        "--official-score-report",
        type=Path,
        default=None,
        help="Write official_score_evidence.v1 after derived scoring.",
    )
    ap.add_argument(
        "--official-aggregation-policy",
        default=None,
        help=f"Required policy: {OFFICIAL_AGGREGATION_POLICY}.",
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

    if args.workload_shard_size is not None and args.workload_shard_size < 1:
        ap.error("--workload-shard-size must be a positive integer")
    if args.gpu_jobs != 1:
        ap.error("--gpu-jobs currently only supports 1")
    if args.official_score_report is not None:
        if args.phase not in {"all", "derived"}:
            ap.error("--official-score-report requires --phase all or derived")
        if args.amd_score_report is None:
            ap.error("--official-score-report requires --amd-score-report")
        if args.scoring_baseline is None:
            ap.error("--official-score-report requires --scoring-baseline")
        if args.official_aggregation_policy != OFFICIAL_AGGREGATION_POLICY:
            ap.error(
                "--official-score-report requires "
                f"--official-aggregation-policy {OFFICIAL_AGGREGATION_POLICY}"
            )
    if args.execution_mode == "pipeline":
        if args.phase not in {"traces", "all"}:
            ap.error(
                "--execution-mode pipeline currently requires --phase traces or all"
            )
        if args.ready_subset is not None or args.execution_closure is not None:
            ap.error(
                "--execution-mode pipeline currently does not support "
                "--ready-subset or --execution-closure"
            )
        if args.long_tail_exclusions is not None:
            ap.error(
                "--execution-mode pipeline currently does not support "
                "--long-tail-exclusions"
            )
    args.timeout_override_rules = _load_timeout_overrides(args.timeout_overrides)
    try:
        long_tail_exclusions = _load_long_tail_exclusions(args.long_tail_exclusions)
    except ValueError as exc:
        ap.error(str(exc))

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
        if args.ready_subset is not None or args.long_tail_exclusions is not None
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
    dataset_manifest_summary = _dataset_manifest_summary(
        dataset_manifest,
        problems_dir=problems_dir,
        output_dir=output_dir,
    )
    readiness_summary = _readiness_summary(readiness)
    ready_subset_summary = _ready_subset_summary(ready_subset)
    source_refs = _source_refs(
        ready_subset_path=args.ready_subset.resolve() if args.ready_subset else None,
        readiness_path=args.readiness.resolve() if args.readiness else None,
        dataset_manifest_path=args.dataset_manifest.resolve()
        if args.dataset_manifest
        else None,
        long_tail_exclusions_path=(
            args.long_tail_exclusions.resolve() if args.long_tail_exclusions else None
        ),
        problems_dir=problems_dir,
        output_dir=output_dir,
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
                workload_refs, long_tail_filtered = _split_long_tail_excluded_workloads(
                    problem_id=problem_id,
                    workload_refs=list(problem_ref.get("workloads", [])),
                    long_tail_exclusions=long_tail_exclusions,
                    workload_shard_size=args.workload_shard_size,
                )
                if long_tail_filtered:
                    problem_ref["workloads"] = workload_refs
                    for workload_ref, exclusion in long_tail_filtered:
                        key = _workload_key(
                            workload_ref.get("uuid"), workload_ref.get("row_index")
                        )
                        readiness_record = readiness_by_workload.get((problem_id, key))
                        metadata = _long_tail_exclusion_closure_metadata(exclusion)
                        closure_records.append(
                            _closure_record(
                                category=str(problem_ref.get("category")),
                                problem_id=problem_id,
                                problem_path=str(problem_ref.get("problem_path")),
                                workload_uuid=workload_ref.get("uuid"),
                                row_index=int(workload_ref.get("row_index", 0)),
                                closure_status=LONG_TAIL_EXCLUSION_STATUS,
                                readiness=readiness_record,
                                filter_reasons=metadata["filter_reasons"],
                                evidence_refs=metadata["evidence_refs"],
                                notes=metadata["notes"],
                            )
                        )
                if (
                    category_filter
                    and problem_ref.get("category") not in category_filter
                ):
                    reason.append("category_filter")
                elif problem_id not in discovered_by_id:
                    reason.append("problem_not_discovered")
                elif not problem_ref.get("workloads"):
                    continue
                else:
                    selected_ids.append(problem_id)
                    continue
                for workload_ref in problem_ref.get("workloads", []):
                    readiness_record = readiness_by_workload.get(
                        (
                            problem_id,
                            _workload_key(
                                workload_ref.get("uuid"), workload_ref.get("row_index")
                            ),
                        )
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
            if args.limit is not None:
                limited_ids = selected_ids[: args.limit]
                filtered_ids = set(selected_ids[args.limit :])
                for problem_id in selected_ids[args.limit :]:
                    problem_ref = ready_problems[problem_id]
                    for workload_ref in problem_ref.get("workloads", []):
                        readiness_record = readiness_by_workload.get(
                            (
                                problem_id,
                                _workload_key(
                                    workload_ref.get("uuid"),
                                    workload_ref.get("row_index"),
                                ),
                            )
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
        elif args.limit is not None:
            problems = problems[: args.limit]

        print(f"Discovered {len(problems)} problems under {problems_dir}")
        if not problems and ready_subset is None:
            print("No problems found. Check the directory path.")
            sys.exit(1)

    jobs = _resolve_jobs(str(args.jobs), phase=args.phase, problem_count=len(problems))
    if args.phase == "derived" and jobs > 1:
        print(f"Using {jobs} derived workers.")
    prepare_jobs = _resolve_prepare_jobs(
        str(args.prepare_jobs), problem_count=len(problems)
    )
    if args.execution_mode == "pipeline":
        print(
            f"Using pipeline trace scheduler "
            f"(prepare_jobs={prepare_jobs}, gpu_jobs={args.gpu_jobs})."
        )
    blob_index = (
        _build_safetensors_blob_index()
        if args.blob_precheck != "off" and _problems_have_safetensors_refs(problems)
        else None
    )
    if blob_index is not None:
        print(
            "Safetensors blob index: "
            f"{blob_index.files} files, {blob_index.total_bytes / (1024**3):.2f} GiB"
        )

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
        load_scoring_baseline_artifact(
            args.scoring_baseline,
            required_schema_version=(
                BASELINE_ARTIFACT_SCHEMA_VERSION
                if args.official_score_report is not None
                else None
            ),
        )
        if args.scoring_baseline is not None
        else None
    )
    if (
        args.official_score_report is not None
        and scoring_baseline is not None
        and scoring_baseline.schema_version != BASELINE_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            "--official-score-report requires --scoring-baseline schema_version "
            f"{BASELINE_ARTIFACT_SCHEMA_VERSION!r}; got "
            f"{scoring_baseline.schema_version!r}"
        )
    provenance = {
        "command_args": _normalized_command_args(args),
        "dataset_root": _first_relative_ref(problems_dir, ROOT),
        "selected_categories": args.category,
        "limit": args.limit,
        "max_workloads": args.max_workloads,
        "workload_shard_size": args.workload_shard_size,
        "execution_mode": args.execution_mode,
        "prepare_jobs": prepare_jobs if args.execution_mode == "pipeline" else None,
        "gpu_jobs": args.gpu_jobs if args.execution_mode == "pipeline" else None,
        "timeout_policy": args.timeout_policy,
        "timeout_overrides": _timeout_override_ref(
            args.timeout_overrides, output_dir=output_dir
        ),
        "blob_precheck": args.blob_precheck,
        "log_order": args.log_order,
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
        "ready_subset_checksum": _sidecar_checksum(
            ready_subset, "ready_subset_checksum"
        ),
        "ready_subset_summary": ready_subset_summary,
        "readiness_path": (
            _first_relative_ref(args.readiness, ROOT, problems_dir, output_dir)
            if args.readiness
            else None
        ),
        "readiness_checksum": (
            _sidecar_checksum(readiness, "readiness_checksum")
            or (ready_subset or {}).get("readiness_checksum")
        ),
        "readiness_summary": readiness_summary,
        "dataset_manifest_path": (
            _first_relative_ref(args.dataset_manifest, ROOT, problems_dir, output_dir)
            if args.dataset_manifest
            else None
        ),
        "dataset_manifest_checksum": _manifest_checksum(dataset_manifest),
        "dataset_source_id": dataset_manifest_summary.get("source_id"),
        "dataset_migration_kind": dataset_manifest_summary.get("migration_kind"),
        "dataset_source_revision": dataset_manifest_summary.get("source_revision"),
        "dataset_license_boundary": _license_boundary_metadata(dataset_manifest),
        "dataset_manifest_summary": dataset_manifest_summary,
        "long_tail_exclusions_path": (
            _first_relative_ref(
                args.long_tail_exclusions, ROOT, problems_dir, output_dir
            )
            if args.long_tail_exclusions
            else None
        ),
        "long_tail_exclusions_checksum": (
            long_tail_exclusions.checksum if long_tail_exclusions is not None else None
        ),
        "long_tail_exclusions_summary": (
            long_tail_exclusions.summary if long_tail_exclusions is not None else {}
        ),
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
            "official_score_report": (
                _first_relative_ref(args.official_score_report, output_dir, ROOT)
                if args.official_score_report
                else None
            ),
            "official_aggregation_policy": args.official_aggregation_policy,
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
        if readiness is not None:
            for workload in readiness.get("workloads", []):
                if workload.get("status") == "ready":
                    continue
                closure_records.append(
                    _closure_record(
                        category=str(workload.get("category")),
                        problem_id=str(workload.get("problem_id")),
                        problem_path=str(workload.get("problem_path")),
                        workload_uuid=workload.get("workload_uuid"),
                        row_index=int(workload.get("row_index", 0)),
                        closure_status="not_attempted",
                        readiness=workload,
                        filter_reasons=["readiness_blocked"],
                    )
                )
        summary_path = write_summary_report(output_dir, summaries)
        _write_execution_closure(
            path=execution_closure_path or (output_dir / "execution_closure.json"),
            records=closure_records,
            provenance=provenance,
            filters={
                "ready_subset": True,
                "category": args.category,
                "limit": args.limit,
                "max_workloads": args.max_workloads,
                "workload_shard_size": args.workload_shard_size,
            },
            provenance_mismatches=provenance_mismatches,
            source_refs=source_refs,
        )
        print(f"\nSummary saved to {summary_path}")
        print(f"Execution closure saved to {execution_closure_path}")
        return

    attempted_ready_keys: set[tuple[str, tuple[str, str | int]]] = set()
    run_trace_phase = args.phase in {"all", "traces"}
    run_derived_phase = args.phase in {"all", "derived"}
    run_timing_phase = args.phase in {"all", "timing"}

    if args.execution_mode == "pipeline":
        effective_gpu_architecture = _effective_gpu_architecture(args.gpu_architecture)
        pipeline_results: list[_PipelineTraceProblemResult | None] = [
            None for _ in problems
        ]
        next_log_index = 0
        with ThreadPoolExecutor(max_workers=prepare_jobs) as executor:
            future_to_index = {
                executor.submit(
                    _prepare_pipeline_trace_problem,
                    problem_dir=problem_dir,
                    index=index,
                    problem_count=len(problems),
                    problems_dir=problems_dir,
                    output_dir=output_dir,
                    args=args,
                    effective_gpu_architecture=effective_gpu_architecture,
                    provenance=provenance,
                    blob_index=blob_index,
                ): index
                for index, problem_dir in enumerate(problems)
            }
            for future in as_completed(future_to_index):
                prepared = future.result()
                result = _run_pipeline_trace_problem(
                    prepared=prepared,
                    args=args,
                    config_path=config_path,
                )
                pipeline_results[result.index] = result
                if args.log_order == "completion":
                    for message in result.messages:
                        print(message)
                else:
                    while (
                        next_log_index < len(pipeline_results)
                        and pipeline_results[next_log_index] is not None
                    ):
                        buffered = pipeline_results[next_log_index]
                        assert buffered is not None
                        for message in buffered.messages:
                            print(message)
                        next_log_index += 1

        for result in pipeline_results:
            if result is not None and result.summary is not None:
                summaries.append(result.summary)

        if args.phase == "all":
            _run_pipeline_post_trace_outputs(
                problems=problems,
                output_dir=output_dir,
                args=args,
                scoring_baseline=scoring_baseline,
                amd_scores=amd_scores,
                benchmark_config=benchmark_config,
                config_path=config_path,
            )

        print_summary(summaries)
        summary_path = write_summary_report(output_dir, summaries)
        print(f"\nSummary saved to {summary_path}")
        print(f"Per-problem traces saved under {output_dir}")
        if args.phase == "all" and args.amd_score_report is not None:
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
        if args.phase == "all" and args.official_score_report is not None:
            report_path = args.official_score_report.resolve()
            write_official_score_report(
                report_path,
                amd_scores,
                aggregation_policy=args.official_aggregation_policy,
                coverage_report=scoring_baseline_coverage_report(
                    scoring_baseline, amd_scores
                ),
                source_score_ref=_relative_ref(
                    args.amd_score_report.resolve(), output_dir
                ),
            )
            print(f"Official score report saved to {report_path}")
        return

    if args.phase == "derived" and jobs > 1:

        def run_derived_item(item):
            problem_dir = item[1]
            problem_id = _problem_id_for(problems_dir, problem_dir)
            return _run_existing_trace_derived_problem(
                problem_dir=problem_dir,
                index=item[0],
                problem_count=len(problems),
                problems_dir=problems_dir,
                output_dir=output_dir,
                problem_ref=ready_problems.get(problem_id),
                readiness_by_workload=readiness_by_workload,
                max_workloads=args.max_workloads,
                amd_score_report=args.amd_score_report,
                amd_sol_bound_dir=args.amd_sol_bound_dir,
                solar_derivation=args.solar_derivation,
                timing_evidence_dir=None,
                scoring_baseline=scoring_baseline,
                derived_sidecar_exclusions=_derived_sidecar_exclusions_for_problem(
                    problem_id=problem_id,
                    workload_path=problem_dir / "workload.jsonl",
                    long_tail_exclusions=long_tail_exclusions,
                    workload_shard_size=args.workload_shard_size,
                ),
            )

        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(run_derived_item, item) for item in enumerate(problems)
            ]
            for future in as_completed(futures):
                result = future.result()
                for message in result["messages"]:
                    print(message)
                if result["summary"] is not None:
                    summaries.append(result["summary"])
                amd_scores.extend(result["amd_scores"])
                closure_records.extend(result["closure_records"])
                attempted_ready_keys.update(result["attempted_ready_keys"])

        print_summary(summaries)
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

        if args.official_score_report is not None:
            report_path = args.official_score_report.resolve()
            write_official_score_report(
                report_path,
                amd_scores,
                aggregation_policy=args.official_aggregation_policy,
                coverage_report=scoring_baseline_coverage_report(
                    scoring_baseline, amd_scores
                ),
                source_score_ref=_relative_ref(
                    args.amd_score_report.resolve(), output_dir
                ),
            )
            print(f"Official score report saved to {report_path}")

        if execution_closure_path is not None:
            if readiness is not None:
                for workload in readiness.get("workloads", []):
                    key = _workload_key(
                        workload.get("workload_uuid"), workload.get("row_index")
                    )
                    problem_id = str(workload.get("problem_id"))
                    if (
                        workload.get("status") == "ready"
                        or (
                            problem_id,
                            key,
                        )
                        in attempted_ready_keys
                    ):
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
                    "workload_shard_size": args.workload_shard_size,
                    "jobs": jobs,
                },
                provenance_mismatches=provenance_mismatches,
                source_refs=source_refs,
            )
            print(f"Execution closure saved to {execution_closure_path}")
        return

    # ROCm execution stays serial here; runner helpers provide the future scheduling seam.
    effective_gpu_architecture = _effective_gpu_architecture(args.gpu_architecture)
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
            filtered_workload_path.write_text(
                "\n".join(selected_lines) + ("\n" if selected_lines else "")
            )
            workload_path = filtered_workload_path
            for workload_ref in cap_filtered:
                key = _workload_key(
                    workload_ref.get("uuid"), workload_ref.get("row_index")
                )
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
                key = _workload_key(
                    workload_ref.get("uuid"), workload_ref.get("row_index")
                )
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
        derived_sidecar_exclusions = _derived_sidecar_exclusions_for_problem(
            problem_id=problem_id,
            workload_path=workload_path,
            long_tail_exclusions=long_tail_exclusions,
            workload_shard_size=args.workload_shard_size,
        )
        if _should_skip_cdna4_low_precision_on_arch(
            definition_payload, effective_gpu_architecture
        ):
            skip_reason = _cdna4_low_precision_skip_reason(effective_gpu_architecture)
            print(f"  Skipping ({skip_reason}).")
            summaries.append(
                _skipped_problem_summary(f"{category}/{problem_name}", skip_reason)
            )
            if selected_workload_refs is not None:
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
                            closure_status="filtered",
                            readiness=readiness_by_workload.get((problem_id, key)),
                            filter_reasons=["cdna3_low_precision_hardware_unsupported"],
                            notes=[skip_reason],
                        )
                    )
            continue

        if not run_trace_phase:
            if not traces_path.exists():
                print(f"  Skipping (--phase {args.phase} requires existing traces).")
                continue
            try:
                traces = json.loads(traces_path.read_text())
            except (OSError, json.JSONDecodeError):
                print(f"  Skipping (--phase {args.phase} found unreadable traces).")
                continue

            summary = inspect_traces(traces, f"{category}/{problem_name}")
            summaries.append(summary)

            if run_derived_phase and (
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
                    derived_sidecar_exclusions=derived_sidecar_exclusions,
                )

            if run_timing_phase and args.timing_evidence_dir is not None:
                definition = definition_payload
                if args.solution_name:
                    solution_file = problem_dir / args.solution_name
                    if not solution_file.exists():
                        print(f"  Skipping timing: {args.solution_name} not found")
                    else:
                        solution = build_solution_for_problem(
                            definition, problem_dir, args.solution_name
                        )
                        solution_path = problem_output_dir / "solution.json"
                        solution_path.write_text(json.dumps(solution, indent=2))
                        timing_root = args.timing_evidence_dir.resolve()
                        collect_timing_evidence_for_problem(
                            definition_path=definition_path,
                            workload_path=workload_path,
                            solution_path=solution_path,
                            output_dir=problem_output_dir,
                            timing_evidence_root=timing_root / category,
                            job_name=f"ref_{problem_name[:40]}",
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
                elif (
                    "reference" not in definition or not definition["reference"].strip()
                ):
                    print("  Skipping timing: no reference code")
                else:
                    solution = build_solution_for_problem(
                        definition, problem_dir, args.solution_name
                    )
                    solution_path = problem_output_dir / "solution.json"
                    solution_path.write_text(json.dumps(solution, indent=2))
                    timing_root = args.timing_evidence_dir.resolve()
                    collect_timing_evidence_for_problem(
                        definition_path=definition_path,
                        workload_path=workload_path,
                        solution_path=solution_path,
                        output_dir=problem_output_dir,
                        timing_evidence_root=timing_root / category,
                        job_name=f"ref_{problem_name[:40]}",
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
                            amd_score_report=args.amd_score_report
                            if run_derived_phase
                            else None,
                            sol_bound_artifact_dir=args.amd_sol_bound_dir
                            if run_derived_phase
                            else None,
                            solar_derivation_dir=args.solar_derivation
                            if run_derived_phase
                            else None,
                            timing_evidence_dir=args.timing_evidence_dir
                            if run_timing_phase
                            else None,
                        )
                    )
            print(f"  Phase {args.phase}: reused existing traces.")
            continue

        # Preserve ordinary resume behavior unless closure provenance is part of the contract.
        if traces_path.exists():
            stale_trace_mismatch: dict[str, object] | None = None
            try:
                traces = json.loads(traces_path.read_text())
                summary = inspect_traces(traces, f"{category}/{problem_name}")
                reuse_decision = _dataset_reuse_decision(
                    rerun=args.rerun,
                    traces_path=traces_path,
                    failed_count=int(summary["failed"]),
                    execution_closure_path=execution_closure_path,
                    provenance=provenance,
                )
            except (OSError, json.JSONDecodeError):
                traces = []
                summary = None
                reuse_decision = None
                stale_trace_mismatch = _stale_provenance_mismatch(
                    observed="unreadable traces"
                )
            if reuse_decision is not None and reuse_decision.should_reuse:
                if run_derived_phase and (
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
                        derived_sidecar_exclusions=derived_sidecar_exclusions,
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
                                amd_score_report=args.amd_score_report
                                if run_derived_phase
                                else None,
                                sol_bound_artifact_dir=args.amd_sol_bound_dir
                                if run_derived_phase
                                else None,
                                solar_derivation_dir=args.solar_derivation
                                if run_derived_phase
                                else None,
                                timing_evidence_dir=args.timing_evidence_dir
                                if run_timing_phase
                                else None,
                            )
                        )
                continue
            if stale_trace_mismatch is not None:
                provenance_mismatches.append(stale_trace_mismatch)
                print("  Re-running (previous trace output is unreadable).")
            elif reuse_decision is not None and reuse_decision.provenance_mismatches:
                provenance_mismatches.extend(list(reuse_decision.provenance_mismatches))
                print("  Re-running (previous output is stale or unreadable).")
            elif (
                reuse_decision is not None
                and reuse_decision.reason == "previous_failed"
            ):
                failed = summary["failed"] if summary is not None else "unknown"
                print(f"  Re-running (previous run had {failed} failures).")
            elif (
                reuse_decision is not None
                and reuse_decision.reason == "rerun_requested"
            ):
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
        elif long_tail_exclusions is not None:
            workload_path, excluded_records, all_excluded = (
                _filter_long_tail_workload_file(
                    workload_path=workload_path,
                    filtered_workload_path=problem_output_dir / "workload.jsonl",
                    category=category,
                    problem_id=problem_id,
                    problem_path=_problem_id_for(problems_dir, problem_dir),
                    long_tail_exclusions=long_tail_exclusions,
                    workload_shard_size=args.workload_shard_size,
                )
            )
            closure_records.extend(excluded_records)
            if all_excluded:
                print("  Skipping (all workloads excluded by --long-tail-exclusions).")
                summaries.append(
                    _skipped_problem_summary(
                        f"{category}/{problem_name}", "long_tail_exclusion"
                    )
                )
                continue

        # Truncate workloads if --max-workloads is set outside ready-subset mode.
        if args.max_workloads is not None and selected_workload_refs is None:
            lines, truncated = _workload_prefix_lines(workload_path, args.max_workloads)
            if truncated:
                truncated_path = problem_output_dir / "workload.jsonl"
                truncated_path.write_text("\n".join(lines[: args.max_workloads]))
                workload_path = truncated_path

        if args.blob_precheck != "off":
            missing_refs = _missing_safetensors_refs(
                workload_path, blob_index=blob_index
            )
            if missing_refs:
                message = (
                    "missing safetensors blobs: "
                    + ", ".join(missing_refs[:3])
                    + (" ..." if len(missing_refs) > 3 else "")
                )
                print(f"  Blob precheck: {message}")
                if args.blob_precheck == "fail":
                    summaries.append(
                        {
                            "problem": f"{category}/{problem_name}",
                            "total": 1,
                            "passed": 0,
                            "failed": 1,
                            "latencies_ms": [],
                            "failure_reasons": [f"  [MISSING_SAFETENSORS] {message}"],
                        }
                    )
                    if selected_workload_refs is not None:
                        for workload_ref in selected_workload_refs:
                            key = _workload_key(
                                workload_ref.get("uuid"),
                                workload_ref.get("row_index"),
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
                                    readiness=readiness_by_workload.get(
                                        (problem_id, key)
                                    ),
                                    summary_ref="summary.json",
                                    solution_ref=None,
                                    notes=[message],
                                )
                            )
                    continue

        # Load definition to build the reference solution
        definition = definition_payload

        # Use named solution file if present, otherwise fall back to reference
        if args.solution_name:
            solution_file = problem_dir / args.solution_name
            if not solution_file.exists():
                print(f"  Skipping: {args.solution_name} not found")
                if selected_workload_refs is not None:
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
                                closure_status="not_attempted",
                                readiness=readiness_by_workload.get((problem_id, key)),
                                filter_reasons=["missing_solution"],
                                notes=[f"{args.solution_name} not found"],
                            )
                        )
                continue
        else:
            if "reference" not in definition or not definition["reference"].strip():
                print("  Skipping: no reference code")
                if selected_workload_refs is not None:
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
                                closure_status="not_attempted",
                                readiness=readiness_by_workload.get((problem_id, key)),
                                filter_reasons=["missing_reference"],
                                notes=["reference code not found"],
                            )
                        )
                continue

        solution = build_solution_for_problem(
            definition, problem_dir, args.solution_name
        )

        solution_path = problem_output_dir / "solution.json"
        solution_path.write_text(json.dumps(solution, indent=2))

        # Call sol-execbench CLI
        job_name = f"ref_{problem_name[:40]}"
        invocation = _run_trace_invocations(
            definition_path=definition_path,
            workload_path=workload_path,
            solution_path=solution_path,
            output_dir=problem_output_dir,
            job_name=job_name,
            definition=definition,
            solution=solution,
            problem_id=problem_id,
            args=args,
            config_path=config_path,
        )
        traces = invocation.traces
        shard_failure_reasons = invocation.shard_failure_reasons
        for message in invocation.messages:
            print(message)

        if not traces and shard_failure_reasons:
            print("  ERROR: CLI returned no traces")
            summaries.append(
                {
                    "problem": f"{category}/{problem_name}",
                    "total": len(shard_failure_reasons),
                    "passed": 0,
                    "failed": len(shard_failure_reasons),
                    "latencies_ms": [],
                    "failure_reasons": shard_failure_reasons,
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
                            notes=shard_failure_reasons,
                        )
                    )
            continue

        # Save raw traces
        traces_path = problem_output_dir / "traces.json"
        traces_path.write_text(json.dumps(traces, indent=2))

        if run_derived_phase and (
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
                derived_sidecar_exclusions=derived_sidecar_exclusions,
            )

        if (
            run_timing_phase
            and args.timing_evidence_dir is not None
            and not shard_failure_reasons
        ):
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
                        skipped=False,
                        traces_path=traces_path,
                        summary_ref="summary.json",
                        solution_path=solution_path,
                        output_dir=output_dir,
                        definition_name=str(definition_payload["name"]),
                        problem_output_dir=problem_output_dir,
                        amd_score_report=args.amd_score_report
                        if run_derived_phase
                        else None,
                        sol_bound_artifact_dir=args.amd_sol_bound_dir
                        if run_derived_phase
                        else None,
                        solar_derivation_dir=args.solar_derivation
                        if run_derived_phase
                        else None,
                        timing_evidence_dir=args.timing_evidence_dir
                        if run_timing_phase
                        else None,
                    )
                )

        # Inspect
        summary = inspect_traces(traces, f"{category}/{problem_name}")
        if shard_failure_reasons:
            summary["total"] += len(shard_failure_reasons)
            summary["failed"] += len(shard_failure_reasons)
            summary["failure_reasons"].extend(shard_failure_reasons)
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

    if run_derived_phase and args.amd_score_report is not None:
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

    if run_derived_phase and args.official_score_report is not None:
        report_path = args.official_score_report.resolve()
        write_official_score_report(
            report_path,
            amd_scores,
            aggregation_policy=args.official_aggregation_policy,
            coverage_report=scoring_baseline_coverage_report(
                scoring_baseline, amd_scores
            ),
            source_score_ref=_relative_ref(args.amd_score_report.resolve(), output_dir),
        )
        print(f"Official score report saved to {report_path}")

    if execution_closure_path is not None:
        if readiness is not None:
            for workload in readiness.get("workloads", []):
                key = _workload_key(
                    workload.get("workload_uuid"), workload.get("row_index")
                )
                problem_id = str(workload.get("problem_id"))
                if (
                    workload.get("status") == "ready"
                    or (
                        problem_id,
                        key,
                    )
                    in attempted_ready_keys
                ):
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
                "workload_shard_size": args.workload_shard_size,
            },
            provenance_mismatches=provenance_mismatches,
            source_refs=source_refs,
        )
        print(f"Execution closure saved to {execution_closure_path}")


if __name__ == "__main__":
    main()
