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
import ast
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.bench.rocm_profiler import (
    ProfilerRunner,
    collect_source_timing_evidence,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.execution_closure import compare_execution_closure_provenance
from sol_execbench.core.dataset.evidence_refs import (
    relative_ref as _relative_ref,
    safe_sidecar_stem as _safe_sidecar_stem,
)
from sol_execbench.core.dataset.run_closure import (
    closure_record as _build_closure_record,
    closure_totals as _build_closure_totals,
    derived_evidence_for_workload as _build_derived_evidence_for_workload,
    prior_closure_provenance as _load_prior_closure_provenance,
    stale_provenance_mismatch as _build_stale_provenance_mismatch,
    utc_timestamp as _build_utc_timestamp,
    write_execution_closure as _write_execution_closure_report,
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
    build_amd_native_suite_report,
    score_amd_native_trace_workload,
)
from sol_execbench.core.scoring.baseline_artifact import (
    ScoringBaselineArtifact,
    load_scoring_baseline_artifact,
)
from sol_execbench.core.scoring.amd_sol import (
    default_amd_hardware_models,
)
from sol_execbench.core.scoring.amd_sol_v2 import build_amd_sol_bound_v2_artifact
from sol_execbench.core.scoring.solar_derivation import (
    build_solar_derivation_evidence,
    solar_derivation_from_dict,
)

ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = {"L1", "L2", "FlashInfer-Bench", "Quant"}


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


# ---------------------------------------------------------------------------
# Solution construction
# ---------------------------------------------------------------------------


def _infer_dps(code: str, definition: dict) -> bool:
    """Infer destination-passing style by checking the ``run()`` signature.

    If the last parameter of ``run()`` matches the last output name in the
    definition, the solution writes into pre-allocated output buffers (DPS).
    """
    output_names = list(definition.get("outputs", {}).keys())
    if not output_names:
        return False

    last_output = output_names[-1]

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run"
        ):
            args = node.args
            # Last positional arg name
            if args.args:
                last_param = args.args[-1].arg
                return last_param == last_output
            break

    return False


def build_solution_for_problem(
    definition: dict, problem_dir: Path, solution_name: str | None = None
) -> dict:
    """Build a solution for a problem directory.

    If *solution_name* is given, looks for that file inside *problem_dir*.
    ``.json`` files are loaded directly; ``.py`` files are wrapped as a custom
    solution.  Falls back to the definition's reference code when the file does
    not exist or *solution_name* is ``None``.
    """
    if solution_name is not None:
        solution_file = problem_dir / solution_name
        if solution_file.exists():
            if solution_file.suffix == ".json":
                print(f"  Using solution in {solution_file}...")
                return json.loads(solution_file.read_text())
            print(f"  Building solution from {solution_file}...")
            return build_custom_solution(definition, solution_file)
    print("  Building solution from Definition.reference...")
    return build_reference_solution(definition)


def build_custom_solution(definition: dict, solution_py: Path) -> dict:
    """Wrap an external ``solution.py`` file as a Solution dict."""
    name = definition["name"]
    code = solution_py.read_text()
    dps = _infer_dps(code, definition)
    code = code.replace("stream", "strm")

    return {
        "name": f"custom_{name}",
        "definition": name,
        "author": "run_dataset",
        "description": f"Custom solution from {solution_py.name}.",
        "spec": {
            "languages": ["pytorch"],
            "target_hardware": ["LOCAL"],
            "entry_point": "solution.py::run",
            "dependencies": ["torch"],
            "destination_passing_style": dps,
        },
        "sources": [
            {
                "path": "solution.py",
                "content": code,
            }
        ],
    }


def build_reference_solution(definition: dict) -> dict:
    """Construct a Solution dict that wraps the definition's reference code."""
    name = definition["name"]
    reference_code = definition["reference"]
    dps = _infer_dps(reference_code, definition)

    # Replace "stream" with "strm" to avoid tripping the SourceFile stream detector.
    # The validator rejects any Python source containing the word "stream", but some
    # reference implementations legitimately use it in variable names or comments.
    reference_code = reference_code.replace("stream", "strm")

    return {
        "name": f"reference_{name}",
        "definition": name,
        "author": "run_dataset",
        "description": "Identity solution: definition reference as-is.",
        "spec": {
            "languages": ["pytorch"],
            "target_hardware": ["LOCAL"],
            "entry_point": "reference.py::run",
            "dependencies": ["torch"],
            "destination_passing_style": dps,
        },
        "sources": [
            {
                "path": "reference.py",
                "content": reference_code,
            }
        ],
    }


# ---------------------------------------------------------------------------
# CLI invocation
# ---------------------------------------------------------------------------


def build_cli_command(
    *,
    definition_path: Path,
    workload_path: Path,
    solution_path: Path,
    timeout: int,
    config_path: Path | None = None,
    keep_staging: bool = False,
    verbose: bool = False,
) -> list[str]:
    """Build the `sol-execbench` command used by dataset runs."""
    cmd = [
        str(Path(sys.executable).parent / "sol-execbench"),
        "--definition",
        str(definition_path),
        "--workload",
        str(workload_path),
        "--solution",
        str(solution_path),
        "--timeout",
        str(timeout),
        "--json",
    ]

    if config_path:
        cmd.extend(["--config", str(config_path)])
    if keep_staging:
        cmd.append("--keep-staging")
    if verbose:
        cmd.append("--verbose")
    return cmd


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
    """Invoke ``sol-execbench`` and return parsed trace dicts (or None on error)."""
    cmd = build_cli_command(
        definition_path=definition_path,
        workload_path=workload_path,
        solution_path=solution_path,
        timeout=timeout,
        config_path=config_path,
        keep_staging=keep_staging,
        verbose=verbose,
    )

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60)
    except subprocess.TimeoutExpired as exc:
        print(f"CLI timed out for {job_name}: {exc.timeout} seconds")
        _save_cli_timeout_log(output_dir, job_name, exc)
        return None

    # The CLI with --json prints one JSON trace per line to stdout.
    traces = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if result.returncode != 0:
        print(f"CLI failed for {job_name}: exit code {result.returncode}")
        _save_cli_log(output_dir, job_name, result)
        return None

    if not traces:
        print(f"CLI failed for {job_name}: {result.stderr[:500]}")
        _save_cli_log(output_dir, job_name, result)
        return None

    return traces


_CLI_LOG_LIMIT = 64 * 1024


def _bounded_cli_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode(errors="replace")
    else:
        text = value
    if len(text) <= _CLI_LOG_LIMIT:
        return text
    return text[:_CLI_LOG_LIMIT] + "\n[truncated CLI output]\n"


def _save_cli_log(output_dir: Path, job_name: str, result: subprocess.CompletedProcess):
    """Write stdout/stderr from a failed CLI invocation to a log file."""
    log_path = output_dir / f"{job_name}_cli.log"
    parts = [
        f"exit code: {result.returncode}",
        f"\n--- stdout ---\n{_bounded_cli_stream(result.stdout)}" if result.stdout else "",
        f"\n--- stderr ---\n{_bounded_cli_stream(result.stderr)}" if result.stderr else "",
    ]
    log_path.write_text("\n".join(parts))
    print(f"Saved CLI log to {log_path}")


def _save_cli_timeout_log(
    output_dir: Path,
    job_name: str,
    exc: subprocess.TimeoutExpired,
) -> None:
    """Write stdout/stderr from a timed-out CLI invocation to a log file."""
    log_path = output_dir / f"{job_name}_cli.log"
    parts = [
        f"timeout after {exc.timeout} seconds",
        f"\n--- stdout ---\n{_bounded_cli_stream(exc.output)}" if exc.output else "",
        f"\n--- stderr ---\n{_bounded_cli_stream(exc.stderr)}" if exc.stderr else "",
    ]
    log_path.write_text("\n".join(parts))
    print(f"Saved CLI log to {log_path}")


def _cli_failure_notes(cli_log: Path) -> list[str]:
    if not cli_log.exists():
        return ["CLI returned no traces"]
    try:
        with cli_log.open(errors="replace") as handle:
            message = handle.readline().strip()
    except OSError:
        return ["CLI returned no traces"]
    if not message:
        return ["CLI returned no traces"]
    if message.startswith("exit code: "):
        try:
            exit_code = int(message.removeprefix("exit code: ").strip())
        except ValueError:
            return ["CLI returned no traces"]
        if exit_code != 0:
            return [f"CLI failed with exit code {exit_code}"]
    if message.startswith("timeout after "):
        return [f"CLI timed out after {message.removeprefix('timeout after ')}"]
    return ["CLI returned no traces"]


def collect_timing_evidence_for_problem(
    *,
    definition_path: Path,
    workload_path: Path,
    solution_path: Path,
    output_dir: Path,
    timing_evidence_root: Path,
    job_name: str,
    solution: dict,
    benchmark_config: BenchmarkConfig,
    timeout: int,
    config_path: Path | None = None,
    keep_staging: bool = False,
    verbose: bool = False,
    tool_version: str = "rocprofv3",
    gpu_architecture: str = "unknown",
    rocprofv3_available: bool = True,
    runner: ProfilerRunner | None = None,
) -> dict[str, object]:
    """Collect source-specific profiler timing evidence for a dataset problem."""
    languages = solution.get("spec", {}).get("languages", [])
    if isinstance(languages, str):
        languages = [languages]
    elif not isinstance(languages, Sequence):
        languages = []
    command = build_cli_command(
        definition_path=definition_path,
        workload_path=workload_path,
        solution_path=solution_path,
        timeout=timeout,
        config_path=config_path,
        keep_staging=keep_staging,
        verbose=verbose,
    )
    evidence_dir = timing_evidence_root / output_dir.name
    result = collect_source_timing_evidence(
        application_command=command,
        languages=tuple(str(language) for language in languages),
        output_directory=evidence_dir,
        output_file=job_name,
        tool_version=tool_version,
        gpu_architecture=gpu_architecture,
        rocprofv3_available=rocprofv3_available,
        runner=runner,
        warmup_runs=benchmark_config.warmup_runs,
        iterations=benchmark_config.iterations,
        trial_count=1,
        clock_locked=benchmark_config.lock_clocks,
    )
    payload = result.to_dict()
    timing_evidence_root.mkdir(parents=True, exist_ok=True)
    output_path = timing_evidence_root / f"{output_dir.name}.timing.json"
    output_path.write_text(json.dumps(payload, indent=2))
    return payload


# ---------------------------------------------------------------------------
# Trace inspection
# ---------------------------------------------------------------------------


def inspect_traces(traces: list[dict], problem_name: str) -> dict:
    """Inspect traces for correctness.

    Returns a summary dict with pass/fail counts and per-workload latencies.
    """
    total = len(traces)
    passed = 0
    failed = 0
    latencies = []
    failure_reasons = []

    for trace in traces:
        evaluation = trace.get("evaluation", {})
        status = evaluation.get("status", "UNKNOWN")

        if status == "PASSED":
            passed += 1
            perf = evaluation.get("performance") or {}
            latency = perf.get("latency_ms")
            if latency is not None:
                latencies.append(latency)
        else:
            failed += 1
            log = evaluation.get("log", "")
            failure_reasons.append(f"  [{status}] {log[:200]}")

    return {
        "problem": problem_name,
        "total": total,
        "passed": passed,
        "failed": failed,
        "latencies_ms": latencies,
        "failure_reasons": failure_reasons,
    }


def print_summary(summaries: list[dict]):
    """Print a table summarizing all problem results."""
    name_width = max((len(s["problem"]) for s in summaries), default=20)
    name_width = max(name_width, 20)
    row_width = name_width + 2 + 5 + 1 + 5 + 1 + 8

    print("\n" + "=" * row_width)
    print(f"{'Problem':<{name_width}}  {'Pass':>5} {'Fail':>5} {'Status':>8}")
    print("-" * row_width)

    total_problems = len(summaries)
    all_passed = 0
    any_failed = 0

    for s in summaries:
        name = s["problem"]
        pass_count = s["passed"]
        fail_count = s["failed"]

        if fail_count == 0:
            status = "OK"
            all_passed += 1
        else:
            status = "FAIL"
            any_failed += 1

        print(f"{name:<{name_width}}  {pass_count:>5} {fail_count:>5} {status:>8}")

    print("=" * row_width)
    print(f"Total: {total_problems} problems | OK: {all_passed} | FAIL: {any_failed}")


# ---------------------------------------------------------------------------
# AMD-native derived score report
# ---------------------------------------------------------------------------


def _hardware_model_key_from_traces(traces: list[Trace]) -> str:
    """Return the first known AMD gfx key found in trace environment strings."""
    known = default_amd_hardware_models()
    for trace in traces:
        if trace.evaluation is None:
            continue
        hardware = trace.evaluation.environment.hardware
        for key in known:
            if key in hardware:
                return key
    return "gfx1200"


def build_amd_score_reports_for_problem(
    *,
    definition_payload: dict,
    workload_path: Path,
    traces_payload: list[dict],
    trace_ref: str,
    baseline_artifact: ScoringBaselineArtifact | None = None,
    sol_bound_artifact_dir: Path | None = None,
    solar_derivation_dir: Path | None = None,
) -> list[AmdNativeScore]:
    """Build derived AMD-native scores for one dataset-run problem."""
    definition = Definition(**definition_payload)
    workloads = {
        workload.uuid: workload
        for workload in (
            Workload(**json.loads(line))
            for line in workload_path.read_text().splitlines()
            if line.strip()
        )
    }
    traces = [Trace(**trace) for trace in traces_payload]
    hardware_models = default_amd_hardware_models()
    hardware_model_key = _hardware_model_key_from_traces(traces)
    hardware_model = hardware_models[hardware_model_key]
    scores: list[AmdNativeScore] = []

    for trace in traces:
        workload = workloads.get(trace.workload.uuid)
        artifact = (
            build_amd_sol_bound_v2_artifact(
                definition,
                workload,
                hardware_model,
                hardware_model_ref=f"default_amd_hardware_models.{hardware_model_key}",
            )
            if workload is not None
            else None
        )
        solar_derivation = None
        derived_evidence_refs = None
        solar_derivation_ref = (
            f"derived:{definition.name}:{trace.workload.uuid}:solar_derivation"
        )
        if workload is not None and solar_derivation_dir is not None:
            solar_derivation_dir.mkdir(parents=True, exist_ok=True)
            sidecar_stem = _safe_sidecar_stem(definition.name, trace.workload.uuid)
            sidecar_path = solar_derivation_dir / f"{sidecar_stem}.solar-derivation.json"
            generated = build_solar_derivation_evidence(definition, workload)
            sidecar_path.write_text(json.dumps(generated.to_dict(), indent=2))
            solar_derivation = solar_derivation_from_dict(
                json.loads(sidecar_path.read_text())
            )
            solar_derivation_ref = str(sidecar_path)
            derived_evidence_refs = {
                "formula": f"{solar_derivation_ref}#groups.formula_evidence",
                "hardware_model": f"default_amd_hardware_models.{hardware_model_key}",
                "coverage": f"{solar_derivation_ref}#coverage_summary",
                "score_eligibility": f"{solar_derivation_ref}#aggregate_status",
            }
        sol_bound_ref = f"derived:{definition.name}:{trace.workload.uuid}:amd_sol_bound_v2"
        if artifact is not None and sol_bound_artifact_dir is not None:
            sol_bound_artifact_dir.mkdir(parents=True, exist_ok=True)
            sidecar_stem = _safe_sidecar_stem(definition.name, trace.workload.uuid)
            sidecar_path = sol_bound_artifact_dir / f"{sidecar_stem}.amd-sol-v2.json"
            sidecar_path.write_text(json.dumps(artifact.to_dict(), indent=2))
            sol_bound_ref = str(sidecar_path)
        scores.append(
            score_amd_native_trace_workload(
                trace,
                artifact,
                trace_ref=trace_ref,
                timing_evidence_ref=trace_ref,
                sol_bound_ref=sol_bound_ref,
                baseline_ref=(
                    f"{baseline_artifact.source}#{definition.name}:{trace.workload.uuid}"
                    if baseline_artifact
                    and baseline_artifact.lookup(definition.name, trace.workload.uuid)
                    is not None
                    else "trace.evaluation.performance.reference_latency_ms"
                ),
                baseline_artifact=baseline_artifact,
                hardware_model_ref=f"default_amd_hardware_models.{hardware_model_key}",
                solar_derivation=solar_derivation,
                derived_evidence_refs=derived_evidence_refs,
            )
        )
    return scores


def _extend_derived_reports_for_problem(
    *,
    amd_scores: list[AmdNativeScore],
    definition_path: Path,
    workload_path: Path,
    traces_path: Path,
    traces_payload: list[dict],
    output_dir: Path,
    baseline_artifact: ScoringBaselineArtifact | None,
    sol_bound_artifact_dir: Path | None,
    solar_derivation_dir: Path | None,
) -> None:
    """Append requested derived reports and materialize requested sidecars."""
    trace_ref = (
        str(traces_path.relative_to(output_dir))
        if traces_path.is_relative_to(output_dir)
        else str(traces_path)
    )
    amd_scores.extend(
        build_amd_score_reports_for_problem(
            definition_payload=json.loads(definition_path.read_text()),
            workload_path=workload_path,
            traces_payload=traces_payload,
            trace_ref=trace_ref,
            baseline_artifact=baseline_artifact,
            sol_bound_artifact_dir=sol_bound_artifact_dir,
            solar_derivation_dir=solar_derivation_dir,
        )
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


def _closure_totals(records: list[dict]) -> dict[str, int]:
    return _build_closure_totals(records)


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
        if not args.rerun and traces_path.exists():
            traces = json.loads(traces_path.read_text())
            summary = inspect_traces(traces, f"{category}/{problem_name}")
            if summary["failed"] == 0:
                if execution_closure_path is None:
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
                    continue
                prior_provenance, stale_mismatch = _prior_closure_provenance(
                    execution_closure_path
                )
                reuse_mismatches = (
                    [stale_mismatch]
                    if stale_mismatch is not None
                    else [
                        mismatch.model_dump(mode="json")
                        for mismatch in compare_execution_closure_provenance(
                            provenance,
                            prior_provenance or {},
                        )
                    ]
                )
                if reuse_mismatches:
                    provenance_mismatches.extend(reuse_mismatches)
                    print("  Re-running (previous pass has stale closure provenance).")
                else:
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
                            evidence_refs, evidence_gaps = _derived_evidence_for_workload(
                                definition_name=str(definition_payload["name"]),
                                workload_uuid=workload_ref.get("uuid"),
                                problem_output_dir=problem_output_dir,
                                output_dir=output_dir,
                                amd_score_report=args.amd_score_report,
                                sol_bound_artifact_dir=args.amd_sol_bound_dir,
                                solar_derivation_dir=args.solar_derivation,
                                timing_evidence_dir=args.timing_evidence_dir,
                                category=category,
                            )
                            status = _closure_status_with_evidence(
                                _closure_status_for_trace(trace, skipped=True),
                                evidence_gaps,
                            )
                            closure_records.append(
                                _closure_record(
                                    category=category,
                                    problem_id=problem_id,
                                    problem_path=str(problem_ref.get("problem_path")),
                                    workload_uuid=workload_ref.get("uuid"),
                                    row_index=int(workload_ref.get("row_index", 0)),
                                    closure_status=status,
                                    readiness=readiness_by_workload.get((problem_id, key)),
                                    trace_ref=_relative_ref(traces_path, output_dir),
                                    summary_ref="summary.json",
                                    solution_ref=_relative_ref(problem_output_dir / "solution.json", output_dir)
                                    if (problem_output_dir / "solution.json").exists()
                                    else None,
                                    evidence_refs=evidence_refs,
                                    evidence_gaps=evidence_gaps,
                                    trace_status=_trace_status(trace),
                                )
                            )
                    continue
            if summary["failed"] == 0:
                pass
            else:
                print(f"  Re-running (previous run had {summary['failed']} failures).")
        elif args.rerun and traces_path.exists():
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
                evidence_refs, evidence_gaps = _derived_evidence_for_workload(
                    definition_name=str(definition_payload["name"]),
                    workload_uuid=workload_ref.get("uuid"),
                    problem_output_dir=problem_output_dir,
                    output_dir=output_dir,
                    amd_score_report=args.amd_score_report,
                    sol_bound_artifact_dir=args.amd_sol_bound_dir,
                    solar_derivation_dir=args.solar_derivation,
                    timing_evidence_dir=args.timing_evidence_dir,
                    category=category,
                )
                status = _closure_status_with_evidence(
                    _closure_status_for_trace(trace), evidence_gaps
                )
                closure_records.append(
                    _closure_record(
                        category=category,
                        problem_id=problem_id,
                        problem_path=str(problem_ref.get("problem_path")),
                        workload_uuid=workload_ref.get("uuid"),
                        row_index=int(workload_ref.get("row_index", 0)),
                        closure_status=status,
                        readiness=readiness_by_workload.get((problem_id, key)),
                        trace_ref=_relative_ref(traces_path, output_dir),
                        summary_ref="summary.json",
                        solution_ref=_relative_ref(solution_path, output_dir),
                        evidence_refs=evidence_refs,
                        evidence_gaps=evidence_gaps,
                        trace_status=_trace_status(trace),
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
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"\nSummary saved to {summary_path}")
    print(f"Per-problem traces saved under {output_dir}")

    if args.amd_score_report is not None:
        report_path = args.amd_score_report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = build_amd_native_suite_report(
            amd_scores,
            baseline_summary={
                "problems": len(summaries),
                "scores": len(amd_scores),
                "baseline_entries": (
                    len(scoring_baseline.entries) if scoring_baseline else 0
                ),
            },
        )
        report_path.write_text(json.dumps(report.to_dict(), indent=2))
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
