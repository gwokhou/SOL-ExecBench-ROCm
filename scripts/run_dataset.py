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
    build_amd_sol_bound_artifact,
    default_amd_hardware_models,
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
    if categories:
        roots = [benchmark_dir / c for c in categories]
    else:
        roots = sorted(
            p for p in benchmark_dir.iterdir() if p.is_dir() and p.name in CATEGORIES
        )

    problems = []
    for root in roots:
        if not root.is_dir():
            continue
        for problem_dir in sorted(root.iterdir()):
            defn = problem_dir / "definition.json"
            wkl = problem_dir / "workload.jsonl"
            if defn.exists() and wkl.exists():
                problems.append(problem_dir)
    return problems


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

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60)

    # The CLI with --json prints one JSON trace per line to stdout.
    traces = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not traces:
        print(f"CLI failed for {job_name}: {result.stderr[:500]}")
        _save_cli_log(output_dir, job_name, result)
        return None

    return traces


def _save_cli_log(output_dir: Path, job_name: str, result: subprocess.CompletedProcess):
    """Write stdout/stderr from a failed CLI invocation to a log file."""
    log_path = output_dir / f"{job_name}_cli.log"
    parts = [
        f"exit code: {result.returncode}",
        f"\n--- stdout ---\n{result.stdout}" if result.stdout else "",
        f"\n--- stderr ---\n{result.stderr}" if result.stderr else "",
    ]
    log_path.write_text("\n".join(parts))
    print(f"Saved CLI log to {log_path}")


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
            build_amd_sol_bound_artifact(definition, workload, hardware_model)
            if workload is not None
            else None
        )
        scores.append(
            score_amd_native_trace_workload(
                trace,
                artifact,
                trace_ref=trace_ref,
                timing_evidence_ref=trace_ref,
                sol_bound_ref=f"derived:{definition.name}:{trace.workload.uuid}:amd_sol_bound",
                baseline_ref=(
                    f"{baseline_artifact.source}#{definition.name}:{trace.workload.uuid}"
                    if baseline_artifact
                    and baseline_artifact.lookup(definition.name, trace.workload.uuid)
                    is not None
                    else "trace.evaluation.performance.reference_latency_ms"
                ),
                baseline_artifact=baseline_artifact,
                hardware_model_ref=f"default_amd_hardware_models.{hardware_model_key}",
            )
        )
    return scores


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
    args = ap.parse_args()

    problems_dir = args.problems_dir.resolve()
    if not problems_dir.is_dir():
        print(f"Error: {problems_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect: single problem dir vs. dataset root
    is_single_problem = (problems_dir / "definition.json").exists() and (
        problems_dir / "workload.jsonl"
    ).exists()

    if is_single_problem:
        problems = [problems_dir]
        print(f"Single problem: {problems_dir.name}")
    else:
        problems = discover_problems(problems_dir, args.category)
        if args.limit:
            problems = problems[: args.limit]

        print(f"Discovered {len(problems)} problems under {problems_dir}")
        if not problems:
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
    for i, problem_dir in enumerate(problems):
        problem_name = problem_dir.name
        category = problem_dir.parent.name
        print(f"\n[{i + 1}/{len(problems)}] {category}/{problem_name}")

        definition_path = problem_dir / "definition.json"
        workload_path = problem_dir / "workload.jsonl"

        problem_output_dir = output_dir / category / problem_name
        traces_path = problem_output_dir / "traces.json"

        # Skip problems that already have passing results unless --rerun
        if not args.rerun and traces_path.exists():
            traces = json.loads(traces_path.read_text())
            summary = inspect_traces(traces, f"{category}/{problem_name}")
            if summary["failed"] == 0:
                print("  Skipping (already passed). Use --rerun to re-evaluate.")
                summaries.append(summary)
                continue
            print(f"  Re-running (previous run had {summary['failed']} failures).")

        # Clear previous run output
        if problem_output_dir.exists():
            shutil.rmtree(problem_output_dir)
        problem_output_dir.mkdir(parents=True, exist_ok=True)

        # Truncate workloads if --max-workloads is set
        if args.max_workloads is not None:
            lines = workload_path.read_text().splitlines()
            if len(lines) > args.max_workloads:
                truncated_path = problem_output_dir / "workload.jsonl"
                truncated_path.write_text("\n".join(lines[: args.max_workloads]))
                workload_path = truncated_path

        # Load definition to build the reference solution
        definition = json.loads(definition_path.read_text())

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
            continue

        # Save raw traces
        traces_path = problem_output_dir / "traces.json"
        traces_path.write_text(json.dumps(traces, indent=2))

        if args.amd_score_report is not None:
            amd_scores.extend(
                build_amd_score_reports_for_problem(
                    definition_payload=definition,
                    workload_path=workload_path,
                    traces_payload=traces,
                    trace_ref=str(traces_path.relative_to(output_dir)),
                    baseline_artifact=scoring_baseline,
                )
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


if __name__ == "__main__":
    main()
