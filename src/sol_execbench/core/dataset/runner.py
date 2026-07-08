"""Importable helpers for dataset-scale SOL ExecBench runs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.bench.rocm_profiler import (
    ProfilerRunner,
    collect_source_timing_evidence,
)
import sol_execbench.core.dataset.cli_execution as cli_execution


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
    command = cli_execution.build_cli_command(
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


def inspect_traces(traces: list[dict], problem_name: str) -> dict:
    """Inspect traces for correctness and return a summary dict."""
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


def print_summary(summaries: list[dict]) -> None:
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
    skipped = 0

    for summary in summaries:
        name = summary["problem"]
        pass_count = summary["passed"]
        fail_count = summary["failed"]

        if summary.get("skipped", 0):
            status = "SKIP"
            skipped += 1
        elif fail_count == 0:
            status = "OK"
            all_passed += 1
        else:
            status = "FAIL"
            any_failed += 1

        print(f"{name:<{name_width}}  {pass_count:>5} {fail_count:>5} {status:>8}")

    print("=" * row_width)
    print(
        f"Total: {total_problems} problems | OK: {all_passed} | "
        f"FAIL: {any_failed} | SKIP: {skipped}"
    )


def write_summary_report(output_dir: Path, summaries: list[dict]) -> Path:
    """Write dataset summary JSON and return its path."""
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    return summary_path
