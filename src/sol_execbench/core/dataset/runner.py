"""Importable helpers for dataset-scale SOL ExecBench runs."""

from __future__ import annotations

import ast
import io
import json
import tokenize
from collections.abc import Sequence
from pathlib import Path

from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.bench.rocm_profiler import (
    ProfilerRunner,
    collect_source_timing_evidence,
)
from sol_execbench.core.dataset import cli_execution
from sol_execbench.core.dataset.amd_score_reports import (
    _build_amd_score_reports_for_problem_impl,
    write_amd_score_report as write_amd_score_report,
)
from sol_execbench.core.scoring.amd_score import (
    AmdNativeScore,
)
from sol_execbench.core.scoring.baseline_artifact import ScoringBaselineArtifact


def infer_destination_passing_style(code: str, definition: dict) -> bool:
    """Infer destination-passing style by checking the ``run()`` signature."""
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
            if args.args:
                last_param = args.args[-1].arg
                return last_param == last_output
            break

    return False


def sanitize_python_source_for_static_review(code: str) -> str:
    """Rename exact ``stream`` identifiers without mutating comments or strings."""
    tokens: list[tokenize.TokenInfo] = []
    reader = io.StringIO(code).readline
    for token in tokenize.generate_tokens(reader):
        if token.type == tokenize.NAME and token.string == "stream":
            token = tokenize.TokenInfo(
                token.type,
                "strm",
                token.start,
                token.end,
                token.line,
            )
        tokens.append(token)
    return tokenize.untokenize(tokens)


def build_solution_for_problem(
    definition: dict, problem_dir: Path, solution_name: str | None = None
) -> dict:
    """Build a reference or custom solution for a problem directory."""
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
    dps = infer_destination_passing_style(code, definition)
    code = sanitize_python_source_for_static_review(code)

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
    dps = infer_destination_passing_style(reference_code, definition)
    reference_code = sanitize_python_source_for_static_review(reference_code)

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


def build_amd_score_reports_for_problem(
    *,
    definition_payload: dict,
    workload_path: Path,
    traces_payload: list[dict],
    trace_ref: str,
    baseline_artifact: ScoringBaselineArtifact | None = None,
    sol_bound_artifact_dir: Path | None = None,
    solar_derivation_dir: Path | None = None,
    sidecar_namespace: str | None = None,
    derived_sidecar_exclusions: dict[str, str] | None = None,
) -> list[AmdNativeScore]:
    """Build derived AMD-native scores for one dataset-run problem."""
    return _build_amd_score_reports_for_problem_impl(
        definition_payload=definition_payload,
        workload_path=workload_path,
        traces_payload=traces_payload,
        trace_ref=trace_ref,
        run_cli_func=cli_execution.run_cli,
        baseline_artifact=baseline_artifact,
        sol_bound_artifact_dir=sol_bound_artifact_dir,
        solar_derivation_dir=solar_derivation_dir,
        sidecar_namespace=sidecar_namespace,
        derived_sidecar_exclusions=derived_sidecar_exclusions,
    )


def extend_derived_reports_for_problem(
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
    derived_sidecar_exclusions: dict[str, str] | None = None,
) -> None:
    """Append requested derived reports and materialize requested sidecars."""
    trace_ref = (
        str(traces_path.relative_to(output_dir))
        if traces_path.is_relative_to(output_dir)
        else str(traces_path)
    )
    sidecar_namespace = str(Path(trace_ref).parent)
    if sidecar_namespace == ".":
        sidecar_namespace = None
    amd_scores.extend(
        build_amd_score_reports_for_problem(
            definition_payload=json.loads(definition_path.read_text()),
            workload_path=workload_path,
            traces_payload=traces_payload,
            trace_ref=trace_ref,
            baseline_artifact=baseline_artifact,
            sol_bound_artifact_dir=sol_bound_artifact_dir,
            solar_derivation_dir=solar_derivation_dir,
            sidecar_namespace=sidecar_namespace,
            derived_sidecar_exclusions=derived_sidecar_exclusions,
        )
    )


_extend_derived_reports_for_problem = extend_derived_reports_for_problem
