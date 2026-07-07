"""Importable helpers for dataset-scale SOL ExecBench runs."""

from __future__ import annotations

import ast
import io
import json
import subprocess
import sys
import tempfile
import tokenize
from collections.abc import Sequence
from pathlib import Path

from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.bench.io import flashinfer_safetensors_env
from sol_execbench.core.bench.rocm_profiler import (
    ProfilerRunner,
    collect_source_timing_evidence,
)
from sol_execbench.core.bench.stderr import filter_benign_rocm_stderr
from sol_execbench.core.dataset import amd_score_reports as _amd_score_reports
from sol_execbench.core.dataset.amd_score_reports import (
    _build_amd_score_reports_for_problem_impl,
    write_amd_score_report,
)
from sol_execbench.core.scoring.amd_score import (
    AmdNativeScore,
)
from sol_execbench.core.scoring.amd_sol_v2 import build_amd_sol_bound_v2_artifact
from sol_execbench.core.scoring.baseline_artifact import ScoringBaselineArtifact
from sol_execbench.core.scoring.solar_derivation import (
    build_solar_derivation_evidence,
    solar_derivation_from_dict,
)

# Keep the imported writer as an explicit runner-module re-export for callers.
_write_amd_score_report = write_amd_score_report


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
    """Build the ``sol-execbench`` command used by dataset runs."""
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
    *,
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
    """Invoke ``sol-execbench`` and return parsed trace dicts."""
    cmd = build_cli_command(
        definition_path=definition_path,
        workload_path=workload_path,
        solution_path=solution_path,
        timeout=timeout,
        config_path=config_path,
        keep_staging=keep_staging,
        verbose=verbose,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = _temporary_stream_path(output_dir, job_name, "stdout")
    stderr_path = _temporary_stream_path(output_dir, job_name, "stderr")
    try:
        try:
            with (
                stdout_path.open("w", encoding="utf-8") as stdout_handle,
                stderr_path.open(
                    "w",
                    encoding="utf-8",
                ) as stderr_handle,
            ):
                result = subprocess.run(
                    cmd,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    timeout=timeout + 60,
                    check=False,
                    env=flashinfer_safetensors_env(),
                )
            if result.stdout:
                stdout_path.write_text(result.stdout, encoding="utf-8")
            if result.stderr:
                stderr_path.write_text(result.stderr, encoding="utf-8")
        except subprocess.TimeoutExpired as exc:
            if exc.output:
                stdout_path.write_bytes(
                    exc.output if isinstance(exc.output, bytes) else exc.output.encode()
                )
            if exc.stderr:
                stderr_path.write_bytes(
                    exc.stderr if isinstance(exc.stderr, bytes) else exc.stderr.encode()
                )
            print(f"CLI timed out for {job_name}: {exc.timeout} seconds")
            save_cli_timeout_log_from_files(
                output_dir,
                job_name,
                int(exc.timeout),
                stdout_path,
                stderr_path,
            )
            return None

        traces = _parse_trace_jsonl(stdout_path)

        if result.returncode != 0:
            print(f"CLI failed for {job_name}: exit code {result.returncode}")
            save_cli_log_from_files(
                output_dir,
                job_name,
                result.returncode,
                stdout_path,
                stderr_path,
            )
            if traces:
                return traces
            return None

        if not traces:
            stderr_tail = bounded_file_stream(stderr_path)
            print(f"CLI failed for {job_name}: {stderr_tail[:500]}")
            save_cli_log_from_files(
                output_dir,
                job_name,
                result.returncode,
                stdout_path,
                stderr_path,
            )
            return None

        return traces
    finally:
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)


CLI_LOG_LIMIT = 64 * 1024


def bounded_cli_stream(value: str | bytes | None) -> str:
    """Return a bounded tail representation of captured CLI output."""
    text = filter_benign_rocm_stderr(value)
    if len(text) <= CLI_LOG_LIMIT:
        return text
    return "[truncated CLI output]\n" + text[-CLI_LOG_LIMIT:]


def _temporary_stream_path(output_dir: Path, job_name: str, stream_name: str) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f"{job_name}_{stream_name}_",
        suffix=".log",
        dir=output_dir,
        delete=False,
    ) as handle:
        return Path(handle.name)


def bounded_file_stream(path: Path) -> str:
    """Return a bounded text representation of a captured stream file."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > CLI_LOG_LIMIT:
                handle.seek(max(0, size - CLI_LOG_LIMIT))
                data = handle.read()
                text = filter_benign_rocm_stderr(data)
                return "[truncated CLI output]\n" + text if text else ""
            return filter_benign_rocm_stderr(handle.read())
    except OSError:
        return ""


def _parse_trace_jsonl(stdout_path: Path) -> list[dict]:
    traces: list[dict] = []
    try:
        with stdout_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return traces


def save_cli_log(output_dir: Path, job_name: str, result: subprocess.CompletedProcess):
    """Write stdout/stderr from a failed CLI invocation to a log file."""
    log_path = output_dir / f"{job_name}_cli.log"
    parts = [
        f"exit code: {result.returncode}",
        f"\n--- stdout ---\n{bounded_cli_stream(result.stdout)}"
        if result.stdout
        else "",
        f"\n--- stderr ---\n{bounded_cli_stream(result.stderr)}"
        if result.stderr
        else "",
    ]
    log_path.write_text("\n".join(parts))
    print(f"Saved CLI log to {log_path}")


def save_cli_log_from_files(
    output_dir: Path,
    job_name: str,
    returncode: int | None,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    """Write bounded stdout/stderr tails from captured stream files."""
    log_path = output_dir / f"{job_name}_cli.log"
    stdout = bounded_file_stream(stdout_path)
    stderr = bounded_file_stream(stderr_path)
    parts = [
        f"exit code: {returncode}",
        f"\n--- stdout ---\n{stdout}" if stdout else "",
        f"\n--- stderr ---\n{stderr}" if stderr else "",
    ]
    log_path.write_text("\n".join(parts))
    print(f"Saved CLI log to {log_path}")


def save_cli_timeout_log(
    output_dir: Path,
    job_name: str,
    exc: subprocess.TimeoutExpired,
) -> None:
    """Write stdout/stderr from a timed-out CLI invocation to a log file."""
    log_path = output_dir / f"{job_name}_cli.log"
    parts = [
        f"timeout after {exc.timeout} seconds",
        f"\n--- stdout ---\n{bounded_cli_stream(exc.output)}" if exc.output else "",
        f"\n--- stderr ---\n{bounded_cli_stream(exc.stderr)}" if exc.stderr else "",
    ]
    log_path.write_text("\n".join(parts))
    print(f"Saved CLI log to {log_path}")


def save_cli_timeout_log_from_files(
    output_dir: Path,
    job_name: str,
    timeout: int,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    """Write bounded stdout/stderr tails after a timed-out CLI invocation."""
    log_path = output_dir / f"{job_name}_cli.log"
    stdout = bounded_file_stream(stdout_path)
    stderr = bounded_file_stream(stderr_path)
    parts = [
        f"timeout after {timeout} seconds",
        f"\n--- stdout ---\n{stdout}" if stdout else "",
        f"\n--- stderr ---\n{stderr}" if stderr else "",
    ]
    log_path.write_text("\n".join(parts))
    print(f"Saved CLI log to {log_path}")


def cli_failure_notes(cli_log: Path) -> list[str]:
    """Return human-readable failure notes from a saved CLI log."""
    if not cli_log.exists():
        return ["CLI returned no traces"]
    try:
        text = cli_log.read_text(errors="replace")
    except OSError:
        return ["CLI returned no traces"]
    lines = text.splitlines()
    message = lines[0].strip() if lines else ""
    if not message:
        return ["CLI returned no traces"]
    timeout_marker = "subprocess.TimeoutExpired:"
    if timeout_marker in text:
        timeout_line = next(
            (line.strip() for line in lines if timeout_marker in line),
            "",
        )
        if "timed out after " in timeout_line:
            timeout_value = timeout_line.rsplit("timed out after ", maxsplit=1)[1]
            return [f"CLI timed out after {timeout_value}"]
        return ["CLI timed out"]
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


_bounded_cli_stream = bounded_cli_stream
_save_cli_log = save_cli_log
_save_cli_timeout_log = save_cli_timeout_log
_cli_failure_notes = cli_failure_notes


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
    _amd_score_reports.build_amd_sol_bound_v2_artifact = build_amd_sol_bound_v2_artifact
    _amd_score_reports.build_solar_derivation_evidence = build_solar_derivation_evidence
    _amd_score_reports.solar_derivation_from_dict = solar_derivation_from_dict
    return _build_amd_score_reports_for_problem_impl(
        definition_payload=definition_payload,
        workload_path=workload_path,
        traces_payload=traces_payload,
        trace_ref=trace_ref,
        run_cli_func=run_cli,
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
