"""Importable helpers for dataset-scale SOL ExecBench runs."""

from __future__ import annotations

import ast
import gc
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
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.evidence_refs import sidecar_stem_for_workload
from sol_execbench.core.scoring.amd_score import (
    AmdNativeScore,
    build_amd_native_suite_report,
    score_amd_native_trace_workload,
)
from sol_execbench.core.scoring.amd_sol import default_amd_hardware_models
from sol_execbench.core.scoring.amd_sol_v2 import (
    AMD_SOL_V2_SCHEMA_VERSION,
    AmdSolBoundV2Artifact,
    AmdSolV2AggregateBound,
    AmdSolV2CoverageSummary,
    build_amd_sol_bound_v2_artifact,
)
from sol_execbench.core.scoring.amd_hardware_models import (
    EstimateConfidence,
    amd_hardware_model_from_dict,
)
from sol_execbench.core.scoring.baseline_artifact import ScoringBaselineArtifact
from sol_execbench.core.scoring.solar_derivation import (
    SolarAggregateStatus,
    build_solar_derivation_evidence,
    solar_derivation_from_dict,
)


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


def _hardware_model_key_from_traces(traces: Sequence[Trace]) -> str:
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


def _hardware_model_key_from_trace_payloads(traces_payload: Sequence[dict]) -> str:
    """Return the first known AMD gfx key without retaining parsed traces."""
    known = default_amd_hardware_models()
    for payload in traces_payload:
        try:
            hardware = str(payload["evaluation"]["environment"].get("hardware", ""))
        except (KeyError, TypeError, AttributeError):
            continue
        for key in known:
            if key in hardware:
                return key
    return "gfx1200"


def _read_json_object(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _minimal_amd_sol_bound_v2_from_payload(
    payload: dict,
) -> AmdSolBoundV2Artifact | None:
    """Parse only score-critical fields from a persisted AMD SOL v2 sidecar."""
    if payload.get("schema_version") != AMD_SOL_V2_SCHEMA_VERSION:
        return None
    aggregate_payload = payload.get("aggregate_bound")
    hardware_payload = payload.get("hardware_model")
    coverage_payload = payload.get("coverage_summary")
    if not (
        isinstance(aggregate_payload, dict)
        and isinstance(hardware_payload, dict)
        and isinstance(coverage_payload, dict)
    ):
        return None

    try:
        aggregate = AmdSolV2AggregateBound(
            status=str(aggregate_payload["status"]),
            scored=bool(aggregate_payload["scored"]),
            sol_bound_ms=float(aggregate_payload["sol_bound_ms"]),
            reason=str(aggregate_payload["reason"]),
            node_ids=tuple(str(item) for item in aggregate_payload["node_ids"]),
        )
        coverage = AmdSolV2CoverageSummary(
            total_ops=int(coverage_payload.get("total_ops", 0)),
            supported_ops=int(coverage_payload.get("supported_ops", 0)),
            inexact_ops=int(coverage_payload.get("inexact_ops", 0)),
            unsupported_ops=int(coverage_payload.get("unsupported_ops", 0)),
            op_family_counts={
                str(key): int(value)
                for key, value in dict(
                    coverage_payload.get("op_family_counts", {})
                ).items()
            },
            confidence_counts_by_family={
                str(family): {
                    str(key): int(value) for key, value in dict(counts).items()
                }
                for family, counts in dict(
                    coverage_payload.get("confidence_counts_by_family", {})
                ).items()
            },
            worst_confidence=EstimateConfidence(
                str(coverage_payload.get("worst_confidence", "unsupported"))
            ),
        )
        hardware_model = amd_hardware_model_from_dict(
            hardware_payload,
            source="AMD SOL v2 sidecar hardware_model",
        )
    except (KeyError, TypeError, ValueError):
        return None

    return AmdSolBoundV2Artifact(
        definition=str(payload.get("definition", "")),
        workload_uuid=str(payload.get("workload_uuid", "")),
        hardware_model_ref=(
            str(payload["hardware_model_ref"])
            if payload.get("hardware_model_ref") is not None
            else None
        ),
        hardware_model=hardware_model,
        bound_graph={},
        operator_work_estimates=(),
        op_bounds=(),
        aggregate_bound=aggregate,
        warnings=tuple(str(item) for item in payload.get("warnings", [])),
        coverage_summary=coverage,
    )


def _minimal_solar_aggregate_from_payload(
    payload: dict,
) -> SolarAggregateStatus | None:
    aggregate_payload = payload.get("aggregate_status")
    if not isinstance(aggregate_payload, dict):
        return None
    try:
        return SolarAggregateStatus(
            status=str(aggregate_payload["status"]),
            score_eligible=bool(aggregate_payload["score_eligible"]),
            reason=str(aggregate_payload["reason"]),
            group_ids=tuple(str(item) for item in aggregate_payload["group_ids"]),
            node_ids=tuple(str(item) for item in aggregate_payload["node_ids"]),
            warnings=tuple(str(item) for item in aggregate_payload["warnings"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


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
    definition = Definition(**definition_payload)
    workloads = {
        workload.uuid: workload
        for workload in (
            Workload(**json.loads(line))
            for line in workload_path.read_text().splitlines()
            if line.strip()
        )
    }
    hardware_models = default_amd_hardware_models()
    hardware_model_key = _hardware_model_key_from_trace_payloads(traces_payload)
    hardware_model = hardware_models[hardware_model_key]
    scores: list[AmdNativeScore] = []
    derived_sidecar_exclusions = derived_sidecar_exclusions or {}

    for trace_index, trace_payload in enumerate(traces_payload):
        trace = Trace(**trace_payload)
        workload = workloads.get(trace.workload.uuid)
        derived_exclusion = derived_sidecar_exclusions.get(trace.workload.uuid)
        sidecar_stem = (
            sidecar_stem_for_workload(
                definition.name,
                trace.workload.uuid,
                problem_namespace=sidecar_namespace,
            )
            if workload is not None
            else None
        )
        sol_bound_ref = (
            f"derived:{definition.name}:{trace.workload.uuid}:amd_sol_bound_v2"
        )
        sol_bound_path = (
            sol_bound_artifact_dir / f"{sidecar_stem}.amd-sol-v2.json"
            if sol_bound_artifact_dir is not None and sidecar_stem is not None
            else None
        )
        artifact = None
        if sol_bound_path is not None and sol_bound_path.exists():
            existing_payload = _read_json_object(sol_bound_path)
            if existing_payload is not None:
                artifact = _minimal_amd_sol_bound_v2_from_payload(existing_payload)
                if artifact is not None:
                    sol_bound_ref = str(sol_bound_path)
        if artifact is None and workload is not None and derived_exclusion is None:
            artifact = build_amd_sol_bound_v2_artifact(
                definition,
                workload,
                hardware_model,
                hardware_model_ref=f"default_amd_hardware_models.{hardware_model_key}",
            )
        solar_derivation = None
        derived_evidence_refs = None
        solar_derivation_ref = (
            f"derived:{definition.name}:{trace.workload.uuid}:solar_derivation"
        )
        solar_derivation_path = (
            solar_derivation_dir / f"{sidecar_stem}.solar-derivation.json"
            if solar_derivation_dir is not None and sidecar_stem is not None
            else None
        )
        if solar_derivation_path is not None and solar_derivation_path.exists():
            existing_payload = _read_json_object(solar_derivation_path)
            if existing_payload is not None:
                solar_derivation = _minimal_solar_aggregate_from_payload(
                    existing_payload
                )
                if solar_derivation is not None:
                    solar_derivation_ref = str(solar_derivation_path)
        if (
            workload is not None
            and solar_derivation_dir is not None
            and derived_exclusion is None
        ):
            solar_derivation_dir.mkdir(parents=True, exist_ok=True)
            assert solar_derivation_path is not None
            if solar_derivation is None:
                generated = build_solar_derivation_evidence(definition, workload)
                generated_payload = generated.to_dict()
                solar_derivation_path.write_text(
                    json.dumps(generated_payload, indent=2)
                )
                solar_derivation_ref = str(solar_derivation_path)
                try:
                    solar_derivation = solar_derivation_from_dict(generated_payload)
                except ValueError as exc:
                    solar_derivation = None
                    derived_evidence_refs = {"solar_derivation_parse_error": str(exc)}
            else:
                solar_derivation_ref = str(solar_derivation_path)
            derived_evidence_refs = {
                "formula": f"{solar_derivation_ref}#groups.formula_evidence",
                "hardware_model": f"default_amd_hardware_models.{hardware_model_key}",
                "coverage": f"{solar_derivation_ref}#coverage_summary",
                "score_eligibility": f"{solar_derivation_ref}#aggregate_status",
                **(derived_evidence_refs or {}),
            }
        elif derived_exclusion is not None:
            derived_evidence_refs = {
                "derived_sidecar_exclusion": derived_exclusion,
                "hardware_model": f"default_amd_hardware_models.{hardware_model_key}",
            }
        if artifact is not None and sol_bound_path is not None:
            assert sol_bound_artifact_dir is not None
            sol_bound_artifact_dir.mkdir(parents=True, exist_ok=True)
            if not sol_bound_path.exists():
                sol_bound_path.write_text(json.dumps(artifact.to_dict(), indent=2))
            sol_bound_ref = str(sol_bound_path)
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
        if trace_index % 16 == 0:
            gc.collect()
    return scores


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


def write_amd_score_report(
    report_path: Path,
    amd_scores: list[AmdNativeScore],
    *,
    problem_count: int,
    baseline_entry_count: int,
) -> None:
    """Write the AMD-native suite score report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_amd_native_suite_report(
        amd_scores,
        baseline_summary={
            "problems": problem_count,
            "scores": len(amd_scores),
            "baseline_entries": baseline_entry_count,
        },
    )
    report_path.write_text(json.dumps(report.to_dict(), indent=2))


_extend_derived_reports_for_problem = extend_derived_reports_for_problem
