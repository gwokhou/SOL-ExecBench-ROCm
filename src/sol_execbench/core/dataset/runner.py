"""Importable helpers for dataset-scale SOL ExecBench runs."""

from __future__ import annotations

import ast
import io
import json
import subprocess
import sys
import tokenize
from pathlib import Path


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

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60)
    except subprocess.TimeoutExpired as exc:
        print(f"CLI timed out for {job_name}: {exc.timeout} seconds")
        save_cli_timeout_log(output_dir, job_name, exc)
        return None

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
        save_cli_log(output_dir, job_name, result)
        return None

    if not traces:
        print(f"CLI failed for {job_name}: {result.stderr[:500]}")
        save_cli_log(output_dir, job_name, result)
        return None

    return traces


CLI_LOG_LIMIT = 64 * 1024


def bounded_cli_stream(value: str | bytes | None) -> str:
    """Return a bounded text representation of captured CLI output."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode(errors="replace")
    else:
        text = value
    if len(text) <= CLI_LOG_LIMIT:
        return text
    return text[:CLI_LOG_LIMIT] + "\n[truncated CLI output]\n"


def save_cli_log(output_dir: Path, job_name: str, result: subprocess.CompletedProcess):
    """Write stdout/stderr from a failed CLI invocation to a log file."""
    log_path = output_dir / f"{job_name}_cli.log"
    parts = [
        f"exit code: {result.returncode}",
        f"\n--- stdout ---\n{bounded_cli_stream(result.stdout)}" if result.stdout else "",
        f"\n--- stderr ---\n{bounded_cli_stream(result.stderr)}" if result.stderr else "",
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


def cli_failure_notes(cli_log: Path) -> list[str]:
    """Return human-readable failure notes from a saved CLI log."""
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


_bounded_cli_stream = bounded_cli_stream
_save_cli_log = save_cli_log
_save_cli_timeout_log = save_cli_timeout_log
_cli_failure_notes = cli_failure_notes
