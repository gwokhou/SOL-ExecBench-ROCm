"""CLI subprocess execution helpers for dataset-scale runs."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from sol_execbench.core.bench.io import flashinfer_safetensors_env
from sol_execbench.core.bench.stderr import filter_benign_rocm_stderr

CLI_LOG_LIMIT = 64 * 1024


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
