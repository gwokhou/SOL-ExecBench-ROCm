# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Evaluation subprocess helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from rich.console import Console

from ..core.bench.io import flashinfer_safetensors_env
from ..core.bench.rocm_profiler import (
    ROCPROFV3_EXECUTABLE,
    Rocprofv3ProfileRequest,
    Rocprofv3ProfileResult,
    collect_rocprofv3_profile,
)
from ..core.bench.stderr import filter_benign_rocm_stderr
from ..core.runtime_evidence import write_json_payload
from .profile_sidecars import _profile_output_directory

console = Console(stderr=True)

NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION = "sol_execbench.no_trace_diagnostics.v1"
_DIAGNOSTIC_TAIL_LIMIT = 8192


def _diagnostic_tail(text: str, *, limit: int = _DIAGNOSTIC_TAIL_LIMIT) -> str:
    """Return a bounded tail for diagnostic-only subprocess output."""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _no_trace_diagnostics_sidecar_path(
    output_file: Path | None,
    staging_dir: Path,
    *,
    keep_staging: bool,
) -> Path:
    """Return a persisted diagnostic sidecar path for no-trace outcomes."""
    if output_file is not None:
        return output_file.with_name(f"{output_file.name}.no-trace-diagnostics.json")
    if keep_staging:
        return staging_dir / "no-trace-diagnostics.json"
    return Path(tempfile.gettempdir()) / f"{staging_dir.name}.no-trace-diagnostics.json"


def _write_no_trace_diagnostics_sidecar(
    *,
    output_file: Path | None,
    staging_dir: Path,
    keep_staging: bool,
    reason: str,
    returncode: int,
    stdout: str,
    stderr: str,
) -> Path | None:
    """Persist bounded diagnostic-only evidence for no-trace outcomes."""
    sidecar_path = _no_trace_diagnostics_sidecar_path(
        output_file,
        staging_dir,
        keep_staging=keep_staging,
    )
    filtered_stderr = filter_benign_rocm_stderr(stderr)
    payload = {
        "schema_version": NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION,
        "diagnostic_only": True,
        "canonical_trace_jsonl": False,
        "reason": reason,
        "returncode": returncode,
        "stdout_tail": _diagnostic_tail(stdout),
        "stderr_tail": _diagnostic_tail(filtered_stderr),
        "stdout_line_count": len(stdout.splitlines()),
        "stderr_line_count": len(filtered_stderr.splitlines()),
        "stdout_truncated": len(stdout) > _DIAGNOSTIC_TAIL_LIMIT,
        "stderr_truncated": len(filtered_stderr) > _DIAGNOSTIC_TAIL_LIMIT,
    }
    try:
        write_json_payload(sidecar_path, payload)
        return sidecar_path
    except OSError as exc:
        console.print(f"[yellow]Failed to write no-trace diagnostics: {exc}[/yellow]")
        return None


def _timeout_output_text(output: str | bytes | None) -> str:
    """Return timeout output as text regardless of subprocess typing."""

    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output


def _run_evaluation_command(
    eval_cmd: list[str],
    *,
    staging_dir: Path,
    timeout: int,
    env_builder: Callable[
        [Mapping[str, str]], dict[str, str]
    ] = flashinfer_safetensors_env,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    """Run the staged evaluation command with the standard ROCm allocator env."""

    env = env_builder({**os.environ, "PYTORCH_ALLOC_CONF": "expandable_segments:True"})
    return runner(
        eval_cmd,
        cwd=staging_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _run_profiled_evaluation(
    eval_cmd: list[str],
    *,
    staging_dir: Path,
    output_file: Path | None,
    timeout: int,
    env_builder: Callable[
        [Mapping[str, str]], dict[str, str]
    ] = flashinfer_safetensors_env,
    subprocess_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    rocprofv3_available: bool | None = None,
    profile_collector: Callable[
        ..., Rocprofv3ProfileResult
    ] = collect_rocprofv3_profile,
) -> tuple[subprocess.CompletedProcess[str] | None, Rocprofv3ProfileResult]:
    """Run evaluation under `rocprofv3`, returning normal execution on failure."""

    output_directory = _profile_output_directory(output_file, staging_dir)
    request = Rocprofv3ProfileRequest(
        application_command=tuple(eval_cmd),
        output_directory=output_directory,
        output_file="profile",
        working_directory=staging_dir,
        timeout_seconds=timeout,
    )
    if rocprofv3_available is None:
        rocprofv3_available = shutil.which(ROCPROFV3_EXECUTABLE) is not None
    profile_result = profile_collector(
        request,
        rocprofv3_available=rocprofv3_available,
        runner=lambda command, cwd, timeout_seconds: subprocess_run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env_builder(
                {
                    **os.environ,
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                    "SOL_EXECBENCH_GRACEFUL_EXIT": "1",
                }
            ),
        ),
    )
    if profile_result.succeeded:
        profiled_proc = subprocess.CompletedProcess(
            args=list(profile_result.command),
            returncode=profile_result.returncode or 0,
            stdout=profile_result.stdout,
            stderr=profile_result.stderr,
        )
        return profiled_proc, profile_result
    return None, profile_result
