# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""No-trace diagnostics sidecar helpers for CLI evaluation."""

from __future__ import annotations

import tempfile
from pathlib import Path

from rich.console import Console

from sol_execbench.core.bench.stderr import filter_benign_rocm_stderr
from sol_execbench.core.runtime_evidence import write_json_payload

NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION = "sol_execbench.no_trace_diagnostics.v1"
_DIAGNOSTIC_TAIL_LIMIT = 8192

console = Console(stderr=True)


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
