# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""No-trace diagnostics sidecar helpers for CLI evaluation."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field
from rich.console import Console

from sol_execbench.core.bench.performance_model.schema_versions import (
    PerformanceArtifactSchema,
    PerformanceDiagnosticArtifactKind,
)
from sol_execbench.core.bench.stderr import filter_benign_rocm_stderr
from sol_execbench.core.data.base_model import CurrentFrozenSchemaModel
from sol_execbench.core.data.json_utils import canonical_json_bytes
from sol_execbench.core.process.logs import redacted_text_tail

_DIAGNOSTIC_TAIL_LIMIT = 8192

console = Console(stderr=True)


@dataclass(frozen=True, slots=True, kw_only=True)
class NoTraceDiagnostics:
    """Bounded subprocess diagnostics for one execution without traces."""

    reason: str
    returncode: int
    stdout: str
    stderr: str


class NoTraceDiagnosticsSidecar(CurrentFrozenSchemaModel):
    """Current bounded no-trace diagnostic artifact."""

    current_schema_version = PerformanceArtifactSchema.PERFORMANCE_DIAGNOSTIC
    current_artifact_kind = PerformanceDiagnosticArtifactKind.NO_TRACE

    schema_version: Literal[
        PerformanceArtifactSchema.PERFORMANCE_DIAGNOSTIC
    ] = PerformanceArtifactSchema.PERFORMANCE_DIAGNOSTIC
    artifact_kind: Literal[PerformanceDiagnosticArtifactKind.NO_TRACE] = (
        PerformanceDiagnosticArtifactKind.NO_TRACE
    )
    diagnostic_only: Literal[True]
    canonical_trace_jsonl: Literal[False]
    reason: str
    returncode: int
    stdout_tail: str = Field(max_length=_DIAGNOSTIC_TAIL_LIMIT)
    stderr_tail: str = Field(max_length=_DIAGNOSTIC_TAIL_LIMIT)
    stdout_line_count: int = Field(ge=0)
    stderr_line_count: int = Field(ge=0)
    stdout_truncated: bool
    stderr_truncated: bool


def _diagnostic_tail(text: str, *, limit: int = _DIAGNOSTIC_TAIL_LIMIT) -> str:
    """Return a bounded redacted tail for diagnostic-only subprocess output."""
    return redacted_text_tail(text, limit)


def _no_trace_diagnostics_sidecar_path(
    output_file: Path | None,
    staging_dir: Path,
    *,
    keep_staging: bool,
) -> Path:
    """Return a persisted diagnostic sidecar path for no-trace outcomes."""
    if output_file is not None:
        return output_file.with_name(
            f"{output_file.name}.no-trace-diagnostics.json",
        )
    if keep_staging:
        return staging_dir / "no-trace-diagnostics.json"
    private_dir = Path(
        tempfile.mkdtemp(prefix=f"{staging_dir.name}.diagnostics."),
    )
    return private_dir / f"{staging_dir.name}.no-trace-diagnostics.json"


def _write_private_json_sidecar(path: Path, payload: object) -> None:
    if path.is_symlink() or path.parent.is_symlink():
        raise OSError(f"refusing to write diagnostic sidecar symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload, sort_keys=False))
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_symlink():
            raise OSError(
                f"refusing to replace symlink with diagnostic sidecar: {path}"
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_no_trace_diagnostics_sidecar(
    *,
    output_file: Path | None,
    staging_dir: Path,
    keep_staging: bool,
    diagnostics: NoTraceDiagnostics,
) -> Path | None:
    """Persist bounded diagnostic-only evidence for no-trace outcomes."""
    try:
        sidecar_path = _no_trace_diagnostics_sidecar_path(
            output_file,
            staging_dir,
            keep_staging=keep_staging,
        )
        filtered_stderr = filter_benign_rocm_stderr(diagnostics.stderr)
        stdout_truncated = (
            len(diagnostics.stdout) > _DIAGNOSTIC_TAIL_LIMIT
            or len(
                redacted_text_tail(
                    diagnostics.stdout,
                    _DIAGNOSTIC_TAIL_LIMIT + 1,
                )
            )
            > _DIAGNOSTIC_TAIL_LIMIT
        )
        stderr_truncated = (
            len(filtered_stderr) > _DIAGNOSTIC_TAIL_LIMIT
            or len(
                redacted_text_tail(
                    filtered_stderr,
                    _DIAGNOSTIC_TAIL_LIMIT + 1,
                )
            )
            > _DIAGNOSTIC_TAIL_LIMIT
        )
        payload = NoTraceDiagnosticsSidecar.model_validate(
            {
                "schema_version": (
                    PerformanceArtifactSchema.PERFORMANCE_DIAGNOSTIC
                ),
                "artifact_kind": PerformanceDiagnosticArtifactKind.NO_TRACE,
                "diagnostic_only": True,
                "canonical_trace_jsonl": False,
                "reason": diagnostics.reason,
                "returncode": diagnostics.returncode,
                "stdout_tail": _diagnostic_tail(diagnostics.stdout),
                "stderr_tail": _diagnostic_tail(filtered_stderr),
                "stdout_line_count": len(diagnostics.stdout.splitlines()),
                "stderr_line_count": len(filtered_stderr.splitlines()),
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            },
        )
        _write_private_json_sidecar(
            sidecar_path,
            payload.model_dump(mode="json"),
        )
        return sidecar_path
    except OSError as exc:
        console.print(
            f"[yellow]Failed to write no-trace diagnostics: {exc}[/yellow]",
        )
        return None
