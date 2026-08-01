# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Execution helpers for static kernel extractors."""

from __future__ import annotations

import subprocess
from pathlib import Path

from sol_execbench.core.bench.static_kernel.artifacts import (
    display_artifact_path,
)
from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceClassification,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceStatus,
    StaticKernelEvidenceToolRun,
)
from sol_execbench.core.integrity.checksums import sha256_file
from sol_execbench.core.process.subprocesses import (
    ProbeCompletedProcess,
    ProbeRunner,
    run_bounded_probe,
)

RAW_OUTPUT_LIMIT = 64 * 1024
TAIL_LIMIT = 4000


def run_static_extractor(
    *,
    tool_id: str,
    command: list[str],
    artifact: StaticKernelEvidenceArtifact,
    evidence_root: Path,
    sidecar_base: Path,
    timeout_seconds: float,
    runner: ProbeRunner | None,
) -> tuple[StaticKernelEvidenceToolRun, StaticKernelEvidenceArtifact | None]:
    """Run one bounded extractor and persist its raw output."""
    effective_runner = runner or run_bounded_probe
    try:
        completed = effective_runner(command, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        return _timeout_result(
            tool_id,
            command,
            artifact,
            evidence_root,
            sidecar_base,
            timeout_seconds,
            exc,
        )
    except OSError as exc:
        return _os_error_result(
            tool_id,
            command,
            artifact,
            evidence_root,
            sidecar_base,
            timeout_seconds,
            exc,
        )
    return _completed_result(
        tool_id,
        command,
        artifact,
        evidence_root,
        sidecar_base,
        timeout_seconds,
        completed,
    )


def _timeout_result(
    tool_id: str,
    command: list[str],
    artifact: StaticKernelEvidenceArtifact,
    evidence_root: Path,
    sidecar_base: Path,
    timeout_seconds: float,
    error: subprocess.TimeoutExpired,
) -> tuple[StaticKernelEvidenceToolRun, StaticKernelEvidenceArtifact]:
    """Persist timeout output and return its stable failure record."""
    raw_artifact = write_raw_extractor_output(
        tool_id=tool_id,
        artifact_id=artifact.artifact_id,
        evidence_root=evidence_root,
        sidecar_base=sidecar_base,
        stdout=decode_output(error.stdout),
        stderr=decode_output(error.stderr),
    )
    return (
        StaticKernelEvidenceToolRun(
            tool_id=tool_id,
            command=command,
            status=StaticKernelEvidenceStatus.FAILED,
            reason_code=StaticKernelEvidenceReasonCode.EXTRACTOR_TIMEOUT,
            stdout_tail=tail_text(error.stdout),
            stderr_tail=tail_text(error.stderr),
            timeout_seconds=timeout_seconds,
            raw_output_path=raw_artifact.persisted_path,
        ),
        raw_artifact,
    )


def _os_error_result(
    tool_id: str,
    command: list[str],
    artifact: StaticKernelEvidenceArtifact,
    evidence_root: Path,
    sidecar_base: Path,
    timeout_seconds: float,
    error: OSError,
) -> tuple[StaticKernelEvidenceToolRun, StaticKernelEvidenceArtifact]:
    """Persist process-start failures in the same raw-output form as runs."""
    raw_artifact = write_raw_extractor_output(
        tool_id=tool_id,
        artifact_id=artifact.artifact_id,
        evidence_root=evidence_root,
        sidecar_base=sidecar_base,
        stdout="",
        stderr=str(error),
    )
    return (
        StaticKernelEvidenceToolRun(
            tool_id=tool_id,
            command=command,
            status=StaticKernelEvidenceStatus.FAILED,
            reason_code=StaticKernelEvidenceReasonCode.EXTRACTOR_FAILED,
            stderr_tail=tail_text(str(error)),
            timeout_seconds=timeout_seconds,
            raw_output_path=raw_artifact.persisted_path,
        ),
        raw_artifact,
    )


def _completed_result(
    tool_id: str,
    command: list[str],
    artifact: StaticKernelEvidenceArtifact,
    evidence_root: Path,
    sidecar_base: Path,
    timeout_seconds: float,
    completed: ProbeCompletedProcess,
) -> tuple[StaticKernelEvidenceToolRun, StaticKernelEvidenceArtifact]:
    """Persist a completed extractor process and classify its return code."""
    raw_artifact = write_raw_extractor_output(
        tool_id=tool_id,
        artifact_id=artifact.artifact_id,
        evidence_root=evidence_root,
        sidecar_base=sidecar_base,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    status = (
        StaticKernelEvidenceStatus.COLLECTED
        if completed.returncode == 0
        else StaticKernelEvidenceStatus.FAILED
    )
    reason_code = (
        StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED
        if completed.returncode == 0
        else StaticKernelEvidenceReasonCode.EXTRACTOR_FAILED
    )
    return (
        StaticKernelEvidenceToolRun(
            tool_id=tool_id,
            command=command,
            status=status,
            reason_code=reason_code,
            returncode=completed.returncode,
            stdout_tail=tail_text(completed.stdout),
            stderr_tail=tail_text(completed.stderr),
            timeout_seconds=timeout_seconds,
            raw_output_path=raw_artifact.persisted_path,
        ),
        raw_artifact,
    )


def write_raw_extractor_output(
    *,
    tool_id: str,
    artifact_id: str,
    evidence_root: Path,
    sidecar_base: Path,
    stdout: object,
    stderr: object,
) -> StaticKernelEvidenceArtifact:
    """Persist bounded extractor output and return its artifact record."""
    output_path = evidence_root / "extractors" / artifact_id / f"{tool_id}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = bounded_raw_output(stdout=stdout, stderr=stderr)
    output_path.write_text(output_text)
    return StaticKernelEvidenceArtifact(
        artifact_id=f"{artifact_id}-{tool_id}-raw-output",
        artifact_type="extractor_raw_output",
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        persisted_path=display_artifact_path(output_path, sidecar_base),
        size_bytes=output_path.stat().st_size,
        sha256=sha256_file(output_path),
        producer=tool_id,
        inspectable=False,
        classification=StaticKernelEvidenceClassification(
            metadata_present=True,
        ),
    )


def bounded_raw_output(*, stdout: object, stderr: object) -> str:
    """Combine and bound decoded extractor streams."""
    text = (
        f"stdout:\n{decode_output(stdout)}\n\nstderr:\n{decode_output(stderr)}"
    )
    return text[-RAW_OUTPUT_LIMIT:]


def decode_output(value: object) -> str:
    """Decode a possibly byte-valued subprocess stream."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def tail_text(value: object, limit: int = TAIL_LIMIT) -> str:
    """Return a bounded decoded tail from a subprocess stream."""
    return decode_output(value)[-limit:]
