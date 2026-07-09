# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Execution helpers for static kernel extractors."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from sol_execbench.core.bench.static_kernel.artifacts import (
    _relative_path_string,
    _sha256_file,
)
from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceClassification,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceStatus,
    StaticKernelEvidenceToolRun,
)
from sol_execbench.core.platform.environment import ProbeCompletedProcess

ExtractorRunner = Callable[[list[str], float], ProbeCompletedProcess]
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
    runner: ExtractorRunner | None,
) -> tuple[StaticKernelEvidenceToolRun, StaticKernelEvidenceArtifact | None]:
    effective_runner = runner or run_extractor_command
    try:
        completed = effective_runner(command, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raw_artifact = write_raw_extractor_output(
            tool_id=tool_id,
            artifact_id=artifact.artifact_id,
            evidence_root=evidence_root,
            sidecar_base=sidecar_base,
            stdout=decode_output(exc.stdout),
            stderr=decode_output(exc.stderr),
        )
        return (
            StaticKernelEvidenceToolRun(
                tool_id=tool_id,
                command=command,
                status=StaticKernelEvidenceStatus.FAILED,
                reason_code=StaticKernelEvidenceReasonCode.EXTRACTOR_TIMEOUT,
                stdout_tail=tail_text(exc.stdout),
                stderr_tail=tail_text(exc.stderr),
                timeout_seconds=timeout_seconds,
                raw_output_path=raw_artifact.persisted_path,
            ),
            raw_artifact,
        )
    except OSError as exc:
        raw_artifact = write_raw_extractor_output(
            tool_id=tool_id,
            artifact_id=artifact.artifact_id,
            evidence_root=evidence_root,
            sidecar_base=sidecar_base,
            stdout="",
            stderr=str(exc),
        )
        return (
            StaticKernelEvidenceToolRun(
                tool_id=tool_id,
                command=command,
                status=StaticKernelEvidenceStatus.FAILED,
                reason_code=StaticKernelEvidenceReasonCode.EXTRACTOR_FAILED,
                stderr_tail=tail_text(str(exc)),
                timeout_seconds=timeout_seconds,
                raw_output_path=raw_artifact.persisted_path,
            ),
            raw_artifact,
        )

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


def run_extractor_command(
    command: list[str], timeout_seconds: float
) -> ProbeCompletedProcess:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return ProbeCompletedProcess(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
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
    output_path = evidence_root / "extractors" / artifact_id / f"{tool_id}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = bounded_raw_output(stdout=stdout, stderr=stderr)
    output_path.write_text(output_text)
    return StaticKernelEvidenceArtifact(
        artifact_id=f"{artifact_id}-{tool_id}-raw-output",
        artifact_type="extractor_raw_output",
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        persisted_path=_relative_path_string(output_path, sidecar_base),
        size_bytes=output_path.stat().st_size,
        sha256=_sha256_file(output_path),
        producer=tool_id,
        inspectable=False,
        classification=StaticKernelEvidenceClassification(metadata_present=True),
    )


def bounded_raw_output(*, stdout: object, stderr: object) -> str:
    text = f"stdout:\n{decode_output(stdout)}\n\nstderr:\n{decode_output(stderr)}"
    return text[-RAW_OUTPUT_LIMIT:]


def decode_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def tail_text(value: object, limit: int = TAIL_LIMIT) -> str:
    return decode_output(value)[-limit:]
