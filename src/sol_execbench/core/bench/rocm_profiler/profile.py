# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Optional ROCm profiler artifact collection."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from sol_execbench.core.bench.rocm_profiler.artifacts import (
    discover_rocprofv3_artifacts,
    is_profile_artifact_candidate,
    normalize_profile_artifact_token,
    profile_artifact_coverage_metadata,
    profile_output_directory_listing,
)
from sol_execbench.core.bench.rocm_profiler.commands import (
    ProfileRunner,
    build_rocprofv3_profile_command,
    default_profile_runner,
)
from sol_execbench.core.bench.rocm_profiler.models import (
    PROFILE_OUTPUT_DIR_NAMES,
    Rocprofv3ArtifactCoverageStatus,
    Rocprofv3ProfileArtifact,
    Rocprofv3ProfileRequest,
    Rocprofv3ProfileResult,
    Rocprofv3ProfileStatus,
    Rocprofv3ReasonCode,
    has_profiler_data_artifact,
)
from sol_execbench.core.bench.rocm_profiler.schema_versions import (
    ProfilerArtifactSchema,
    ProfilerSessionArtifactKind,
)
from sol_execbench.core.data.base_model import CurrentFrozenSchemaModel
from sol_execbench.core.text_utils import subprocess_text, text_tail


class Rocprofv3Diagnostics(CurrentFrozenSchemaModel):
    """Current bounded no-data profiler diagnostic artifact."""

    current_schema_version = ProfilerArtifactSchema.ROCPROFV3_SESSION
    current_artifact_kind = ProfilerSessionArtifactKind.DIAGNOSTICS

    schema_version: Literal[ProfilerArtifactSchema.ROCPROFV3_SESSION] = (
        ProfilerArtifactSchema.ROCPROFV3_SESSION
    )
    artifact_kind: Literal[ProfilerSessionArtifactKind.DIAGNOSTICS] = (
        ProfilerSessionArtifactKind.DIAGNOSTICS
    )
    generated_at: str
    diagnostic_only: Literal[True]
    score_authority: Literal[False]
    status: Literal["no_profiler_data_artifacts"]
    returncode: int
    command: list[str] = Field(min_length=1)
    working_directory: str | None
    output_directory: str
    output_file: str
    output_format: str
    output_directory_listing: list[str] | tuple[str, ...]
    stdout_tail: str = Field(max_length=4096)
    stderr_tail: str = Field(max_length=4096)
    reason_codes: list[str] = Field(min_length=1)


def collect_rocprofv3_profile(
    request: Rocprofv3ProfileRequest,
    *,
    rocprofv3_available: bool = True,
    runner: ProfileRunner | None = None,
) -> Rocprofv3ProfileResult:
    """Collect optional `rocprofv3` artifacts without changing score semantics."""
    command = build_rocprofv3_profile_command(
        request.application_command,
        output_directory=str(request.output_directory),
        output_file=request.output_file,
        executable=request.executable,
        include_hip_runtime=request.include_hip_runtime,
        output_format=request.output_format,
    )
    if not rocprofv3_available:
        return _unavailable_result(request, command)

    prepare_profile_output_directory(
        request.output_directory,
        request.output_file,
    )
    run = runner or default_profile_runner
    try:
        completed = run(
            command,
            request.working_directory,
            request.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return _timeout_result(request, command, exc)
    return _completed_profile_result(request, command, completed)


def _unavailable_result(
    request: Rocprofv3ProfileRequest,
    command: Sequence[str],
) -> Rocprofv3ProfileResult:
    return Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.UNAVAILABLE,
        command=tuple(command),
        output_directory=request.output_directory,
        output_file=request.output_file,
        skipped_reason=f"{request.executable} is not available on PATH",
        working_directory=request.working_directory,
        timeout_seconds=request.timeout_seconds,
        profiler_available=False,
        artifact_coverage_status=Rocprofv3ArtifactCoverageStatus.UNAVAILABLE,
        reason_codes=(Rocprofv3ReasonCode.UNAVAILABLE,),
        **profile_result_metadata(request),
    )


def _timeout_result(
    request: Rocprofv3ProfileRequest,
    command: Sequence[str],
    error: subprocess.TimeoutExpired,
) -> Rocprofv3ProfileResult:
    """Preserve partial artifacts when a bounded profiler process times out."""
    artifacts = discover_rocprofv3_artifacts(
        request.output_directory,
        request.output_file,
    )
    coverage, reasons, warnings = profile_artifact_coverage_metadata(
        artifacts,
        command_succeeded=False,
    )
    return Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.FAILED,
        command=tuple(command),
        output_directory=request.output_directory,
        output_file=request.output_file,
        artifacts=artifacts,
        stdout=subprocess_text(error.stdout),
        stderr=subprocess_text(error.stderr),
        failed_reason=f"rocprofv3 command timed out after {request.timeout_seconds} seconds",
        working_directory=request.working_directory,
        timeout_seconds=request.timeout_seconds,
        profiler_available=True,
        artifact_coverage_status=coverage,
        reason_codes=(Rocprofv3ReasonCode.COMMAND_TIMEOUT, *reasons),
        warnings=warnings,
        **profile_result_metadata(request, artifacts),
    )


def _completed_profile_result(
    request: Rocprofv3ProfileRequest,
    command: Sequence[str],
    completed: subprocess.CompletedProcess[str],
) -> Rocprofv3ProfileResult:
    """Classify a completed ROCm profiler process and its registered artifacts."""
    artifacts = discover_rocprofv3_artifacts(
        request.output_directory,
        request.output_file,
    )
    if completed.returncode != 0:
        return _failed_command_result(request, command, completed, artifacts)
    artifacts = _diagnostic_artifacts_when_empty(
        request,
        completed,
        command,
        artifacts,
    )
    if not artifacts:
        return _no_artifacts_result(request, command, completed)
    return _successful_profile_result(request, command, completed, artifacts)


def _failed_command_result(
    request: Rocprofv3ProfileRequest,
    command: Sequence[str],
    completed: subprocess.CompletedProcess[str],
    artifacts: Sequence[Rocprofv3ProfileArtifact],
) -> Rocprofv3ProfileResult:
    coverage, reasons, warnings = profile_artifact_coverage_metadata(
        artifacts,
        command_succeeded=False,
    )
    return Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.FAILED,
        command=tuple(command),
        output_directory=request.output_directory,
        output_file=request.output_file,
        artifacts=tuple(artifacts),
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        failed_reason=f"rocprofv3 command failed with exit code {completed.returncode}",
        working_directory=request.working_directory,
        timeout_seconds=request.timeout_seconds,
        profiler_available=True,
        artifact_coverage_status=coverage,
        reason_codes=(Rocprofv3ReasonCode.COMMAND_FAILED, *reasons),
        warnings=warnings,
        **profile_result_metadata(request, artifacts),
    )


def _diagnostic_artifacts_when_empty(
    request: Rocprofv3ProfileRequest,
    completed: subprocess.CompletedProcess[str],
    command: Sequence[str],
    artifacts: Sequence[Rocprofv3ProfileArtifact],
) -> tuple[Rocprofv3ProfileArtifact, ...]:
    if artifacts:
        return tuple(artifacts)
    write_rocprofv3_diagnostic_artifact(request, completed, command)
    return discover_rocprofv3_artifacts(
        request.output_directory,
        request.output_file,
    )


def _no_artifacts_result(
    request: Rocprofv3ProfileRequest,
    command: Sequence[str],
    completed: subprocess.CompletedProcess[str],
) -> Rocprofv3ProfileResult:
    return Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.FAILED,
        command=tuple(command),
        output_directory=request.output_directory,
        output_file=request.output_file,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        failed_reason="rocprofv3 completed without registered artifacts",
        working_directory=request.working_directory,
        timeout_seconds=request.timeout_seconds,
        profiler_available=True,
        artifact_coverage_status=Rocprofv3ArtifactCoverageStatus.NONE,
        reason_codes=(Rocprofv3ReasonCode.NO_REGISTERED_ARTIFACTS,),
        **profile_result_metadata(request),
    )


def _successful_profile_result(
    request: Rocprofv3ProfileRequest,
    command: Sequence[str],
    completed: subprocess.CompletedProcess[str],
    artifacts: Sequence[Rocprofv3ProfileArtifact],
) -> Rocprofv3ProfileResult:
    coverage, reasons, warnings = profile_artifact_coverage_metadata(
        artifacts,
        command_succeeded=True,
    )
    status = Rocprofv3ProfileStatus.PARTIAL
    failed_reason: str | None = None
    if coverage is Rocprofv3ArtifactCoverageStatus.COMPLETE:
        status = Rocprofv3ProfileStatus.SUCCESS
    elif coverage is Rocprofv3ArtifactCoverageStatus.DIAGNOSTIC_LOGS_ONLY:
        failed_reason = (
            "rocprofv3 completed without profiler data artifacts; "
            "diagnostic log artifact registered"
        )
    return Rocprofv3ProfileResult(
        status=status,
        command=tuple(command),
        output_directory=request.output_directory,
        output_file=request.output_file,
        artifacts=tuple(artifacts),
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        failed_reason=failed_reason,
        working_directory=request.working_directory,
        timeout_seconds=request.timeout_seconds,
        profiler_available=True,
        artifact_coverage_status=coverage,
        reason_codes=reasons,
        warnings=warnings,
        **profile_result_metadata(request, artifacts),
    )


def profile_result_metadata(
    request: Rocprofv3ProfileRequest,
    artifacts: Sequence[Rocprofv3ProfileArtifact] = (),
) -> dict[str, Any]:
    """Fields every Rocprofv3ProfileResult derives from the request + artifacts.

    Collapses the ``output_format`` / ``profiler_data_artifacts`` /
    ``output_directory_listing`` trio that would otherwise be hand-copied onto
    every return site in ``collect_rocprofv3_profile``.
    """
    return {
        "output_format": request.output_format,
        "profiler_data_artifacts": has_profiler_data_artifact(artifacts),
        "output_directory_listing": profile_output_directory_listing(
            request.output_directory,
        ),
    }


def prepare_profile_output_directory(
    output_directory: Path,
    output_file: str,
) -> None:
    """Remove stale artifacts that would be registered for this profile run."""
    output_directory.mkdir(parents=True, exist_ok=True)
    for path in sorted(
        output_directory.rglob("*"),
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    ):
        if path.is_file():
            if is_profile_artifact_candidate(
                path,
                output_directory,
                output_file,
            ):
                path.unlink(missing_ok=True)
            continue
        if not path.is_dir():
            continue
        try:
            relative_parts = path.relative_to(output_directory).parts
        except ValueError:
            continue
        normalized_parts = {
            normalize_profile_artifact_token(part) for part in relative_parts
        }
        if (
            output_file in relative_parts
            or path.name.startswith(output_file)
            or bool(normalized_parts & PROFILE_OUTPUT_DIR_NAMES)
            or any(
                re.fullmatch(r"(?:pass|pmc)-\d+", part)
                for part in normalized_parts
            )
        ):
            shutil.rmtree(path, ignore_errors=True)


def write_rocprofv3_diagnostic_artifact(
    request: Rocprofv3ProfileRequest,
    completed: subprocess.CompletedProcess[str],
    command: Sequence[str],
) -> Path | None:
    """Persist bounded profiler execution diagnostics when rocprof writes no data."""
    path = request.output_directory / f"{request.output_file}.diagnostics.json"
    payload = Rocprofv3Diagnostics.model_validate(
        {
            "schema_version": ProfilerArtifactSchema.ROCPROFV3_SESSION,
            "artifact_kind": ProfilerSessionArtifactKind.DIAGNOSTICS,
            "generated_at": datetime.now(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "diagnostic_only": True,
            "score_authority": False,
            "status": "no_profiler_data_artifacts",
            "returncode": completed.returncode,
            "command": list(command),
            "working_directory": (
                str(request.working_directory)
                if request.working_directory is not None
                else None
            ),
            "output_directory": str(request.output_directory),
            "output_file": request.output_file,
            "output_format": request.output_format,
            "output_directory_listing": profile_output_directory_listing(
                request.output_directory,
            ),
            "stdout_tail": text_tail(completed.stdout or "", limit=4096),
            "stderr_tail": text_tail(completed.stderr or "", limit=4096),
            "reason_codes": [
                Rocprofv3ReasonCode.NO_REGISTERED_ARTIFACTS,
                Rocprofv3ReasonCode.DIAGNOSTIC_LOG_REGISTERED,
            ],
        },
    )
    try:
        path.write_text(
            json.dumps(payload.model_dump(mode="json"), sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        return None
    return path
