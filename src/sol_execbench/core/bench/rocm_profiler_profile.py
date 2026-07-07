# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Optional ROCm profiler artifact collection."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sol_execbench.core.bench.rocm_profiler_artifacts import (
    PROFILE_OUTPUT_DIR_NAMES,
    discover_rocprofv3_artifacts,
    has_profiler_data_artifact,
    is_profile_artifact_candidate,
    normalize_profile_artifact_token,
    profile_artifact_coverage_metadata,
    profile_output_directory_listing,
)
from sol_execbench.core.bench.rocm_profiler_commands import (
    ProfileRunner,
    build_rocprofv3_profile_command,
    default_profile_runner,
)
from sol_execbench.core.bench.rocm_profiler_models import (
    ROCPROF_REASON_COMMAND_FAILED,
    ROCPROF_REASON_COMMAND_TIMEOUT,
    ROCPROF_REASON_DIAGNOSTIC_LOG_REGISTERED,
    ROCPROF_REASON_NO_REGISTERED_ARTIFACTS,
    ROCPROF_REASON_UNAVAILABLE,
    Rocprofv3ProfileArtifact,
    Rocprofv3ProfileRequest,
    Rocprofv3ProfileResult,
    _tail as tail,
)
from sol_execbench.core.text_utils import subprocess_text


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
        return Rocprofv3ProfileResult(
            status="unavailable",
            command=tuple(command),
            output_directory=request.output_directory,
            output_file=request.output_file,
            skipped_reason=f"{request.executable} is not available on PATH",
            working_directory=request.working_directory,
            timeout_seconds=request.timeout_seconds,
            profiler_available=False,
            artifact_coverage_status="unavailable",
            reason_codes=(ROCPROF_REASON_UNAVAILABLE,),
            **profile_result_metadata(request),
        )

    prepare_profile_output_directory(request.output_directory, request.output_file)
    run = runner or default_profile_runner
    try:
        completed = run(
            command,
            request.working_directory,
            request.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        # rocprofv3 may have flushed partial artifacts before being killed;
        # discover them so they are not silently lost, and report a timeout
        # (not command_failed) reason code.
        timeout_artifacts = discover_rocprofv3_artifacts(
            request.output_directory,
            request.output_file,
        )
        timeout_coverage, timeout_reasons, timeout_warnings = (
            profile_artifact_coverage_metadata(
                timeout_artifacts,
                command_succeeded=False,
            )
        )
        return Rocprofv3ProfileResult(
            status="failed",
            command=tuple(command),
            output_directory=request.output_directory,
            output_file=request.output_file,
            artifacts=timeout_artifacts,
            stdout=subprocess_text(exc.stdout),
            stderr=subprocess_text(exc.stderr),
            failed_reason=(
                f"rocprofv3 command timed out after {request.timeout_seconds} seconds"
            ),
            working_directory=request.working_directory,
            timeout_seconds=request.timeout_seconds,
            profiler_available=True,
            artifact_coverage_status=timeout_coverage,
            reason_codes=(ROCPROF_REASON_COMMAND_TIMEOUT, *timeout_reasons),
            warnings=timeout_warnings,
            **profile_result_metadata(request, timeout_artifacts),
        )

    artifacts = discover_rocprofv3_artifacts(
        request.output_directory,
        request.output_file,
    )
    if completed.returncode != 0:
        coverage_status, coverage_reasons, coverage_warnings = (
            profile_artifact_coverage_metadata(
                artifacts,
                command_succeeded=False,
            )
        )
        return Rocprofv3ProfileResult(
            status="failed",
            command=tuple(command),
            output_directory=request.output_directory,
            output_file=request.output_file,
            artifacts=artifacts,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            failed_reason=f"rocprofv3 command failed with exit code {completed.returncode}",
            working_directory=request.working_directory,
            timeout_seconds=request.timeout_seconds,
            profiler_available=True,
            artifact_coverage_status=coverage_status,
            reason_codes=(ROCPROF_REASON_COMMAND_FAILED, *coverage_reasons),
            warnings=coverage_warnings,
            **profile_result_metadata(request, artifacts),
        )
    if not artifacts:
        # rocprof wrote nothing on disk. Persist a diagnostic log so the run is
        # not silent, then re-check. If a diagnostic artifact landed, fall
        # through to the unified tail below, which classifies it as
        # ``diagnostic_logs_only`` (status=partial) -- the same outcome that an
        # explicit early return here would produce. Only the genuinely-empty
        # case needs its own failed return, because the tail treats empty
        # coverage as success.
        write_rocprofv3_diagnostic_artifact(request, completed, command)
        artifacts = discover_rocprofv3_artifacts(
            request.output_directory,
            request.output_file,
        )
        if not artifacts:
            return Rocprofv3ProfileResult(
                status="failed",
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
                artifact_coverage_status="none",
                reason_codes=(ROCPROF_REASON_NO_REGISTERED_ARTIFACTS,),
                **profile_result_metadata(request),
            )

    coverage_status, coverage_reasons, coverage_warnings = (
        profile_artifact_coverage_metadata(
            artifacts,
            command_succeeded=True,
        )
    )
    status = "success"
    failed_reason = None
    if coverage_status == "diagnostic_logs_only":
        status = "partial"
        failed_reason = (
            "rocprofv3 completed without profiler data artifacts; "
            "diagnostic log artifact registered"
        )
    elif coverage_status == "partial":
        status = "partial"
    return Rocprofv3ProfileResult(
        status=status,
        command=tuple(command),
        output_directory=request.output_directory,
        output_file=request.output_file,
        artifacts=artifacts,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        failed_reason=failed_reason,
        working_directory=request.working_directory,
        timeout_seconds=request.timeout_seconds,
        profiler_available=True,
        artifact_coverage_status=coverage_status,
        reason_codes=coverage_reasons,
        warnings=coverage_warnings,
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
            request.output_directory
        ),
    }


def prepare_profile_output_directory(output_directory: Path, output_file: str) -> None:
    """Remove stale artifacts that would be registered for this profile run."""
    output_directory.mkdir(parents=True, exist_ok=True)
    for path in sorted(
        output_directory.rglob("*"),
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    ):
        if path.is_file():
            if is_profile_artifact_candidate(path, output_directory, output_file):
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
        ):
            shutil.rmtree(path, ignore_errors=True)


def write_rocprofv3_diagnostic_artifact(
    request: Rocprofv3ProfileRequest,
    completed: subprocess.CompletedProcess[str],
    command: Sequence[str],
) -> Path | None:
    """Persist bounded profiler execution diagnostics when rocprof writes no data."""
    path = request.output_directory / f"{request.output_file}.diagnostics.json"
    payload = {
        "schema_version": "sol_execbench.rocprofv3_diagnostics.v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
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
            request.output_directory
        ),
        "stdout_tail": tail(completed.stdout or ""),
        "stderr_tail": tail(completed.stderr or ""),
        "reason_codes": [
            ROCPROF_REASON_NO_REGISTERED_ARTIFACTS,
            ROCPROF_REASON_DIAGNOSTIC_LOG_REGISTERED,
        ],
    }
    try:
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        return None
    return path


# Compatibility aliases for the old monolithic module's private helper names.
_subprocess_text = subprocess_text
_profile_result_metadata = profile_result_metadata
_prepare_profile_output_directory = prepare_profile_output_directory
_write_rocprofv3_diagnostic_artifact = write_rocprofv3_diagnostic_artifact
_tail = tail
