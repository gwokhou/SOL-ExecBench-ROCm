# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Explicit rocprofv3 counter-mode collection lifecycle."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from importlib.resources import as_file, files
from pathlib import Path

from sol_execbench.core.bench.rocm_profiler.artifacts import (
    discover_rocprofv3_artifacts,
    profile_artifact_coverage_metadata,
    profile_output_directory_listing,
)
from sol_execbench.core.bench.rocm_profiler.commands import (
    ProfileRunner,
    default_profile_runner,
)
from sol_execbench.core.bench.rocm_profiler.counter_provenance import (
    Rocprofv3CounterProvenance,
)
from sol_execbench.core.bench.rocm_profiler.counters import (
    ROCPROFV3_AVAIL_EXECUTABLE,
    build_rocprofv3_counter_command,
    counter_names_in_csv,
    counter_pass_index,
    load_counter_manifest,
    normalize_counter_name,
    parse_available_architectures,
    parse_available_counters,
    select_counter_groups,
    write_counter_job,
)
from sol_execbench.core.bench.rocm_profiler.models import (
    ROCPROF_REASON_COMMAND_FAILED,
    ROCPROF_REASON_COMMAND_TIMEOUT,
    ROCPROF_REASON_UNAVAILABLE,
    Rocprofv3ArtifactCoverageStatus,
    Rocprofv3ArtifactKind,
    Rocprofv3ProfileArtifact,
    Rocprofv3ProfileRequest,
    Rocprofv3ProfileResult,
    Rocprofv3ProfileStatus,
    has_profiler_data_artifact,
)
from sol_execbench.core.bench.rocm_profiler.profile import (
    prepare_profile_output_directory,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import (
    sha256_bytes,
    sha256_file,
    stable_json_checksum,
)
from sol_execbench.core.integrity.schema_versions import (
    ROCPROFV3_COUNTER_PROVENANCE_SCHEMA_VERSION,
)
from sol_execbench.core.platform.runtime import (
    resolve_rocm_tool,
    resolve_tool_path,
)
from sol_execbench.core.process.environment import (
    ENV_SOL_EXECBENCH_COUNTER_REPLAY,
    ENV_SOL_EXECBENCH_REPLAY_PASS_INDEX,
)
from sol_execbench.core.text_utils import subprocess_text

COUNTER_REASON_AVAIL_FAILED = "rocprof_counter_availability_failed"
COUNTER_REASON_UNSUPPORTED = "rocprof_required_counters_unsupported"
COUNTER_REASON_COLLECTED = "rocprof_counters_collected"
COUNTER_REASON_ARTIFACT_INCOMPLETE = "rocprof_counter_artifact_incomplete"


def collect_rocprofv3_counters(
    request: Rocprofv3ProfileRequest,
    *,
    rocprofv3_available: bool = True,
    runner: ProfileRunner | None = None,
) -> Rocprofv3ProfileResult:
    """Collect controlled gfx1200 counter passes and audit provenance."""
    run = runner or default_profile_runner
    if not rocprofv3_available:
        return _unavailable(request, "rocprofv3 is not available")
    avail_path = resolve_rocm_tool(ROCPROFV3_AVAIL_EXECUTABLE)
    if avail_path is None:
        return _unavailable(request, "rocprofv3-avail is not available")
    availability = _run_availability(request, run, str(avail_path))
    if isinstance(availability, Rocprofv3ProfileResult):
        return availability
    resource = files("sol_execbench.data.rocprofv3_counters").joinpath(
        "gfx1200_v1.yaml",
    )
    with as_file(resource) as manifest_path:
        manifest = load_counter_manifest(manifest_path)
        architectures = parse_available_architectures(availability.stdout)
        if manifest.architecture not in architectures:
            return _unsupported(
                request,
                [f"architecture:{manifest.architecture}"],
                availability,
            )
        groups, missing = select_counter_groups(
            manifest,
            parse_available_counters(availability.stdout),
        )
        if missing:
            return _unsupported(request, missing, availability)
        return _collect_selected(
            request,
            groups=groups,
            manifest_path=manifest_path,
            availability=availability,
            runner=run,
        )


def _run_availability(
    request: Rocprofv3ProfileRequest,
    runner: ProfileRunner,
    executable: str,
) -> subprocess.CompletedProcess[str] | Rocprofv3ProfileResult:
    command = [executable, "-d", "0", "info", "--pmc"]
    try:
        completed = runner(
            command,
            request.working_directory,
            request.timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        return _failed(
            request,
            tuple(command),
            COUNTER_REASON_AVAIL_FAILED,
            f"rocprofv3-avail timed out after {request.timeout_seconds} seconds",
            stdout=subprocess_text(error.stdout),
            stderr=subprocess_text(error.stderr),
        )
    if completed.returncode != 0:
        return _failed(
            request,
            tuple(command),
            COUNTER_REASON_AVAIL_FAILED,
            f"rocprofv3-avail failed with exit code {completed.returncode}",
            completed=completed,
        )
    return completed


def _collect_selected(
    request: Rocprofv3ProfileRequest,
    *,
    groups: list[list[str]],
    manifest_path: Path,
    availability: subprocess.CompletedProcess[str],
    runner: ProfileRunner,
) -> Rocprofv3ProfileResult:
    prepare_profile_output_directory(
        request.output_directory,
        request.output_file,
    )
    config_paths: list[Path] = []
    completed_passes: list[subprocess.CompletedProcess[str]] = []
    commands: list[list[str]] = []
    for pass_index, group in enumerate(groups, start=1):
        pass_directory = request.output_directory / f"pass_{pass_index}"
        config_path = request.output_directory / (
            f"{request.output_file}.pass_{pass_index}.counters.yaml"
        )
        write_counter_job(
            config_path,
            [group],
            output_directory=str(pass_directory),
        )
        application = (
            "env",
            f"{ENV_SOL_EXECBENCH_COUNTER_REPLAY}=1",
            f"{ENV_SOL_EXECBENCH_REPLAY_PASS_INDEX}={pass_index}",
            *request.application_command,
        )
        command = build_rocprofv3_counter_command(
            application,
            input_path=config_path,
            executable=request.executable,
        )
        config_paths.append(config_path)
        commands.append(command)
        try:
            completed = runner(
                command,
                request.working_directory,
                request.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            return _counter_timeout(request, command, error)
        completed_passes.append(completed)
        if completed.returncode != 0:
            break
    provenance = _write_provenance(
        request,
        config_paths=config_paths,
        manifest_path=manifest_path,
        availability=availability,
    )
    completed = _aggregate_completed(commands, completed_passes)
    return _counter_completed(
        request,
        [item for command in commands for item in command],
        completed,
        provenance,
        groups=groups,
    )


def _write_provenance(
    request: Rocprofv3ProfileRequest,
    *,
    config_paths: Sequence[Path],
    manifest_path: Path,
    availability: subprocess.CompletedProcess[str],
) -> dict[str, str]:
    executable = resolve_rocm_tool(request.executable)
    application_executable = resolve_tool_path(request.application_command[0])
    provenance = {
        "profiler_sha256": (
            sha256_file(executable) if executable is not None else "unresolved"
        ),
        "counter_definition_sha256": sha256_file(manifest_path),
        "configuration_sha256": stable_json_checksum(
            [sha256_file(path) for path in config_paths],
        ),
        "availability_sha256": sha256_bytes(availability.stdout.encode()),
        "application_executable_sha256": (
            sha256_file(application_executable)
            if application_executable is not None
            else "unresolved"
        ),
        "application_command_sha256": stable_json_checksum(
            list(request.application_command),
        ),
    }
    path = (
        request.output_directory
        / f"{request.output_file}.counter-metadata.json"
    )
    payload = Rocprofv3CounterProvenance.model_validate(
        {
            "schema_version": ROCPROFV3_COUNTER_PROVENANCE_SCHEMA_VERSION,
            "diagnostic_only": True,
            "score_authority": False,
            "replay_phase": "evidence",
            **provenance,
        },
    )
    atomic_write_json_value(path, payload.model_dump(mode="json"))
    return provenance


def _aggregate_completed(
    commands: Sequence[Sequence[str]],
    completed: Sequence[subprocess.CompletedProcess[str]],
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[item for command in commands for item in command],
        returncode=next(
            (result.returncode for result in completed if result.returncode),
            0,
        ),
        stdout="\n".join(result.stdout or "" for result in completed),
        stderr="\n".join(result.stderr or "" for result in completed),
    )


def _counter_completed(
    request: Rocprofv3ProfileRequest,
    command: list[str],
    completed: subprocess.CompletedProcess[str],
    provenance: dict[str, str],
    *,
    groups: Sequence[Sequence[str]],
) -> Rocprofv3ProfileResult:
    artifacts = discover_rocprofv3_artifacts(
        request.output_directory,
        request.output_file,
    )
    coverage, coverage_reasons, warnings = profile_artifact_coverage_metadata(
        artifacts,
        command_succeeded=completed.returncode == 0,
    )
    artifact_reasons = _counter_artifact_reasons(artifacts, groups)
    success = (
        completed.returncode == 0
        and coverage is Rocprofv3ArtifactCoverageStatus.COMPLETE
        and not artifact_reasons
    )
    failed_reason = None
    if not success:
        detail = ",".join(artifact_reasons)
        failed_reason = (
            "rocprofv3 counter collection did not produce complete profiler data"
            + (f": {detail}" if detail else "")
        )
    return Rocprofv3ProfileResult(
        status=(
            Rocprofv3ProfileStatus.SUCCESS
            if success
            else Rocprofv3ProfileStatus.FAILED
        ),
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
        artifact_coverage_status=coverage,
        reason_codes=(
            (COUNTER_REASON_COLLECTED, *coverage_reasons)
            if success
            else (
                ROCPROF_REASON_COMMAND_FAILED,
                *(
                    (COUNTER_REASON_ARTIFACT_INCOMPLETE,)
                    if artifact_reasons
                    else ()
                ),
                *coverage_reasons,
            )
        ),
        warnings=warnings,
        output_format="csv,rocpd",
        profiler_data_artifacts=has_profiler_data_artifact(artifacts),
        output_directory_listing=profile_output_directory_listing(
            request.output_directory,
        ),
        provenance=provenance,
    )


def _counter_artifact_reasons(
    artifacts: Sequence[Rocprofv3ProfileArtifact],
    groups: Sequence[Sequence[str]],
) -> list[str]:
    csv_by_pass = _artifacts_by_pass(
        artifacts,
        Rocprofv3ArtifactKind.COUNTER_CSV,
    )
    expected_passes = set(range(1, len(groups) + 1))
    observed_passes = set(csv_by_pass)
    reasons = [
        f"unexpected_counter_pass:{index}"
        for index in sorted(observed_passes - expected_passes)
    ]
    for pass_index, group in enumerate(groups, start=1):
        csv_artifacts = csv_by_pass.get(pass_index, [])
        if len(csv_artifacts) != 1:
            reasons.append(f"counter_pass_csv_count:{pass_index}")
        else:
            reasons.extend(
                _counter_csv_reasons(csv_artifacts[0].path, group, pass_index),
            )
    if not any(
        artifact.kind is Rocprofv3ArtifactKind.ROCPD for artifact in artifacts
    ):
        reasons.append("counter_rocpd_missing")
    return reasons


def _artifacts_by_pass(
    artifacts: Sequence[Rocprofv3ProfileArtifact],
    kind: Rocprofv3ArtifactKind,
) -> dict[int, list[Rocprofv3ProfileArtifact]]:
    result: dict[int, list[Rocprofv3ProfileArtifact]] = {}
    for artifact in artifacts:
        if artifact.kind is not kind:
            continue
        index = counter_pass_index(artifact.path)
        if index is not None:
            result.setdefault(index, []).append(artifact)
    return result


def _counter_csv_reasons(
    path: Path,
    requested: Sequence[str],
    pass_index: int,
) -> list[str]:
    try:
        observed = counter_names_in_csv(path)
    except (OSError, ValueError):
        return [f"counter_pass_csv_invalid:{pass_index}"]
    expected = {normalize_counter_name(name) for name in requested}
    if expected <= observed:
        return []
    return [f"counter_pass_counter_missing:{pass_index}"]


def _counter_timeout(
    request: Rocprofv3ProfileRequest,
    command: list[str],
    error: subprocess.TimeoutExpired,
) -> Rocprofv3ProfileResult:
    artifacts = discover_rocprofv3_artifacts(
        request.output_directory,
        request.output_file,
    )
    return Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.FAILED,
        command=tuple(command),
        output_directory=request.output_directory,
        output_file=request.output_file,
        artifacts=artifacts,
        stdout=subprocess_text(error.stdout),
        stderr=subprocess_text(error.stderr),
        failed_reason=(
            "rocprofv3 counter command timed out after "
            f"{request.timeout_seconds} seconds"
        ),
        working_directory=request.working_directory,
        timeout_seconds=request.timeout_seconds,
        profiler_available=True,
        artifact_coverage_status=Rocprofv3ArtifactCoverageStatus.PARTIAL,
        reason_codes=(ROCPROF_REASON_COMMAND_TIMEOUT,),
        output_format="csv,rocpd",
        profiler_data_artifacts=bool(artifacts),
        output_directory_listing=profile_output_directory_listing(
            request.output_directory,
        ),
    )


def _unavailable(
    request: Rocprofv3ProfileRequest,
    reason: str,
) -> Rocprofv3ProfileResult:
    return Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.UNAVAILABLE,
        command=(),
        output_directory=request.output_directory,
        output_file=request.output_file,
        skipped_reason=reason,
        working_directory=request.working_directory,
        timeout_seconds=request.timeout_seconds,
        profiler_available=False,
        artifact_coverage_status=Rocprofv3ArtifactCoverageStatus.UNAVAILABLE,
        reason_codes=(ROCPROF_REASON_UNAVAILABLE,),
        output_format="csv,rocpd",
    )


def _unsupported(
    request: Rocprofv3ProfileRequest,
    missing: list[str],
    availability: subprocess.CompletedProcess[str],
) -> Rocprofv3ProfileResult:
    return Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.UNAVAILABLE,
        command=tuple(str(item) for item in availability.args),
        output_directory=request.output_directory,
        output_file=request.output_file,
        stdout=availability.stdout,
        stderr=availability.stderr,
        skipped_reason=f"required counters are unsupported: {', '.join(missing)}",
        working_directory=request.working_directory,
        timeout_seconds=request.timeout_seconds,
        profiler_available=True,
        artifact_coverage_status=Rocprofv3ArtifactCoverageStatus.UNAVAILABLE,
        reason_codes=(COUNTER_REASON_UNSUPPORTED,),
        output_format="csv,rocpd",
    )


def _failed(
    request: Rocprofv3ProfileRequest,
    command: tuple[str, ...],
    reason_code: str,
    message: str,
    *,
    completed: subprocess.CompletedProcess[str] | None = None,
    stdout: str = "",
    stderr: str = "",
) -> Rocprofv3ProfileResult:
    return Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.FAILED,
        command=command,
        output_directory=request.output_directory,
        output_file=request.output_file,
        returncode=completed.returncode if completed else None,
        stdout=completed.stdout if completed else stdout,
        stderr=completed.stderr if completed else stderr,
        failed_reason=message,
        working_directory=request.working_directory,
        timeout_seconds=request.timeout_seconds,
        profiler_available=True,
        artifact_coverage_status=Rocprofv3ArtifactCoverageStatus.NONE,
        reason_codes=(reason_code,),
        output_format="csv,rocpd",
    )


__all__ = [
    "COUNTER_REASON_ARTIFACT_INCOMPLETE",
    "COUNTER_REASON_AVAIL_FAILED",
    "COUNTER_REASON_COLLECTED",
    "COUNTER_REASON_UNSUPPORTED",
    "collect_rocprofv3_counters",
]
