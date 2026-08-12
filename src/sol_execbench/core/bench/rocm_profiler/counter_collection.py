# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Explicit rocprofv3 counter-mode collection lifecycle."""

from __future__ import annotations

import csv
import gzip
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import replace
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
    Rocprofv3ArtifactCoverageStatus,
    Rocprofv3ArtifactKind,
    Rocprofv3ProfileArtifact,
    Rocprofv3ProfileRequest,
    Rocprofv3ProfileResult,
    Rocprofv3ProfileStatus,
    Rocprofv3ReasonCode,
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
    SchemaVersion,
)
from sol_execbench.core.platform.runtime import (
    resolve_rocm_tool,
    resolve_tool_path,
)
from sol_execbench.core.process.environment import (
    ENV_SOL_EXECBENCH_COUNTER_REPLAY,
    ENV_SOL_EXECBENCH_REPLAY_PASS_INDEX,
)
from sol_execbench.core.text_utils import normalize_ascii_alnum, subprocess_text


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
        "gfx1200_v3.yaml",
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
        pmc_checks = _run_pmc_checks(
            request,
            run,
            str(avail_path),
            groups,
        )
        if isinstance(pmc_checks, Rocprofv3ProfileResult):
            return pmc_checks
        return _collect_selected(
            request,
            groups=groups,
            manifest_path=manifest_path,
            availability=availability,
            pmc_checks=pmc_checks,
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
            Rocprofv3ReasonCode.COUNTER_AVAILABILITY_FAILED,
            f"rocprofv3-avail timed out after {request.timeout_seconds} seconds",
            stdout=subprocess_text(error.stdout),
            stderr=subprocess_text(error.stderr),
        )
    if completed.returncode != 0:
        return _failed(
            request,
            tuple(command),
            Rocprofv3ReasonCode.COUNTER_AVAILABILITY_FAILED,
            f"rocprofv3-avail failed with exit code {completed.returncode}",
            completed=completed,
        )
    return completed


def _run_pmc_checks(
    request: Rocprofv3ProfileRequest,
    runner: ProfileRunner,
    executable: str,
    groups: Sequence[Sequence[str]],
) -> list[subprocess.CompletedProcess[str]] | Rocprofv3ProfileResult:
    completed_checks: list[subprocess.CompletedProcess[str]] = []
    for group in groups:
        command = [executable, "-d", "0", "pmc-check", *group]
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
                Rocprofv3ReasonCode.COUNTER_PMC_CHECK_FAILED,
                "rocprofv3-avail pmc-check timed out",
                stdout=subprocess_text(error.stdout),
                stderr=subprocess_text(error.stderr),
            )
        if completed.returncode != 0:
            return _failed(
                request,
                tuple(command),
                Rocprofv3ReasonCode.COUNTER_PMC_CHECK_FAILED,
                "rocprofv3-avail rejected a selected counter group",
                completed=completed,
            )
        completed_checks.append(completed)
    return completed_checks


def _collect_selected(
    request: Rocprofv3ProfileRequest,
    *,
    groups: list[list[str]],
    manifest_path: Path,
    availability: subprocess.CompletedProcess[str],
    pmc_checks: Sequence[subprocess.CompletedProcess[str]],
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
        pmc_checks=pmc_checks,
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
    pmc_checks: Sequence[subprocess.CompletedProcess[str]],
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
        "pmc_check_sha256": stable_json_checksum(
            [
                {
                    "command": list(item.args),
                    "returncode": item.returncode,
                    "stderr": item.stderr or "",
                    "stdout": item.stdout or "",
                }
                for item in pmc_checks
            ]
        ),
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
            "schema_version": SchemaVersion.ROCPROFV3_COUNTER_PROVENANCE,
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
    discovered = discover_rocprofv3_artifacts(
        request.output_directory,
        request.output_file,
    )
    artifacts = _candidate_replay_artifacts(discovered)
    artifacts = _compact_candidate_replay_artifacts(discovered, artifacts)
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
            (Rocprofv3ReasonCode.COUNTERS_COLLECTED, *coverage_reasons)
            if success
            else (
                Rocprofv3ReasonCode.COMMAND_FAILED,
                *(
                    (Rocprofv3ReasonCode.COUNTER_ARTIFACT_INCOMPLETE,)
                    if artifact_reasons
                    else ()
                ),
                *coverage_reasons,
            )
        ),
        warnings=warnings,
        output_format="csv",
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
        marker_artifacts = [
            artifact
            for artifact in _artifacts_by_pass(
                artifacts,
                Rocprofv3ArtifactKind.TRACE_CSV,
            ).get(pass_index, [])
            if "marker" in artifact.path.name.lower()
        ]
        if len(marker_artifacts) != 1:
            reasons.append(f"counter_pass_marker_count:{pass_index}")
        elif len(csv_artifacts) == 1:
            try:
                _select_counter_rows_in_marker_ranges(
                    csv_artifacts[0].path,
                    marker_artifacts[0].path,
                )
            except (OSError, ValueError):
                reasons.append(
                    f"counter_pass_marker_window_invalid:{pass_index}"
                )
    return reasons


def _candidate_replay_artifacts(
    artifacts: Sequence[Rocprofv3ProfileArtifact],
) -> tuple[Rocprofv3ProfileArtifact, ...]:
    """Select the ROCTx-marked candidate process while retaining metadata."""
    marker_processes = {
        (artifact.path.parent, _artifact_process_prefix(artifact.path))
        for artifact in artifacts
        if artifact.kind is Rocprofv3ArtifactKind.TRACE_CSV
        and "marker" in artifact.path.name.lower()
    }
    selected: list[Rocprofv3ProfileArtifact] = []
    process_kinds = {
        Rocprofv3ArtifactKind.AGENT_INFO_CSV,
        Rocprofv3ArtifactKind.COUNTER_CSV,
        Rocprofv3ArtifactKind.TRACE_CSV,
    }
    for artifact in artifacts:
        if artifact.kind not in process_kinds:
            selected.append(artifact)
            continue
        identity = (
            artifact.path.parent,
            _artifact_process_prefix(artifact.path),
        )
        if identity in marker_processes:
            selected.append(artifact)
    return tuple(selected)


def _compact_candidate_replay_artifacts(
    discovered: Sequence[Rocprofv3ProfileArtifact],
    selected: Sequence[Rocprofv3ProfileArtifact],
) -> tuple[Rocprofv3ProfileArtifact, ...]:
    """Keep candidate evidence only."""
    selected_paths = {artifact.path for artifact in selected}
    markers = {
        (
            artifact.path.parent,
            _artifact_process_prefix(artifact.path),
        ): artifact
        for artifact in selected
        if artifact.kind is Rocprofv3ArtifactKind.TRACE_CSV
        and "marker" in artifact.path.name.lower()
    }
    process_kinds = {
        Rocprofv3ArtifactKind.AGENT_INFO_CSV,
        Rocprofv3ArtifactKind.COUNTER_CSV,
        Rocprofv3ArtifactKind.TRACE_CSV,
    }
    compacted: list[Rocprofv3ProfileArtifact] = []
    for artifact in selected:
        identity = (
            artifact.path.parent,
            _artifact_process_prefix(artifact.path),
        )
        if artifact.kind is Rocprofv3ArtifactKind.COUNTER_CSV:
            marker = markers.get(identity)
            try:
                compacted.append(
                    _filter_counter_csv_to_marker_ranges(artifact, marker)
                )
            except (OSError, ValueError):
                compacted.append(artifact)
        else:
            compacted.append(artifact)
    for artifact in discovered:
        if (
            artifact.kind in process_kinds
            and artifact.path not in selected_paths
        ):
            artifact.path.unlink(missing_ok=True)
    return tuple(compacted)


def _filter_counter_csv_to_marker_ranges(
    artifact: Rocprofv3ProfileArtifact,
    marker: Rocprofv3ProfileArtifact | None,
) -> Rocprofv3ProfileArtifact:
    if marker is None:
        raise ValueError("candidate_marker_artifact_missing")
    fieldnames, rows = _select_counter_rows_in_marker_ranges(
        artifact.path,
        marker.path,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=artifact.path.parent,
            prefix=f".{artifact.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(artifact.path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return replace(artifact, size_bytes=artifact.path.stat().st_size)


def _select_counter_rows_in_marker_ranges(
    counter_path: Path,
    marker_path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    intervals = _candidate_marker_intervals(marker_path)
    with counter_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("counter_csv_header_missing")
        fieldnames = list(reader.fieldnames)
        normalized = {normalize_ascii_alnum(name): name for name in fieldnames}
        start_key = normalized.get("starttimestamp")
        end_key = normalized.get("endtimestamp")
        if start_key is None or end_key is None:
            raise ValueError("counter_csv_timestamp_columns_missing")
        selected: list[dict[str, str]] = []
        matched_intervals: set[int] = set()
        for row in reader:
            start = _parse_timestamp(row.get(start_key))
            end = _parse_timestamp(row.get(end_key))
            if end < start:
                raise ValueError("counter_csv_timestamp_order_invalid")
            for index, (marker_start, marker_end) in enumerate(intervals):
                if marker_start <= start and end <= marker_end:
                    selected.append(row)
                    matched_intervals.add(index)
                    break
    if len(matched_intervals) != len(intervals):
        raise ValueError("candidate_marker_interval_without_dispatch")
    return fieldnames, selected


def _candidate_marker_intervals(path: Path) -> list[tuple[int, int]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("marker_csv_header_missing")
        normalized = {
            normalize_ascii_alnum(name): name for name in reader.fieldnames
        }
        domain_key = normalized.get("domain")
        function_key = normalized.get("function") or normalized.get("name")
        start_key = normalized.get("starttimestamp")
        end_key = normalized.get("endtimestamp")
        if None in (domain_key, function_key, start_key, end_key):
            raise ValueError("marker_csv_required_columns_missing")
        intervals = []
        for row in reader:
            domain = (row.get(domain_key) or "").strip().upper()
            function = (row.get(function_key) or "").strip()
            if domain != "MARKER_CORE_RANGE_API":
                continue
            if not function.startswith("sol_execbench/"):
                continue
            marker_prefix, separator, iteration = function.rpartition(
                "/iteration/"
            )
            if not separator or not marker_prefix or not iteration.isdecimal():
                continue
            start = _parse_timestamp(row.get(start_key))
            end = _parse_timestamp(row.get(end_key))
            if end < start:
                raise ValueError("marker_csv_timestamp_order_invalid")
            intervals.append((start, end))
    if not intervals:
        raise ValueError("candidate_marker_intervals_missing")
    return intervals


def _parse_timestamp(value: str | None) -> int:
    if value is None:
        raise ValueError("timestamp_missing")
    try:
        return int(value.strip())
    except ValueError as error:
        raise ValueError("timestamp_invalid") from error


def _compress_rocpd(
    artifact: Rocprofv3ProfileArtifact,
) -> Rocprofv3ProfileArtifact:
    source = artifact.path
    destination = source.with_name(f"{source.name}.gz")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if destination.exists() or temporary.exists():
        raise ValueError("compressed_rocpd_output_exists")
    try:
        with (
            source.open("rb") as input_handle,
            temporary.open("xb") as raw_output,
            gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=1,
                fileobj=raw_output,
                mtime=0,
            ) as output_handle,
        ):
            shutil.copyfileobj(input_handle, output_handle)
        temporary.replace(destination)
        source.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return replace(
        artifact,
        path=destination,
        size_bytes=destination.stat().st_size,
    )


def _artifact_process_prefix(path: Path) -> str:
    """Return rocprofv3's process-scoped filename prefix."""
    return path.name.split("_", maxsplit=1)[0]


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
        reason_codes=(Rocprofv3ReasonCode.COMMAND_TIMEOUT,),
        output_format="csv",
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
        reason_codes=(Rocprofv3ReasonCode.UNAVAILABLE,),
        output_format="csv",
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
        reason_codes=(Rocprofv3ReasonCode.REQUIRED_COUNTERS_UNSUPPORTED,),
        output_format="csv",
    )


def _failed(
    request: Rocprofv3ProfileRequest,
    command: tuple[str, ...],
    reason_code: Rocprofv3ReasonCode,
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
        output_format="csv",
    )


__all__ = [
    "collect_rocprofv3_counters",
]
