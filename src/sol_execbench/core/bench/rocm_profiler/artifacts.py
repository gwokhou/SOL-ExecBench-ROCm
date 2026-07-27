# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ROCm profiler artifact discovery and classification."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from sol_execbench.core.bench.rocm_profiler.models import (
    _PROFILE_ARTIFACT_SUFFIXES,
    PROFILE_OUTPUT_DIR_NAMES,
    ROCPROF_REASON_ARTIFACTS_REGISTERED,
    ROCPROF_REASON_DIAGNOSTIC_LOG_REGISTERED,
    ROCPROF_REASON_NO_REGISTERED_ARTIFACTS,
    ROCPROF_REASON_PARTIAL_ARTIFACT_COVERAGE,
    ROCPROF_WARNING_INCOMPLETE_ARTIFACT_COVERAGE,
    ROCPROF_WARNING_NO_PROFILER_DATA_ARTIFACTS,
    Rocprofv3ArtifactCoverageStatus,
    Rocprofv3ArtifactKind,
    Rocprofv3ProfileArtifact,
    is_profiler_data_artifact,
)


def discover_rocprofv3_artifacts(
    output_directory: Path,
    output_file: str,
) -> tuple[Rocprofv3ProfileArtifact, ...]:
    """Register profiler artifacts produced for an output-file prefix."""
    artifacts: list[Rocprofv3ProfileArtifact] = []
    if not output_directory.exists():
        return ()

    for path in sorted(
        output_directory.rglob("*"),
        key=profile_artifact_sort_key,
    ):
        if not path.is_file():
            continue
        if not is_profile_artifact_candidate(
            path,
            output_directory,
            output_file,
        ):
            continue
        artifacts.append(
            Rocprofv3ProfileArtifact(
                path=path,
                kind=classify_profile_artifact(path),
                size_bytes=path.stat().st_size,
            ),
        )
    return tuple(artifacts)


def profile_artifact_sort_key(path: Path) -> tuple[str, ...]:
    """Return a deterministic path-component sort key."""
    return tuple(path.parts)


def is_profile_artifact_candidate(
    path: Path,
    output_directory: Path,
    output_file: str,
) -> bool:
    """Return whether a path belongs to the requested profiler output."""
    name = path.name
    if output_file and name.startswith(output_file):
        return True
    if not output_file:
        return False

    if not is_known_profile_artifact_name(path):
        return False

    try:
        relative_parts = path.relative_to(output_directory).parts[:-1]
    except ValueError:
        return False
    if not relative_parts:
        return is_unprefixed_profile_artifact_name(path)
    normalized_parts = {
        normalize_profile_artifact_token(part) for part in relative_parts
    }
    if output_file in relative_parts:
        return True
    return bool(normalized_parts & PROFILE_OUTPUT_DIR_NAMES)


def is_known_profile_artifact_name(path: Path) -> bool:
    """Return whether a filename uses a recognized profiler artifact form."""
    if path.suffix.lower() in _PROFILE_ARTIFACT_SUFFIXES:
        return True
    normalized_name = normalize_profile_artifact_token(path.name)
    return normalized_name in {
        "agent-info",
        "counter-collection",
        "kernel-trace",
        "metadata",
    }


def is_unprefixed_profile_artifact_name(path: Path) -> bool:
    """Return whether an unprefixed filename is a profiler artifact."""
    suffix = path.suffix.lower()
    normalized_name = normalize_profile_artifact_token(path.stem or path.name)
    if suffix in {".db", ".sqlite", ".sqlite3", ".rocpd", ".pftrace", ".otf2"}:
        return True
    if suffix == ".json":
        return normalized_name in {
            "agent-info",
            "metadata",
            "out-config",
            "results",
        }
    if suffix == ".csv":
        return any(
            token in normalized_name
            for token in (
                "agent",
                "counter",
                "hip",
                "hsa",
                "kernel",
                "marker",
                "memory",
                "rocdecode",
                "rocjpeg",
                "runtime",
                "trace",
            )
        )
    return False


def normalize_profile_artifact_token(value: str) -> str:
    """Normalize an artifact name or directory token for matching."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def profile_artifact_coverage_metadata(
    artifacts: Sequence[Rocprofv3ProfileArtifact],
    *,
    command_succeeded: bool,
) -> tuple[Rocprofv3ArtifactCoverageStatus, tuple[str, ...], tuple[str, ...]]:
    """Classify profiler artifact coverage and return reasons and warnings."""
    if not artifacts:
        return Rocprofv3ArtifactCoverageStatus.NONE, (), ()

    has_profiler_data_artifact = any(
        is_profiler_data_artifact(artifact) for artifact in artifacts
    )
    if command_succeeded and has_profiler_data_artifact:
        return (
            Rocprofv3ArtifactCoverageStatus.COMPLETE,
            (ROCPROF_REASON_ARTIFACTS_REGISTERED,),
            (),
        )

    if command_succeeded and any(
        artifact.kind is Rocprofv3ArtifactKind.DIAGNOSTIC_JSON
        for artifact in artifacts
    ):
        return (
            Rocprofv3ArtifactCoverageStatus.DIAGNOSTIC_LOGS_ONLY,
            (
                ROCPROF_REASON_NO_REGISTERED_ARTIFACTS,
                ROCPROF_REASON_DIAGNOSTIC_LOG_REGISTERED,
            ),
            (ROCPROF_WARNING_NO_PROFILER_DATA_ARTIFACTS,),
        )

    return (
        Rocprofv3ArtifactCoverageStatus.PARTIAL,
        (ROCPROF_REASON_PARTIAL_ARTIFACT_COVERAGE,),
        (ROCPROF_WARNING_INCOMPLETE_ARTIFACT_COVERAGE,),
    )


def classify_profile_artifact(path: Path) -> Rocprofv3ArtifactKind:
    """Classify a profiler artifact from its filename and suffix."""
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in {".db", ".sqlite", ".sqlite3", ".rocpd"}:
        return Rocprofv3ArtifactKind.ROCPD
    if suffix == ".csv":
        if "agent" in name:
            return Rocprofv3ArtifactKind.AGENT_INFO_CSV
        if "counter" in name:
            return Rocprofv3ArtifactKind.COUNTER_CSV
        return Rocprofv3ArtifactKind.TRACE_CSV
    if suffix == ".json":
        if "diagnostic" in name:
            return Rocprofv3ArtifactKind.DIAGNOSTIC_JSON
        return Rocprofv3ArtifactKind.METADATA_JSON
    if suffix == ".pftrace" or ("perfetto" in name and suffix == ".trace"):
        return Rocprofv3ArtifactKind.PERFETTO_TRACE
    if suffix == ".otf2":
        return Rocprofv3ArtifactKind.OTF2_TRACE
    return Rocprofv3ArtifactKind.OTHER


def profile_output_directory_listing(output_directory: Path) -> tuple[str, ...]:
    """Return a bounded deterministic listing of profiler output files."""
    if not output_directory.exists():
        return ()
    listing: list[str] = []
    for path in sorted(
        output_directory.rglob("*"),
        key=profile_artifact_sort_key,
    ):
        try:
            relative = path.relative_to(output_directory).as_posix()
        except ValueError:
            continue
        if path.is_dir():
            listing.append(f"{relative}/")
        elif path.is_file():
            listing.append(f"{relative}:{path.stat().st_size}")
    return tuple(listing[:200])
