# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ROCm profiler artifact discovery and classification."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from sol_execbench.core.bench.rocm_profiler_models import (
    ROCPROF_REASON_ARTIFACTS_REGISTERED,
    ROCPROF_REASON_DIAGNOSTIC_LOG_REGISTERED,
    ROCPROF_REASON_NO_REGISTERED_ARTIFACTS,
    ROCPROF_REASON_PARTIAL_ARTIFACT_COVERAGE,
    ROCPROF_WARNING_INCOMPLETE_ARTIFACT_COVERAGE,
    ROCPROF_WARNING_NO_PROFILER_DATA_ARTIFACTS,
    _NON_DATA_ARTIFACT_KINDS,
    _PROFILE_ARTIFACT_SUFFIXES,
    _PROFILE_OUTPUT_DIR_NAMES,
    Rocprofv3ProfileArtifact,
    has_profiler_data_artifact,
)


PROFILE_OUTPUT_DIR_NAMES = _PROFILE_OUTPUT_DIR_NAMES


def discover_rocprofv3_artifacts(
    output_directory: Path,
    output_file: str,
) -> tuple[Rocprofv3ProfileArtifact, ...]:
    """Register profiler artifacts produced for an output-file prefix."""
    artifacts: list[Rocprofv3ProfileArtifact] = []
    if not output_directory.exists():
        return ()

    for path in sorted(output_directory.rglob("*"), key=profile_artifact_sort_key):
        if not path.is_file():
            continue
        if not is_profile_artifact_candidate(path, output_directory, output_file):
            continue
        artifacts.append(
            Rocprofv3ProfileArtifact(
                path=path,
                kind=classify_profile_artifact(path),
                size_bytes=path.stat().st_size,
            )
        )
    return tuple(artifacts)


def profile_artifact_sort_key(path: Path) -> tuple[str, ...]:
    return tuple(path.parts)


def is_profile_artifact_candidate(
    path: Path,
    output_directory: Path,
    output_file: str,
) -> bool:
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
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def profile_artifact_coverage_metadata(
    artifacts: Sequence[Rocprofv3ProfileArtifact],
    *,
    command_succeeded: bool,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    if not artifacts:
        return "none", (), ()

    has_profiler_data_artifact = any(
        is_profiler_data_artifact(artifact) for artifact in artifacts
    )
    if command_succeeded and has_profiler_data_artifact:
        return "complete", (ROCPROF_REASON_ARTIFACTS_REGISTERED,), ()

    if command_succeeded and any(
        artifact.kind == "diagnostic_json" for artifact in artifacts
    ):
        return (
            "diagnostic_logs_only",
            (
                ROCPROF_REASON_NO_REGISTERED_ARTIFACTS,
                ROCPROF_REASON_DIAGNOSTIC_LOG_REGISTERED,
            ),
            (ROCPROF_WARNING_NO_PROFILER_DATA_ARTIFACTS,),
        )

    return (
        "partial",
        (ROCPROF_REASON_PARTIAL_ARTIFACT_COVERAGE,),
        (ROCPROF_WARNING_INCOMPLETE_ARTIFACT_COVERAGE,),
    )


def is_profiler_data_artifact(artifact: Rocprofv3ProfileArtifact) -> bool:
    return artifact.kind not in _NON_DATA_ARTIFACT_KINDS


def classify_profile_artifact(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in {".db", ".sqlite", ".sqlite3", ".rocpd"}:
        return "rocpd"
    if suffix == ".csv":
        if "agent" in name:
            return "agent_info_csv"
        if "counter" in name:
            return "counter_csv"
        return "trace_csv"
    if suffix == ".json":
        if "diagnostic" in name:
            return "diagnostic_json"
        return "metadata_json"
    if suffix == ".pftrace" or ("perfetto" in name and suffix == ".trace"):
        return "perfetto_trace"
    if suffix == ".otf2":
        return "otf2_trace"
    return "other"


def profile_output_directory_listing(output_directory: Path) -> tuple[str, ...]:
    if not output_directory.exists():
        return ()
    listing: list[str] = []
    for path in sorted(output_directory.rglob("*"), key=profile_artifact_sort_key):
        try:
            relative = path.relative_to(output_directory).as_posix()
        except ValueError:
            continue
        if path.is_dir():
            listing.append(f"{relative}/")
        elif path.is_file():
            listing.append(f"{relative}:{path.stat().st_size}")
    return tuple(listing[:200])


# Compatibility aliases for the old monolithic module's private helper names.
_profile_artifact_sort_key = profile_artifact_sort_key
_is_profile_artifact_candidate = is_profile_artifact_candidate
_is_known_profile_artifact_name = is_known_profile_artifact_name
_is_unprefixed_profile_artifact_name = is_unprefixed_profile_artifact_name
_normalize_profile_artifact_token = normalize_profile_artifact_token
_profile_artifact_coverage_metadata = profile_artifact_coverage_metadata
_is_profiler_data_artifact = is_profiler_data_artifact
_has_profiler_data_artifact = has_profiler_data_artifact
_classify_profile_artifact = classify_profile_artifact
_profile_output_directory_listing = profile_output_directory_listing
