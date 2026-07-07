# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Profile sidecar helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from ..core.bench.profile_summary import (
    ProfileSummaryArtifactCitation,
    build_profile_summary_sidecar,
    profile_summary_artifact_citation_from_path,
)
from ..core.bench.rocm_profiler import Rocprofv3ProfileResult
from ..core.checksums import sha256_file
from ..core.runtime_evidence import write_json_payload

console = Console(stderr=True)


def _profile_output_directory(output_file: Path | None, staging_dir: Path) -> Path:
    """Return the profiler artifact directory for an evaluation run."""

    if output_file is not None:
        return output_file.with_name(f"{output_file.name}.rocprofv3")
    return staging_dir / "rocprofv3"


def _profile_sidecar_path(
    output_file: Path | None,
    profile_result: Rocprofv3ProfileResult,
) -> Path:
    """Return the profile metadata sidecar path for an evaluation run."""

    if output_file is not None:
        return output_file.with_name(f"{output_file.name}.profile.json")
    return profile_result.output_directory / "profile.json"


def _write_profile_sidecar(
    output_file: Path | None,
    profile_result: Rocprofv3ProfileResult | None,
) -> Path | None:
    """Write optional profiler metadata without changing trace JSONL."""

    if profile_result is None:
        return None

    sidecar_path = _profile_sidecar_path(output_file, profile_result)
    try:
        write_json_payload(sidecar_path, profile_result.to_dict())
        console.print(f"[green]Saved profiling metadata to {sidecar_path}[/green]")
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Profiling metadata skipped: {exc}[/yellow]")
        return None


def _profile_summary_sidecar_path(output_file: Path | None) -> Path | None:
    """Return the optional profile summary sidecar path for a trace output."""

    if output_file is None:
        return None
    return output_file.with_name(f"{output_file.name}.profile-summary.json")


def _write_profile_summary_sidecar(
    output_file: Path | None,
    profile_result: Rocprofv3ProfileResult | None,
    *,
    profile_sidecar_path: Path | None = None,
    run_id: str | None = None,
) -> Path | None:
    """Write optional normalized profile summary without changing trace JSONL."""

    sidecar_path = _profile_summary_sidecar_path(output_file)
    if sidecar_path is None or output_file is None:
        return None
    try:
        trace_sha256 = sha256_file(output_file) if output_file.is_file() else None
        if run_id is None:
            run_id = trace_sha256
        sidecar = build_profile_summary_sidecar(
            profile_result=profile_result,
            trace_path=str(output_file),
            run_id=run_id,
            artifact_citations=_profile_summary_artifact_citations(
                output_file=output_file,
                profile_result=profile_result,
                profile_sidecar_path=profile_sidecar_path,
                trace_sha256=trace_sha256,
            ),
        )
        write_json_payload(sidecar_path, sidecar)
        console.print(f"[green]Saved profile summary to {sidecar_path}[/green]")
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Profile summary skipped: {exc}[/yellow]")
        return None


def _profile_summary_artifact_citations(
    *,
    output_file: Path,
    profile_result: Rocprofv3ProfileResult | None,
    profile_sidecar_path: Path | None,
    trace_sha256: str | None = None,
) -> list[ProfileSummaryArtifactCitation]:
    """Return compact citations for profile-summary evidence inputs."""

    citations = [
        profile_summary_artifact_citation_from_path(
            kind="trace",
            label="canonical_trace_jsonl",
            path=output_file,
            sha256=trace_sha256,
        )
    ]
    if profile_sidecar_path is not None:
        citations.append(
            profile_summary_artifact_citation_from_path(
                kind="profile_metadata",
                label="rocprofv3_profile_metadata",
                path=profile_sidecar_path,
            )
        )
    if profile_result is not None:
        for artifact in profile_result.artifacts:
            citations.append(
                profile_summary_artifact_citation_from_path(
                    kind="profiler_artifact",
                    label=artifact.kind,
                    path=artifact.path,
                    status=profile_result.status,
                    size_bytes=artifact.size_bytes,
                )
            )
    return citations
