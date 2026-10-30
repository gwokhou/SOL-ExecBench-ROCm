# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Projection configuration and qualification independently of saved scores."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from solar.analysis.orojenesis.configuration import (
    orojenesis_mapper_thread_count,
)
from solar.analysis.orojenesis.runner import OrojenesisRunner
from solar.artifacts import stable_json_checksum
from solar.artifacts.bundle import read_regular_file
from solar.artifacts.document import parse_yaml_artifact
from solar.reuse_contracts import ProfileScenario, ProjectionRequest
from solar.rocm.architecture import ArchitectureProfile

MAX_PROFILE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class ProjectionConfiguration:
    """Validated profile, mapper and explicit behavior-changing inputs."""

    profile: ArchitectureProfile
    runner: OrojenesisRunner | None
    requested_sha256: str
    scenario: ProfileScenario
    audit_verified: bool
    identity: dict[str, object]


def configure_projection(
    request: ProjectionRequest,
    source_architecture: Mapping[str, object],
) -> ProjectionConfiguration:
    """Drop inherited audit claims when parameters or qualification inputs change."""
    profile = load_projection_profile(request.architecture)
    requested_sha256 = stable_json_checksum(profile.to_dict())
    changed = (
        request.hypothetical
        or profile.qualification_sha256()
        != ArchitectureProfile.load(source_architecture).qualification_sha256()
    )
    scenario = (
        ProfileScenario.HYPOTHETICAL if changed else ProfileScenario.IDENTICAL
    )
    if changed:
        profile = replace(
            profile,
            audit_evidence={
                "status": "unavailable",
                "reason_code": "hypothetical_profile",
            },
        )
    if request.require_verified_audit:
        profile.require_verified_audit_evidence()
    runner = (
        OrojenesisRunner(
            request.orojenesis_home,
            timeout_seconds=request.mapper_timeout_seconds,
        )
        if request.orojenesis_home is not None or request.require_orojenesis
        else None
    )
    identity: dict[str, object] = {
        "architecture": profile.to_dict(),
        "requested_architecture_sha256": requested_sha256,
        "scenario": scenario.value,
        "require_verified_audit": request.require_verified_audit,
        "require_orojenesis": request.require_orojenesis,
        "mapper_timeout_seconds": request.mapper_timeout_seconds,
        "mapper_threads": orojenesis_mapper_thread_count(),
        "mapper": runner.toolchain_identity if runner is not None else None,
        "output_bytes": request.output_bytes,
        "restricted": request.restricted.model_dump(mode="json")
        if request.restricted is not None
        else None,
    }
    return ProjectionConfiguration(
        profile,
        runner,
        requested_sha256,
        scenario,
        request.require_verified_audit,
        identity,
    )


def load_projection_profile(
    architecture: str | Path | dict[str, object],
) -> ArchitectureProfile:
    """Normalize an architecture at the bounded external-profile boundary."""
    if isinstance(architecture, (str, Path)) and Path(architecture).is_file():
        path = Path(architecture)
        document = parse_yaml_artifact(
            path,
            read_regular_file(path, max_bytes=MAX_PROFILE_BYTES),
            max_bytes=MAX_PROFILE_BYTES,
        ).data
        return ArchitectureProfile.load(document)
    return ArchitectureProfile.load(architecture)
