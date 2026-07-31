# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed evidence manifest for one performance workload."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.diagnostic_sidecar import (
    CurrentDiagnosticSidecarAuthority,
    DiagnosticSidecarStatus,
)
from sol_execbench.core.data.base_model import (
    StrictArtifactModel,
)
from sol_execbench.core.data.json_utils import load_json_file
from sol_execbench.core.integrity import (
    SHA256Digest,
    sha256_file,
    stable_json_checksum,
    verify_artifact_file,
)
from sol_execbench.core.integrity.schema_versions import (
    PERFORMANCE_EVIDENCE_MANIFEST_SCHEMA_VERSION,
)

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class PerformanceEvidenceArtifactKind(StrEnum):
    """Closed artifact kinds admitted by the performance builder."""

    TRACE = "trace"
    TIMING = "timing_evidence"
    ACCESS_PATTERN = "access_pattern_evidence"
    PROFILE_SUMMARY = "profile_summary"
    STATIC_EVIDENCE = "static_evidence"
    COUNTER_PROVENANCE = "counter_provenance"
    COUNTER_CSV = "counter_csv"
    ROCPD = "rocpd"
    ENVIRONMENT = "environment"
    REPLAY_EVIDENCE = "replay_evidence"


class PerformanceEvidenceArtifact(StrictArtifactModel):
    """One manifest-relative content-addressed artifact."""

    model_config = _MODEL_CONFIG

    kind: PerformanceEvidenceArtifactKind
    path: str
    sha256: SHA256Digest
    size_bytes: int = Field(ge=0)


class PerformanceRunIdentity(StrictArtifactModel):
    """Identity shared by every artifact in one diagnostic replay."""

    model_config = _MODEL_CONFIG

    run_id: SHA256Digest
    definition: str
    definition_sha256: SHA256Digest
    workload_uuid: str
    workload_sha256: SHA256Digest
    solution_sha256: SHA256Digest
    candidate_sha256: SHA256Digest
    gpu_architecture: str
    gpu_id: str | None = None
    gpu_bdf: str | None = None
    rocm_version: str | None = None
    compiler_version: str | None = None
    clock_mode: str
    power_profile: str | None = None
    timing_protocol: str


class PerformanceEvidenceManifest(CurrentDiagnosticSidecarAuthority):
    """Root manifest binding one workload to all diagnostic evidence."""

    model_config = _MODEL_CONFIG
    current_schema_version = PERFORMANCE_EVIDENCE_MANIFEST_SCHEMA_VERSION

    schema_version: Literal[
        "sol_execbench.performance_evidence_manifest.v4"
    ] = PERFORMANCE_EVIDENCE_MANIFEST_SCHEMA_VERSION
    status: DiagnosticSidecarStatus
    identity: PerformanceRunIdentity
    artifacts: list[PerformanceEvidenceArtifact] = Field(min_length=1)
    code_object_sha256: list[SHA256Digest] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def artifacts_are_unique(self) -> PerformanceEvidenceManifest:
        """Reject duplicate paths and contradictory aggregate status."""
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("performance evidence repeats artifact path")
        if (
            self.status is DiagnosticSidecarStatus.AVAILABLE
            and self.reason_codes
        ):
            raise ValueError("available evidence cannot carry reason_codes")
        return self

    def artifact(
        self,
        kind: PerformanceEvidenceArtifactKind,
    ) -> PerformanceEvidenceArtifact | None:
        """Return the unique artifact of one kind, when present."""
        matches = [item for item in self.artifacts if item.kind is kind]
        if len(matches) > 1:
            raise ValueError(f"multiple evidence artifacts for {kind}")
        return matches[0] if matches else None

    def artifacts_of_kind(
        self,
        kind: PerformanceEvidenceArtifactKind,
    ) -> list[PerformanceEvidenceArtifact]:
        """Return every artifact of a repeatable kind."""
        return [item for item in self.artifacts if item.kind is kind]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible manifest data."""
        return self.model_dump(mode="json")


def candidate_sha256(
    *,
    solution_sha256: str,
    compile_command_sha256: str | None,
    compiler_sha256: str | None,
    code_object_sha256: list[str],
) -> str:
    """Return the canonical candidate build identity."""
    return stable_json_checksum(
        {
            "solution_sha256": solution_sha256,
            "compile_command_sha256": compile_command_sha256,
            "compiler_sha256": compiler_sha256,
            "code_object_sha256": sorted(code_object_sha256),
        },
    )


def artifact_reference(
    *,
    kind: PerformanceEvidenceArtifactKind,
    path: Path,
    root: Path,
) -> PerformanceEvidenceArtifact:
    """Build a root-confined artifact reference."""
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            "performance artifact escapes manifest root"
        ) from error
    return PerformanceEvidenceArtifact(
        kind=kind,
        path=relative.as_posix(),
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def load_and_verify_performance_evidence_manifest(
    path: Path,
    *,
    require_complete: bool = False,
) -> PerformanceEvidenceManifest:
    """Load a root manifest and verify every cited artifact."""
    manifest = load_json_file(PerformanceEvidenceManifest, path)
    if (
        require_complete
        and manifest.status is not DiagnosticSidecarStatus.AVAILABLE
    ):
        raise ValueError(
            "performance evidence is incomplete: "
            + ",".join(manifest.reason_codes)
        )
    required = {
        PerformanceEvidenceArtifactKind.TRACE,
        PerformanceEvidenceArtifactKind.TIMING,
        PerformanceEvidenceArtifactKind.ACCESS_PATTERN,
        PerformanceEvidenceArtifactKind.PROFILE_SUMMARY,
        PerformanceEvidenceArtifactKind.STATIC_EVIDENCE,
    }
    if require_complete:
        required |= {
            PerformanceEvidenceArtifactKind.COUNTER_PROVENANCE,
            PerformanceEvidenceArtifactKind.COUNTER_CSV,
            PerformanceEvidenceArtifactKind.REPLAY_EVIDENCE,
        }
    observed = {artifact.kind for artifact in manifest.artifacts}
    missing = sorted(kind for kind in required - observed)
    if missing:
        raise ValueError(
            "performance evidence required artifacts missing: "
            + ",".join(missing)
        )
    for artifact in manifest.artifacts:
        verify_artifact_file(
            path.parent,
            artifact.path,
            expected_sha256=artifact.sha256,
            expected_size_bytes=artifact.size_bytes,
        )
    return manifest


__all__ = [
    "PerformanceEvidenceArtifact",
    "PerformanceEvidenceArtifactKind",
    "PerformanceEvidenceManifest",
    "PerformanceRunIdentity",
    "artifact_reference",
    "candidate_sha256",
    "load_and_verify_performance_evidence_manifest",
]
