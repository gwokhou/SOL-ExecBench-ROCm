# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Governed deterministic release archive packaging.

A release object is created only from a verified publication projection. The
archive is byte-for-byte reproducible (``tar --sort=name --mtime=@0 ...``),
and the attestation binds the publication, archive, inventory, source
revision, and producer version. This module never touches an official-score
release bundle; it only packages diagnostic publication projections.
"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field

from sol_execbench.core.bench.performance_model.builder import (
    SemanticCharacterizationLoader,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticLifecycleStage,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
)
from sol_execbench.core.bench.performance_model.lifecycle.identity import (
    publication_id as lifecycle_publication_id,
    release_id as lifecycle_release_id,
)
from sol_execbench.core.bench.performance_model.lifecycle.models import (
    PRODUCER_VERSION,
    DiagnosticReleaseLifecycleManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import (
    releases_dir,
)
from sol_execbench.core.bench.performance_model.publication import (
    DiagnosticPublicationProjection,
    SolarManifestProjectionVerifier,
    verify_diagnostic_publication_projection,
)
from sol_execbench.core.data.base_model import CurrentFrozenSchemaModel
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import (
    SHA256Digest,
    sha256_file,
    stable_json_checksum,
)
from sol_execbench.core.integrity.schema_versions import SchemaVersion


class DiagnosticReleaseArchive(CurrentFrozenSchemaModel):
    """One deterministic zstd archive of a verified publication."""

    current_schema_version = SchemaVersion.DIAGNOSTIC_RELEASE_ARCHIVE

    schema_version: Literal[SchemaVersion.DIAGNOSTIC_RELEASE_ARCHIVE] = (
        SchemaVersion.DIAGNOSTIC_RELEASE_ARCHIVE
    )
    name: str = Field(min_length=1)
    sha256: SHA256Digest
    size_bytes: int = Field(ge=0)
    algorithm: Literal["zstd"] = "zstd"
    publication_manifest_sha256: SHA256Digest
    source_revision: str = Field(min_length=1)


class DiagnosticReleaseAttestation(CurrentFrozenSchemaModel):
    """Release object binding a verified publication to its archive."""

    current_schema_version = SchemaVersion.DIAGNOSTIC_RELEASE_ATTESTATION

    schema_version: Literal[SchemaVersion.DIAGNOSTIC_RELEASE_ATTESTATION] = (
        SchemaVersion.DIAGNOSTIC_RELEASE_ATTESTATION
    )
    release_id: SHA256Digest
    publication_id: SHA256Digest
    archive: DiagnosticReleaseArchive
    uncompressed_size_bytes: int = Field(ge=0)
    case_count: int = Field(ge=220)
    inventory_sha256: SHA256Digest
    source_revision: str = Field(min_length=1)
    producer_version: str = PRODUCER_VERSION
    created_at: str = Field(min_length=1)
    retention_class: Literal["publication_release"] = "publication_release"
    diagnostic_only: Literal[True] = True
    score_authority: Literal[False] = False
    leaderboard_authority: Literal[False] = False


class TarRunner(Protocol):
    """Boundary adapter for deterministic tar invocations."""

    def __call__(self, argv: list[str], *, cwd: Path | None = None) -> None:
        """Run ``argv`` and raise ``ValueError`` on any failure."""
        ...


def _system_tar(argv: list[str], *, cwd: Path | None = None) -> None:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ValueError(f"deterministic tar failed: {detail}")


def _run_deterministic_tar(
    source_dir: Path,
    archive_output: Path,
    *,
    tar_runner: TarRunner,
) -> None:
    """Create the byte-reproducible zstd archive of one publication tree."""
    source = source_dir.resolve()
    archive = archive_output.resolve()
    if not source.is_dir():
        raise ValueError(f"publication root is not a directory: {source}")
    if archive.exists():
        raise FileExistsError(f"release archive already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "tar",
        "--sort=name",
        "--mtime=@0",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        "--zstd",
        "-cf",
        str(archive),
        "-C",
        str(source.parent),
        source.name,
    ]
    tar_runner(argv, cwd=source.parent)


def _extract_publication(
    archive_path: Path,
    unpack_root: Path,
    *,
    tar_runner: TarRunner,
) -> Path:
    """Unpack one release archive and return its publication manifest path."""
    unpack_root.mkdir(parents=True, exist_ok=True)
    tar_runner(
        [
            "tar",
            "--zstd",
            "-xf",
            str(archive_path.resolve()),
            "-C",
            str(unpack_root.resolve()),
        ],
    )
    manifests = list(unpack_root.glob("*/publication.json"))
    if len(manifests) != 1:
        raise ValueError(
            "release archive must contain exactly one publication tree",
        )
    return manifests[0]


def package_diagnostic_publication(
    *,
    manifest_path: Path,
    archive_output: Path,
    attestation_output: Path,
    source_revision: str,
    semantic_loader: SemanticCharacterizationLoader,
    solar_verifier: SolarManifestProjectionVerifier,
    tar_runner: TarRunner = _system_tar,
    store_root_path: Path | None = None,
) -> DiagnosticReleaseAttestation:
    """Package one verified publication into a deterministic release object.

    Verification of the full publication tree precedes any archive creation;
    a release object can never be built from an unverified projection.
    """
    manifest = manifest_path.resolve()
    projection = verify_diagnostic_publication_projection(
        manifest,
        semantic_loader=semantic_loader,
        solar_verifier=solar_verifier,
    )
    manifest_sha256 = sha256_file(manifest)
    pub_id = lifecycle_publication_id(
        source_corpus_sha256=projection.source_corpus_sha256,
        publication_manifest_sha256=manifest_sha256,
    )
    _run_deterministic_tar(
        manifest.parent,
        archive_output,
        tar_runner=tar_runner,
    )
    archive_sha256 = sha256_file(archive_output)
    archive_size = archive_output.stat().st_size
    rel_id = lifecycle_release_id(
        publication_id=pub_id,
        archive_sha256=archive_sha256,
        source_revision=source_revision,
        producer_version=PRODUCER_VERSION,
    )
    inventory_sha256 = stable_json_checksum(
        [
            {
                "path": item.path,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in projection.artifacts
        ],
    )
    attestation = DiagnosticReleaseAttestation(
        release_id=rel_id,
        publication_id=pub_id,
        archive=DiagnosticReleaseArchive(
            name=archive_output.name,
            sha256=archive_sha256,
            size_bytes=archive_size,
            publication_manifest_sha256=manifest_sha256,
            source_revision=source_revision,
        ),
        uncompressed_size_bytes=projection.uncompressed_size_bytes,
        case_count=projection.case_count,
        inventory_sha256=inventory_sha256,
        source_revision=source_revision,
        created_at=datetime.now(UTC).isoformat(),
    )
    atomic_write_json_value(
        attestation_output,
        attestation.model_dump(mode="json"),
    )
    if store_root_path is not None:
        _write_release_manifest(
            store_root_path,
            attestation,
            attestation_sha256=sha256_file(attestation_output),
        )
    return attestation


def verify_diagnostic_release_archive(
    *,
    archive_path: Path,
    semantic_loader: SemanticCharacterizationLoader,
    solar_verifier: SolarManifestProjectionVerifier,
    expected_sha256: str | None = None,
    unpack_root: Path | None = None,
    tar_runner: TarRunner = _system_tar,
) -> DiagnosticPublicationProjection:
    """Verify a downloaded release archive against its publication contract."""
    archive = archive_path.resolve()
    actual = sha256_file(archive)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("release archive SHA-256 does not match expectation")
    temporary: tempfile.TemporaryDirectory[str] | None = None
    root = unpack_root.resolve() if unpack_root is not None else None
    if root is None:
        temporary = tempfile.TemporaryDirectory(prefix="release-verify-")
        root = Path(temporary.name)
    try:
        manifest_path = _extract_publication(
            archive, root, tar_runner=tar_runner
        )
        return verify_diagnostic_publication_projection(
            manifest_path,
            semantic_loader=semantic_loader,
            solar_verifier=solar_verifier,
        )
    finally:
        if temporary is not None:
            temporary.cleanup()


def _write_release_manifest(
    store_root_path: Path,
    attestation: DiagnosticReleaseAttestation,
    *,
    attestation_sha256: SHA256Digest,
) -> None:
    directory = releases_dir(store_root_path) / attestation.release_id
    if directory.exists():
        raise FileExistsError(f"release object already exists: {directory}")
    directory.mkdir(parents=True)
    manifest = DiagnosticReleaseLifecycleManifest(
        stage=DiagnosticLifecycleStage.RELEASE,
        stage_id=attestation.release_id,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.PUBLICATION_RELEASE,
        source_revision=attestation.source_revision,
        parents=(),
        created_at=attestation.created_at,
        archive_sha256=attestation.archive.sha256,
        archive_size_bytes=attestation.archive.size_bytes,
        attestation_sha256=attestation_sha256,
        published=False,
    )
    atomic_write_json_value(
        directory / "manifest.json",
        manifest.model_dump(mode="json"),
    )


__all__ = [
    "DiagnosticReleaseArchive",
    "DiagnosticReleaseAttestation",
    "TarRunner",
    "package_diagnostic_publication",
    "verify_diagnostic_release_archive",
]
