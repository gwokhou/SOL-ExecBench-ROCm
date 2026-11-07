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

import shutil
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Literal

import zstandard
from pydantic import Field

from sol_execbench.core.bench.performance_model.builder import (
    SemanticCharacterizationLoader,
)
from sol_execbench.core.bench.performance_model.lifecycle.blob_store import (
    BlobStore,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
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
    DiagnosticPublicationLifecycleManifest,
    DiagnosticReleaseLifecycleManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.schema_versions import (
    DiagnosticLifecycleSchema,
    DiagnosticReleasePackageArtifactKind,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
    DiagnosticLifecycleParent,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import (
    publication_registry_dir,
    releases_dir,
)
from sol_execbench.core.bench.performance_model.publication import (
    DiagnosticPublicationProjection,
    SolarManifestProjectionVerifier,
    verify_diagnostic_publication_projection,
)
from sol_execbench.core.data.base_model import CurrentFrozenSchemaModel
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
)
from sol_execbench.core.integrity import (
    SHA256Digest,
    sha256_file,
    stable_json_checksum,
)
from sol_execbench.core.platform.rdna4_validation import (
    HardwareValidationBinding,
)


class DiagnosticReleaseArchive(CurrentFrozenSchemaModel):
    """One deterministic zstd archive of a verified publication."""

    current_schema_version = (
        DiagnosticLifecycleSchema.DIAGNOSTIC_RELEASE_PACKAGE
    )
    current_artifact_kind = DiagnosticReleasePackageArtifactKind.ARCHIVE

    schema_version: Literal[
        DiagnosticLifecycleSchema.DIAGNOSTIC_RELEASE_PACKAGE
    ] = DiagnosticLifecycleSchema.DIAGNOSTIC_RELEASE_PACKAGE
    artifact_kind: Literal[DiagnosticReleasePackageArtifactKind.ARCHIVE] = (
        DiagnosticReleasePackageArtifactKind.ARCHIVE
    )
    name: str = Field(min_length=1)
    sha256: SHA256Digest
    size_bytes: int = Field(ge=0)
    algorithm: Literal["zstd"] = "zstd"
    publication_manifest_sha256: SHA256Digest
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class DiagnosticReleaseAttestation(CurrentFrozenSchemaModel):
    """Release object binding a verified publication to its archive."""

    current_schema_version = (
        DiagnosticLifecycleSchema.DIAGNOSTIC_RELEASE_PACKAGE
    )
    current_artifact_kind = DiagnosticReleasePackageArtifactKind.ATTESTATION

    schema_version: Literal[
        DiagnosticLifecycleSchema.DIAGNOSTIC_RELEASE_PACKAGE
    ] = DiagnosticLifecycleSchema.DIAGNOSTIC_RELEASE_PACKAGE
    artifact_kind: Literal[DiagnosticReleasePackageArtifactKind.ATTESTATION] = (
        DiagnosticReleasePackageArtifactKind.ATTESTATION
    )
    release_id: SHA256Digest
    purpose: DiagnosticEvidencePurpose = DiagnosticEvidencePurpose.PRODUCTION
    publication_id: SHA256Digest
    archive: DiagnosticReleaseArchive
    uncompressed_size_bytes: int = Field(ge=0)
    case_count: int = Field(ge=220)
    inventory_sha256: SHA256Digest
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    hardware_validation: HardwareValidationBinding
    producer_version: str = PRODUCER_VERSION
    created_at: str = Field(min_length=1)
    retention_class: Literal["publication_release"] = "publication_release"
    diagnostic_only: Literal[True] = True
    score_authority: Literal[False] = False
    leaderboard_authority: Literal[False] = False


_MAX_COMPRESSED_BYTES = 1 << 30
_MAX_EXPANDED_BYTES = 10 << 30
_MAX_ARCHIVE_MEMBERS = 100_000


def _tar_info(path: Path, name: str) -> tarfile.TarInfo:
    """Build deterministic metadata for one regular file or directory."""
    info = tarfile.TarInfo(name)
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    if path.is_dir():
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
    elif path.is_file() and not path.is_symlink():
        info.type = tarfile.REGTYPE
        info.mode = 0o644
        info.size = path.stat().st_size
    else:
        raise ValueError(f"release tree contains a non-regular entry: {path}")
    return info


def _write_tar(source: Path, stream: BinaryIO) -> None:
    entries = (source, *sorted(source.rglob("*")))
    if len(entries) > _MAX_ARCHIVE_MEMBERS:
        raise ValueError("release tree exceeds the archive member limit")
    with tarfile.open(
        fileobj=stream, mode="w|", format=tarfile.PAX_FORMAT
    ) as tar:
        for path in entries:
            relative = path.relative_to(source.parent).as_posix()
            info = _tar_info(path, relative)
            if info.isreg():
                with path.open("rb") as source_file:
                    tar.addfile(info, source_file)
            else:
                tar.addfile(info)


def _run_deterministic_tar(
    source_dir: Path,
    archive_output: Path,
) -> None:
    """Create the byte-reproducible zstd archive of one publication tree."""
    source = source_dir.resolve()
    archive = archive_output.resolve()
    if not source.is_dir():
        raise ValueError(f"publication root is not a directory: {source}")
    if archive.exists():
        raise FileExistsError(f"release archive already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstandard.ZstdCompressor(
        level=19,
        threads=0,
        write_checksum=True,
        write_content_size=False,
        write_dict_id=False,
    )
    try:
        with (
            archive.open("xb") as destination,
            compressor.stream_writer(destination, closefd=False) as stream,
        ):
            _write_tar(source, stream)
    except Exception:
        archive.unlink(missing_ok=True)
        raise


def _extract_publication(
    archive_path: Path,
    unpack_root: Path,
) -> Path:
    """Safely unpack one bounded release and return its manifest path."""
    if archive_path.stat().st_size > _MAX_COMPRESSED_BYTES:
        raise ValueError("release archive exceeds the compressed size limit")
    unpack_root.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    total_size = 0
    decoder = zstandard.ZstdDecompressor()
    with (
        archive_path.open("rb") as source,
        decoder.stream_reader(source, closefd=False) as stream,
        tarfile.open(fileobj=stream, mode="r|") as archive,
    ):
        for index, member in enumerate(archive, start=1):
            if index > _MAX_ARCHIVE_MEMBERS:
                raise ValueError("release archive has too many members")
            target = _validated_member_path(member, unpack_root, seen)
            total_size += member.size
            if total_size > _MAX_EXPANDED_BYTES:
                raise ValueError("release archive exceeds expanded limit")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read archive member {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as destination:
                shutil.copyfileobj(extracted, destination)
    manifests = list(unpack_root.glob("*/publication.json"))
    if len(manifests) != 1:
        raise ValueError(
            "release archive must contain exactly one publication tree",
        )
    return manifests[0]


def _validated_member_path(
    member: tarfile.TarInfo,
    root: Path,
    seen: set[str],
) -> Path:
    """Reject traversal, duplicates, links, and special archive members."""
    path = Path(member.name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe release archive member: {member.name}")
    normalized = path.as_posix().rstrip("/")
    if normalized in seen:
        raise ValueError(f"duplicate release archive member: {member.name}")
    seen.add(normalized)
    if not (member.isdir() or member.isreg()):
        raise ValueError(f"unsupported release archive member: {member.name}")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"unsafe release archive member: {member.name}")
    return target


def package_diagnostic_publication(
    *,
    manifest_path: Path,
    archive_output: Path,
    attestation_output: Path,
    source_revision: str,
    hardware_validation: HardwareValidationBinding,
    semantic_loader: SemanticCharacterizationLoader,
    solar_verifier: SolarManifestProjectionVerifier,
    store_root_path: Path | None = None,
    purpose: DiagnosticEvidencePurpose = DiagnosticEvidencePurpose.PRODUCTION,
) -> DiagnosticReleaseAttestation:
    """Package one verified publication into a deterministic release object.

    Verification of the full publication tree precedes any archive creation;
    a release object can never be built from an unverified projection.
    """
    if hardware_validation.source_revision != source_revision:
        raise ValueError("hardware validation source revision mismatch")
    manifest = manifest_path.resolve()
    projection = verify_diagnostic_publication_projection(
        manifest,
        semantic_loader=semantic_loader,
        solar_verifier=solar_verifier,
    )
    if projection.purpose is not purpose:
        raise ValueError("release purpose does not match publication purpose")
    manifest_sha256 = sha256_file(manifest)
    pub_id = _publication_identity(
        projection=projection,
        manifest_sha256=manifest_sha256,
        source_revision=source_revision,
        purpose=purpose,
        store_root_path=store_root_path,
    )
    _run_deterministic_tar(
        manifest.parent,
        archive_output,
    )
    attestation = _build_attestation(
        projection=projection,
        publication_id=pub_id,
        publication_manifest_sha256=manifest_sha256,
        archive_output=archive_output,
        source_revision=source_revision,
        hardware_validation=hardware_validation,
        purpose=purpose,
    )
    atomic_write_json_value(
        attestation_output,
        attestation.model_dump(mode="json"),
    )
    if store_root_path is not None:
        BlobStore(store_root_path).put_file(
            manifest, expected_sha256=manifest_sha256
        )
        _write_release_manifest(
            store_root_path,
            attestation,
            attestation_sha256=sha256_file(attestation_output),
            archive_path=archive_output,
            attestation_path=attestation_output,
        )
    return attestation


def _publication_identity(
    *,
    projection: DiagnosticPublicationProjection,
    manifest_sha256: SHA256Digest,
    source_revision: str,
    purpose: DiagnosticEvidencePurpose,
    store_root_path: Path | None,
) -> SHA256Digest:
    """Resolve the governed publication node consumed by a release."""
    if store_root_path is None:
        missing_parent = "0" * 64
        return lifecycle_publication_id(
            acceptance_id=missing_parent,
            calibration_id=missing_parent,
            development_snapshot_id=missing_parent,
            model_build_id=missing_parent,
            source_corpus_sha256=projection.source_corpus_sha256,
            publication_manifest_sha256=manifest_sha256,
            uncompressed_size_bytes=projection.uncompressed_size_bytes,
            case_count=projection.case_count,
            source_revision=source_revision,
            purpose=purpose,
        )
    matches: list[DiagnosticPublicationLifecycleManifest] = []
    for path in sorted(
        publication_registry_dir(store_root_path).glob("*/manifest.json")
    ):
        candidate = load_json_file(DiagnosticPublicationLifecycleManifest, path)
        if (
            candidate.purpose is purpose
            and candidate.source_revision == source_revision
            and candidate.publication_manifest_sha256 == manifest_sha256
            and candidate.source_corpus_sha256
            == projection.source_corpus_sha256
        ):
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(
            "release candidate requires exactly one matching lifecycle "
            f"publication manifest, found {len(matches)}"
        )
    return matches[0].stage_id


def _build_attestation(
    *,
    projection: DiagnosticPublicationProjection,
    publication_id: SHA256Digest,
    publication_manifest_sha256: SHA256Digest,
    archive_output: Path,
    source_revision: str,
    hardware_validation: HardwareValidationBinding,
    purpose: DiagnosticEvidencePurpose,
) -> DiagnosticReleaseAttestation:
    archive_sha256 = sha256_file(archive_output)
    archive_size = archive_output.stat().st_size
    release_id = lifecycle_release_id(
        publication_id=publication_id,
        archive_sha256=archive_sha256,
        source_revision=source_revision,
        producer_version=PRODUCER_VERSION,
        archive_size_bytes=archive_size,
        hardware_validation_receipt_sha256=(hardware_validation.receipt_sha256),
        purpose=purpose,
    )
    inventory_sha256 = stable_json_checksum(
        [
            {
                "path": item.path,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in projection.artifacts
        ]
    )
    return DiagnosticReleaseAttestation(
        release_id=release_id,
        purpose=purpose,
        publication_id=publication_id,
        archive=DiagnosticReleaseArchive(
            name=archive_output.name,
            sha256=archive_sha256,
            size_bytes=archive_size,
            publication_manifest_sha256=publication_manifest_sha256,
            source_revision=source_revision,
        ),
        uncompressed_size_bytes=projection.uncompressed_size_bytes,
        case_count=projection.case_count,
        inventory_sha256=inventory_sha256,
        source_revision=source_revision,
        hardware_validation=hardware_validation,
        created_at=datetime.now(UTC).isoformat(),
    )


def verify_diagnostic_release_archive(
    *,
    archive_path: Path,
    semantic_loader: SemanticCharacterizationLoader,
    solar_verifier: SolarManifestProjectionVerifier,
    expected_sha256: str | None = None,
    unpack_root: Path | None = None,
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
        manifest_path = _extract_publication(archive, root)
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
    archive_path: Path,
    attestation_path: Path,
) -> None:
    directory = releases_dir(store_root_path) / attestation.release_id
    if directory.exists():
        raise FileExistsError(f"release object already exists: {directory}")
    blob_store = BlobStore(store_root_path)
    blob_store.put_file(
        archive_path, expected_sha256=attestation.archive.sha256
    )
    blob_store.put_file(attestation_path, expected_sha256=attestation_sha256)
    publication_manifest = (
        publication_registry_dir(store_root_path)
        / attestation.publication_id
        / "manifest.json"
    )
    if not publication_manifest.is_file():
        raise ValueError(
            "release candidate requires its lifecycle publication manifest"
        )
    publication_object_sha256 = blob_store.put_file(publication_manifest)
    directory.mkdir(parents=True)
    manifest = DiagnosticReleaseLifecycleManifest(
        stage=DiagnosticLifecycleStage.RELEASE,
        purpose=attestation.purpose,
        stage_id=attestation.release_id,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.PUBLICATION_RELEASE,
        source_revision=attestation.source_revision,
        parents=(
            DiagnosticLifecycleParent(
                stage=DiagnosticLifecycleStage.PUBLICATION,
                purpose=attestation.purpose,
                stage_id=attestation.publication_id,
                sha256=publication_object_sha256,
            ),
        ),
        exact_inventory=(
            DiagnosticLifecycleArtifact(
                relative_path=archive_path.name,
                sha256=attestation.archive.sha256,
                size_bytes=attestation.archive.size_bytes,
            ),
            DiagnosticLifecycleArtifact(
                relative_path=attestation_path.name,
                sha256=attestation_sha256,
                size_bytes=attestation_path.stat().st_size,
            ),
        ),
        created_at=attestation.created_at,
        archive_sha256=attestation.archive.sha256,
        archive_size_bytes=attestation.archive.size_bytes,
        attestation_sha256=attestation_sha256,
        hardware_validation_receipt_sha256=(
            attestation.hardware_validation.receipt_sha256
        ),
        published=False,
    )
    atomic_write_json_value(
        directory / "manifest.json",
        manifest.model_dump(mode="json"),
    )


__all__ = [
    "DiagnosticReleaseArchive",
    "DiagnosticReleaseAttestation",
    "package_diagnostic_publication",
    "verify_diagnostic_release_archive",
]
