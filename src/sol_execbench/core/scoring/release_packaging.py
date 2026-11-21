# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Governed deterministic packaging of an official-score release.

A score-release object is created only from an already-verified release bundle.
``package_score_release`` first runs the full fail-closed verifier
(``verify_and_score_release``), then collects the exact transitively-referenced
evidence set (the same files the verifier re-checks) and writes a
byte-reproducible zstd archive plus a content-addressed attestation that binds
the bundle, archive, inventory, source revision, and reproduced official score.

``verify_score_release_archive`` is the inverse: it optional-checks the archive
SHA-256, extracts it, locates the release bundle, and re-runs
``verify_and_score_release`` to reproduce the official score. Together the two
functions (exposed as ``score release-package`` / ``score release-verify``) form
the no-LLM distribution closed loop, mirroring the diagnostic release packaging
in ``sol_execbench.core.bench.performance_model.release.packaging`` but with a
score-authority attestation contract.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Literal, Protocol, cast

import zstandard
from pydantic import Field

from sol_execbench.core.data.base_model import CurrentFrozenSchemaModel
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import (
    SHA256Digest,
    sha256_file,
    stable_json_checksum,
    validate_relative_artifact_path,
    verify_artifact_file,
)
from sol_execbench.core.platform.rdna4_validation import (
    HardwareValidationBinding,
)
from sol_execbench.core.process.logs import redacted_text_tail
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from sol_execbench.core.scoring.release_builders import artifact_reference
from sol_execbench.core.scoring.release_models import (
    ArtifactReference,
    BaselineStatement,
    CandidateStatement,
    ReleaseBundle,
    ReleaseModel,
    SolarIndexStatement,
)
from sol_execbench.core.scoring.release_verifier import (
    OfficialScoreResult,
    verify_and_score_release,
)
from sol_execbench.core.scoring.schema_versions import (
    ReleaseArtifactSchema,
    ReleasePackageArtifactKind,
)
from sol_execbench.core.solar_bridge.models import SolarRequestManifest

# Fixed archive root name so the produced archive is byte-identical across hosts
# and independent of the source workspace directory name.
_ARCHIVE_ROOT_NAME = "release"
_BUNDLE_FILENAME = "release-bundle.json"
_MAX_COMPRESSED_BYTES = 1 << 30
_MAX_EXPANDED_BYTES = 2 << 30
_MAX_ARCHIVE_MEMBERS = 100_000
_TAR_TIMEOUT_SECONDS = 300.0
_COPY_CHUNK_BYTES = 65_536


class ScoreReleaseArchive(CurrentFrozenSchemaModel):
    """One deterministic zstd archive of a verified score release."""

    current_schema_version = ReleaseArtifactSchema.RELEASE_PACKAGE
    current_artifact_kind = ReleasePackageArtifactKind.ARCHIVE

    schema_version: Literal[ReleaseArtifactSchema.RELEASE_PACKAGE] = (
        ReleaseArtifactSchema.RELEASE_PACKAGE
    )
    artifact_kind: Literal[ReleasePackageArtifactKind.ARCHIVE] = (
        ReleasePackageArtifactKind.ARCHIVE
    )
    name: str = Field(min_length=1)
    sha256: SHA256Digest
    size_bytes: int = Field(ge=0)
    algorithm: Literal["zstd"] = "zstd"
    bundle_sha256: SHA256Digest
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class ScoreReleaseAttestation(CurrentFrozenSchemaModel):
    """Release object binding a verified score-release bundle to its archive."""

    current_schema_version = ReleaseArtifactSchema.RELEASE_PACKAGE
    current_artifact_kind = ReleasePackageArtifactKind.ATTESTATION

    schema_version: Literal[ReleaseArtifactSchema.RELEASE_PACKAGE] = (
        ReleaseArtifactSchema.RELEASE_PACKAGE
    )
    artifact_kind: Literal[ReleasePackageArtifactKind.ATTESTATION] = (
        ReleasePackageArtifactKind.ATTESTATION
    )
    release_id: SHA256Digest
    bundle_sha256: SHA256Digest
    archive: ScoreReleaseArchive
    inventory_sha256: SHA256Digest
    uncompressed_size_bytes: int = Field(ge=0)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    hardware_validation: HardwareValidationBinding
    official_score: float = Field(ge=0, le=1)
    scored_workloads: int = Field(ge=1)
    baseline_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    score_authority: Literal[True] = True
    diagnostic_only: Literal[False] = False


class TarRunner(Protocol):
    """Boundary adapter for deterministic tar invocations."""

    def __call__(self, argv: list[str], *, cwd: Path | None = None) -> None:
        """Run ``argv`` and raise ``ValueError`` on any failure."""
        ...


class _BoundedBinaryReader:
    """Stream reader wrapper enforcing an upper bound on decompressed bytes."""

    def __init__(self, stream: BinaryIO, max_bytes: int) -> None:
        self._stream = stream
        self._max_bytes = max_bytes
        self._bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._stream.read(size)
        self._bytes_read += len(chunk)
        if self._bytes_read > self._max_bytes:
            raise ValueError("release archive exceeds expanded limit")
        return chunk


def _system_tar(argv: list[str], *, cwd: Path | None = None) -> None:
    if sys.platform != "linux":
        raise RuntimeError(
            "official score release packaging requires Linux and GNU tar",
        )
    try:
        completed = run_in_process_group_bounded(
            argv,
            cwd=cwd,
            timeout=_TAR_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("deterministic tar timed out") from exc
    if completed.returncode != 0:
        detail = redacted_text_tail(
            (completed.stderr or completed.stdout or "").strip()
        )
        raise ValueError(f"deterministic tar failed: {detail}")


def _deterministic_zstd_tar(
    source_dir: Path,
    archive_output: Path,
    *,
    tar_runner: TarRunner,
) -> None:
    """Create the byte-reproducible zstd archive of one staged release tree."""
    source = source_dir.resolve()
    archive = archive_output.resolve()
    if not source.is_dir():
        raise ValueError(f"release staging root is not a directory: {source}")
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


def _reject_symlink_walk(root: Path, relative_parts: tuple[str, ...]) -> None:
    if root.is_symlink():
        raise ValueError(f"release root must not be a symlink: {root}")
    current = root
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"release path traverses a symlink: {current}")


def _validated_tar_member(
    member: tarfile.TarInfo,
    unpack_root: Path,
    seen: set[str],
) -> Path:
    raw = member.name
    if (
        "\x00" in raw
        or "\\" in raw
        or raw.startswith("/")
        or not raw.rstrip("/")
    ):
        raise ValueError(f"unsafe release archive member: {raw}")
    parts = tuple(raw.rstrip("/").split("/"))
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe release archive member: {raw}")
    if (
        member.issym()
        or member.islnk()
        or not (member.isdir() or member.isreg())
    ):
        raise ValueError(f"unsupported release archive member: {raw}")
    if member.size < 0:
        raise ValueError(f"invalid release archive member size: {raw}")
    normalized = "/".join(parts)
    if normalized in seen:
        raise ValueError(f"duplicate release archive member: {raw}")
    seen.add(normalized)
    _reject_symlink_walk(unpack_root, parts[:-1])
    target = unpack_root.joinpath(*parts)
    if target.is_symlink() or not target.resolve().is_relative_to(
        unpack_root.resolve()
    ):
        raise ValueError(f"unsafe release archive member: {raw}")
    return target


def _copy_tar_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    target: Path,
) -> None:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"cannot read archive member: {member.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("xb") as destination:
        while chunk := extracted.read(_COPY_CHUNK_BYTES):
            written += len(chunk)
            if written > member.size:
                raise ValueError(f"archive member size mismatch: {member.name}")
            destination.write(chunk)
    if written != member.size:
        raise ValueError(f"archive member truncated: {member.name}")


def _extract_zstd_tar_stream(archive_path: Path, unpack_root: Path) -> None:
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("release archive must be a regular file")
    if archive_path.stat().st_size > _MAX_COMPRESSED_BYTES:
        raise ValueError("release archive exceeds the compressed size limit")
    seen: set[str] = set()
    total_size = 0
    decoder = zstandard.ZstdDecompressor()
    try:
        with (
            archive_path.open("rb") as source,
            decoder.stream_reader(source, closefd=False) as zstd_stream,
        ):
            bounded = _BoundedBinaryReader(zstd_stream, _MAX_EXPANDED_BYTES)
            with tarfile.open(
                fileobj=cast(BinaryIO, bounded),
                mode="r|",
            ) as archive:
                for index, member in enumerate(archive, start=1):
                    if index > _MAX_ARCHIVE_MEMBERS:
                        raise ValueError("release archive has too many members")
                    target = _validated_tar_member(member, unpack_root, seen)
                    total_size += member.size
                    if total_size > _MAX_EXPANDED_BYTES:
                        raise ValueError(
                            "release archive exceeds expanded limit"
                        )
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        _copy_tar_member(archive, member, target)
    except (zstandard.ZstdError, tarfile.TarError, EOFError) as exc:
        raise ValueError("invalid release archive") from exc


def _scan_unpacked_tree(unpack_root: Path) -> tuple[list[Path], list[Path]]:
    if unpack_root.is_symlink() or not unpack_root.is_dir():
        raise ValueError("unpack root must be a regular directory")
    resolved_root = unpack_root.resolve()
    files: list[Path] = []
    directories: list[Path] = []
    member_count = 0
    expanded_bytes = 0
    for dirpath, dirnames, filenames in os.walk(unpack_root, followlinks=False):
        current_dir = Path(dirpath)
        if current_dir.is_symlink() or not current_dir.resolve().is_relative_to(
            resolved_root
        ):
            raise ValueError("release archive contains an unsafe directory")
        directories.append(current_dir)
        for name in dirnames:
            entry = current_dir / name
            member_count += 1
            if member_count > _MAX_ARCHIVE_MEMBERS:
                raise ValueError("release archive has too many members")
            if entry.is_symlink() or not entry.is_dir():
                raise ValueError("release archive contains a symlink directory")
        for name in filenames:
            entry = current_dir / name
            member_count += 1
            if member_count > _MAX_ARCHIVE_MEMBERS:
                raise ValueError("release archive has too many members")
            if (
                entry.is_symlink()
                or not entry.is_file()
                or not entry.resolve().is_relative_to(resolved_root)
            ):
                raise ValueError(
                    "release archive contains a non-regular or escaping file"
                )
            expanded_bytes += entry.stat().st_size
            if expanded_bytes > _MAX_EXPANDED_BYTES:
                raise ValueError("release archive exceeds expanded limit")
            files.append(entry)
    return files, directories


def _verify_extracted_inventory(
    unpack_root: Path,
    files: list[Path],
    directories: list[Path],
) -> Path:
    resolved_root = unpack_root.resolve()
    bundles = [item for item in files if item.name == _BUNDLE_FILENAME]
    if len(bundles) != 1:
        raise ValueError(
            "release archive must contain exactly one release bundle",
        )
    bundle_path = bundles[0]
    resolved_bundle = bundle_path.resolve()
    if (
        bundle_path.is_symlink()
        or not bundle_path.is_file()
        or not resolved_bundle.is_relative_to(resolved_root)
        or resolved_bundle == resolved_root
    ):
        raise ValueError("release bundle is invalid or outside unpack root")
    bundle_root = resolved_bundle.parent
    evidence = collect_release_evidence(
        resolved_bundle, bundle_root=bundle_root
    )
    expected_files = {(bundle_root / item.path).resolve() for item in evidence}
    actual_files = {item.resolve() for item in files}
    if actual_files != expected_files:
        raise ValueError(
            "release archive contains unreferenced or missing files"
        )
    expected_dirs: set[Path] = {resolved_root}
    for file_path in expected_files:
        for parent in file_path.parents:
            if parent.is_relative_to(resolved_root):
                expected_dirs.add(parent)
    actual_dirs = {item.resolve() for item in directories}
    if not actual_dirs <= expected_dirs:
        raise ValueError("release archive contains unreferenced directories")
    return resolved_bundle


def _extract_release(
    archive_path: Path,
    unpack_root: Path,
    *,
    tar_runner: TarRunner | None = None,
) -> Path:
    """Unpack one release archive and return its verified release-bundle.json."""
    if unpack_root.is_symlink():
        raise ValueError("unpack root must not be a symlink")
    unpack_root.mkdir(parents=True, exist_ok=True)
    if tar_runner is None:
        _extract_zstd_tar_stream(archive_path, unpack_root)
    else:
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
    files, directories = _scan_unpacked_tree(unpack_root)
    return _verify_extracted_inventory(unpack_root, files, directories)


def _load_statement[Statement: ReleaseModel](
    bundle_root: Path,
    reference: ArtifactReference,
    model: type[Statement],
) -> Statement:
    safe_rel = validate_relative_artifact_path(reference.path)
    _reject_symlink_walk(bundle_root, Path(safe_rel).parts)
    path = verify_artifact_file(
        bundle_root,
        safe_rel,
        expected_sha256=reference.sha256,
        expected_size_bytes=reference.size_bytes,
    )
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _collect_solar_evidence_paths(
    bundle_root: Path,
    root: Path,
    solar: SolarIndexStatement,
) -> set[str]:
    paths: set[str] = {solar.corpus_manifest.path}
    for entry in solar.entries:
        manifest_rel = validate_relative_artifact_path(entry.manifest.path)
        _reject_symlink_walk(bundle_root, Path(manifest_rel).parts)
        manifest_file = verify_artifact_file(
            root,
            manifest_rel,
            expected_sha256=entry.manifest.sha256,
            expected_size_bytes=entry.manifest.size_bytes,
        )
        paths.add(manifest_rel)
        manifest_payload = SolarRequestManifest.from_yaml(
            manifest_file.read_text(encoding="utf-8"),
        )
        for artifact in manifest_payload.artifacts:
            rel_art = validate_relative_artifact_path(
                artifact.path, "SOLAR artifact"
            )
            art_candidate = manifest_file.parent / rel_art
            if art_candidate.is_symlink():
                raise ValueError(f"SOLAR artifact is a symlink: {rel_art}")
            resolved = art_candidate.resolve()
            if not resolved.is_relative_to(root) or resolved == root:
                raise ValueError(
                    f"SOLAR artifact escapes bundle root: {rel_art}"
                )
            rel_to_root = resolved.relative_to(root)
            _reject_symlink_walk(bundle_root, rel_to_root.parts)
            paths.add(rel_to_root.as_posix())
    return paths


def collect_release_evidence(
    bundle_path: Path,
    *,
    bundle_root: Path,
) -> tuple[ArtifactReference, ...]:
    """Return the exact transitively-referenced evidence set of one bundle.

    The set is the transitive closure of ``ArtifactReference`` paths from the
    ``ReleaseBundle`` plus each SOLAR manifest's own ``artifacts`` list -- i.e.
    exactly the files ``verify_and_score_release`` re-checks. Files the verifier
    never reads (execution plans, trace sidecars) are therefore excluded by
    construction.
    """
    if (
        bundle_root.is_symlink()
        or bundle_path.is_symlink()
        or not bundle_path.is_file()
    ):
        raise ValueError("release bundle and root must not be symlinks")
    root = bundle_root.resolve()
    resolved_bundle = bundle_path.resolve()
    if not resolved_bundle.is_relative_to(root) or resolved_bundle == root:
        raise ValueError("release bundle must reside below bundle root")
    _reject_symlink_walk(bundle_root, resolved_bundle.relative_to(root).parts)
    bundle = ReleaseBundle.model_validate_json(
        resolved_bundle.read_text(encoding="utf-8"),
    )
    baseline = _load_statement(bundle_root, bundle.baseline, BaselineStatement)
    candidate = _load_statement(
        bundle_root, bundle.candidate, CandidateStatement
    )
    solar = _load_statement(bundle_root, bundle.solar, SolarIndexStatement)

    paths: set[str] = {
        resolved_bundle.relative_to(root).as_posix(),
        bundle.corpus_manifest.path,
        bundle.baseline.path,
        bundle.candidate.path,
        bundle.solar.path,
    }
    for statement in (baseline, candidate):
        paths.add(statement.corpus_manifest.path)
        paths.add(statement.environment.path)
        for problem in statement.problems:
            paths.add(problem.implementation.path)
            paths.add(problem.trace.path)

    paths.update(_collect_solar_evidence_paths(bundle_root, root, solar))
    for rel in paths:
        _reject_symlink_walk(
            bundle_root,
            Path(validate_relative_artifact_path(rel)).parts,
        )
    return tuple(
        artifact_reference(bundle_root, bundle_root / path)
        for path in sorted(paths)
    )


def package_score_release(
    *,
    bundle_path: Path,
    corpus_manifest_path: Path,
    archive_output: Path,
    attestation_output: Path,
    source_revision: str,
    hardware_validation: HardwareValidationBinding,
    tar_runner: TarRunner = _system_tar,
) -> ScoreReleaseAttestation:
    """Package one verified release bundle into a deterministic release object.

    Full fail-closed verification precedes any archive creation; a release
    object can never be built from an unverified bundle. ``corpus_manifest_path``
    is the repository-authoritative corpus manifest whose ``authored_root``
    resolves to the on-disk problem definitions and workloads.
    """
    if hardware_validation.source_revision != source_revision:
        raise ValueError("hardware validation source revision mismatch")
    if bundle_path.is_symlink() or bundle_path.parent.is_symlink():
        raise ValueError(f"release bundle must not be a symlink: {bundle_path}")
    bundle = bundle_path.resolve()
    if not bundle.is_file():
        raise ValueError(f"release bundle is not a regular file: {bundle}")
    bundle_root = bundle.parent
    result = verify_and_score_release(
        bundle,
        corpus_manifest_path=corpus_manifest_path,
    )

    inventory = collect_release_evidence(bundle, bundle_root=bundle_root)
    _archive_score_inventory(
        inventory,
        bundle_root=bundle_root,
        archive_output=archive_output,
        tar_runner=tar_runner,
    )

    attestation = _build_score_release_attestation(
        bundle=bundle,
        archive_output=archive_output,
        inventory=inventory,
        result=result,
        source_revision=source_revision,
        hardware_validation=hardware_validation,
    )
    atomic_write_json_value(
        attestation_output,
        attestation.model_dump(mode="json"),
    )
    return attestation


def _build_score_release_attestation(
    *,
    bundle: Path,
    archive_output: Path,
    inventory: tuple[ArtifactReference, ...],
    result: OfficialScoreResult,
    source_revision: str,
    hardware_validation: HardwareValidationBinding,
) -> ScoreReleaseAttestation:
    bundle_sha256 = sha256_file(bundle)
    archive_sha256 = sha256_file(archive_output)
    inventory_sha256 = stable_json_checksum(
        [item.model_dump(mode="json") for item in inventory],
    )
    release_id = stable_json_checksum(
        {
            "kind": "score_release",
            "bundle_sha256": bundle_sha256,
            "archive_sha256": archive_sha256,
            "source_revision": source_revision,
            "hardware_validation_receipt_sha256": (
                hardware_validation.receipt_sha256
            ),
        },
    )
    return ScoreReleaseAttestation(
        release_id=release_id,
        bundle_sha256=bundle_sha256,
        archive=ScoreReleaseArchive(
            name=archive_output.name,
            sha256=archive_sha256,
            size_bytes=archive_output.stat().st_size,
            bundle_sha256=bundle_sha256,
            source_revision=source_revision,
        ),
        inventory_sha256=inventory_sha256,
        uncompressed_size_bytes=sum(item.size_bytes for item in inventory),
        source_revision=source_revision,
        hardware_validation=hardware_validation,
        official_score=result.suite.score,
        scored_workloads=result.suite.scored_workloads,
        baseline_id=result.baseline_id,
        candidate_id=result.candidate_id,
        created_at=datetime.now(UTC).isoformat(),
    )


def _archive_score_inventory(
    inventory: tuple[ArtifactReference, ...],
    *,
    bundle_root: Path,
    archive_output: Path,
    tar_runner: TarRunner,
) -> None:
    staging_parent = Path(tempfile.mkdtemp(prefix="score-release-stage-"))
    staging_root = staging_parent / _ARCHIVE_ROOT_NAME
    staging_root.mkdir()
    try:
        for reference in inventory:
            destination = staging_root / reference.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundle_root / reference.path, destination)
        _deterministic_zstd_tar(
            staging_root, archive_output, tar_runner=tar_runner
        )
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def verify_score_release_archive(
    *,
    archive_path: Path,
    corpus_manifest_path: Path,
    expected_sha256: str | None = None,
    unpack_root: Path | None = None,
    tar_runner: TarRunner | None = None,
) -> OfficialScoreResult:
    """Verify a downloaded release archive and reproduce its official score.

    ``corpus_manifest_path`` is the repository-authoritative corpus manifest
    (checked out alongside the archive) whose ``authored_root`` resolves to the
    on-disk problem definitions and workloads.
    """
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("release archive must be a regular file")
    archive = archive_path.resolve()
    if archive.stat().st_size > _MAX_COMPRESSED_BYTES:
        raise ValueError("release archive exceeds the compressed size limit")
    actual = sha256_file(archive)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("release archive SHA-256 does not match expectation")
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if unpack_root is not None:
        if unpack_root.is_symlink():
            raise ValueError("unpack root must not be a symlink")
        if unpack_root.exists() and any(unpack_root.iterdir()):
            raise ValueError("unpack root must be empty")
        root = unpack_root.resolve()
    else:
        temporary = tempfile.TemporaryDirectory(prefix="release-verify-")
        root = Path(temporary.name)
    try:
        bundle_path = _extract_release(archive, root, tar_runner=tar_runner)
        return verify_and_score_release(
            bundle_path,
            corpus_manifest_path=corpus_manifest_path,
        )
    finally:
        if temporary is not None:
            temporary.cleanup()


__all__ = [
    "ScoreReleaseArchive",
    "ScoreReleaseAttestation",
    "TarRunner",
    "collect_release_evidence",
    "package_score_release",
    "verify_score_release_archive",
]
