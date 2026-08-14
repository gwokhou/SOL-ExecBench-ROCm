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

import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field

from sol_execbench.core.data.base_model import CurrentFrozenSchemaModel
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import (
    SHA256Digest,
    sha256_file,
    stable_json_checksum,
    verify_artifact_file,
)
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
    source_revision: str = Field(min_length=1)


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
    source_revision: str = Field(min_length=1)
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


def _extract_release(
    archive_path: Path,
    unpack_root: Path,
    *,
    tar_runner: TarRunner,
) -> Path:
    """Unpack one release archive and return its release-bundle.json path."""
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
    bundles = list(unpack_root.rglob(_BUNDLE_FILENAME))
    if len(bundles) != 1:
        raise ValueError(
            "release archive must contain exactly one release bundle",
        )
    return bundles[0]


def _load_statement[Statement: ReleaseModel](
    bundle_root: Path,
    reference: ArtifactReference,
    model: type[Statement],
) -> Statement:
    path = verify_artifact_file(
        bundle_root,
        reference.path,
        expected_sha256=reference.sha256,
        expected_size_bytes=reference.size_bytes,
    )
    return model.model_validate_json(path.read_text(encoding="utf-8"))


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
    root = bundle_root.resolve()
    bundle = ReleaseBundle.model_validate_json(
        bundle_path.read_text(encoding="utf-8"),
    )
    baseline = _load_statement(root, bundle.baseline, BaselineStatement)
    candidate = _load_statement(root, bundle.candidate, CandidateStatement)
    solar = _load_statement(root, bundle.solar, SolarIndexStatement)

    paths: set[str] = set()
    paths.add(bundle_path.resolve().relative_to(root).as_posix())
    paths.add(bundle.corpus_manifest.path)
    paths.add(bundle.baseline.path)
    paths.add(bundle.candidate.path)
    paths.add(bundle.solar.path)

    for statement in (baseline, candidate):
        paths.add(statement.corpus_manifest.path)
        paths.add(statement.environment.path)
        for problem in statement.problems:
            paths.add(problem.implementation.path)
            paths.add(problem.trace.path)

    paths.add(solar.corpus_manifest.path)
    for entry in solar.entries:
        manifest_rel = entry.manifest.path
        paths.add(manifest_rel)
        manifest_file = root / manifest_rel
        manifest_payload = SolarRequestManifest.from_yaml(
            manifest_file.read_text(encoding="utf-8"),
        )
        for artifact in manifest_payload.artifacts:
            resolved = (manifest_file.parent / artifact.path).resolve()
            paths.add(resolved.relative_to(root).as_posix())

    return tuple(
        artifact_reference(root, root / path) for path in sorted(paths)
    )


def package_score_release(
    *,
    bundle_path: Path,
    corpus_manifest_path: Path,
    archive_output: Path,
    attestation_output: Path,
    source_revision: str,
    tar_runner: TarRunner = _system_tar,
) -> ScoreReleaseAttestation:
    """Package one verified release bundle into a deterministic release object.

    Full fail-closed verification precedes any archive creation; a release
    object can never be built from an unverified bundle. ``corpus_manifest_path``
    is the repository-authoritative corpus manifest whose ``authored_root``
    resolves to the on-disk problem definitions and workloads.
    """
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

    bundle_sha256 = sha256_file(bundle)
    archive_sha256 = sha256_file(archive_output)
    archive_size = archive_output.stat().st_size
    inventory_sha256 = stable_json_checksum(
        [
            {
                "path": item.path,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in inventory
        ],
    )
    release_id = stable_json_checksum(
        {
            "kind": "score_release",
            "bundle_sha256": bundle_sha256,
            "archive_sha256": archive_sha256,
            "source_revision": source_revision,
        },
    )
    attestation = ScoreReleaseAttestation(
        release_id=release_id,
        bundle_sha256=bundle_sha256,
        archive=ScoreReleaseArchive(
            name=archive_output.name,
            sha256=archive_sha256,
            size_bytes=archive_size,
            bundle_sha256=bundle_sha256,
            source_revision=source_revision,
        ),
        inventory_sha256=inventory_sha256,
        uncompressed_size_bytes=sum(item.size_bytes for item in inventory),
        source_revision=source_revision,
        official_score=result.suite.score,
        scored_workloads=result.suite.scored_workloads,
        baseline_id=result.baseline_id,
        candidate_id=result.candidate_id,
        created_at=datetime.now(UTC).isoformat(),
    )
    atomic_write_json_value(
        attestation_output,
        attestation.model_dump(mode="json"),
    )
    return attestation


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
    tar_runner: TarRunner = _system_tar,
) -> OfficialScoreResult:
    """Verify a downloaded release archive and reproduce its official score.

    ``corpus_manifest_path`` is the repository-authoritative corpus manifest
    (checked out alongside the archive) whose ``authored_root`` resolves to the
    on-disk problem definitions and workloads.
    """
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
