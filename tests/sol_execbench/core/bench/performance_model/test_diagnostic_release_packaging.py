from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import NoReturn

import pytest
import zstandard

from sol_execbench.core.bench.performance_model import release as release_module
from sol_execbench.core.bench.performance_model.lifecycle import (
    DiagnosticLifecycleStage,
    DiagnosticPublicationLifecycleManifest,
    DiagnosticReleaseLifecycleManifest,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
    publication_id,
    publication_registry_dir,
    releases_dir,
)
from sol_execbench.core.bench.performance_model.publication import (
    DiagnosticPublicationArtifact,
    DiagnosticPublicationProjection,
)
from sol_execbench.core.bench.performance_model.release import (
    DiagnosticReleaseAttestation,
    packaging,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
)
from sol_execbench.core.integrity import sha256_file


def _noop_semantic_loader(
    manifest_path: Path,
    *,
    workload_uuid: str,
    definition: str,
) -> NoReturn:
    raise AssertionError("semantic loader must not run in packager tests")


_PATHS = (
    "development.json",
    "calibration/profile.json",
    "calibration/profile.audit.json",
    "source-inference.json",
    "inference.json",
)


def _publication(root: Path) -> DiagnosticPublicationProjection:
    artifacts: list[DiagnosticPublicationArtifact] = []
    for relative in sorted(_PATHS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        artifacts.append(
            DiagnosticPublicationArtifact(
                path=relative,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    # The packager hashes the self-describing manifest, which is not part of
    # the exact inventory (matching the production publication contract).
    (root / "publication.json").write_text("{}", encoding="utf-8")
    indexed = {item.path: item for item in artifacts}
    return DiagnosticPublicationProjection(
        case_count=880,
        source_corpus_sha256="a" * 64,
        corpus=indexed[_PATHS[0]],
        calibration_profile=indexed[_PATHS[1]],
        calibration_audit=indexed[_PATHS[2]],
        source_inference_profile=indexed[_PATHS[3]],
        inference_profile=indexed[_PATHS[4]],
        artifacts=artifacts,
        uncompressed_size_bytes=sum(item.size_bytes for item in artifacts),
    )


def _package(
    root: Path,
    tmp_path: Path,
    *,
    store_root: Path | None = None,
    revision: str = "19f195a8",
    index: str = "",
) -> tuple[Path, Path, DiagnosticReleaseAttestation]:
    archive = tmp_path / f"release{index}.tar.zst"
    attestation = tmp_path / f"attestation{index}.json"
    attestation_value = release_module.package_diagnostic_publication(
        manifest_path=root / "publication.json",
        archive_output=archive,
        attestation_output=attestation,
        source_revision=revision,
        semantic_loader=_noop_semantic_loader,
        solar_verifier=lambda path: None,
        store_root_path=store_root,
    )
    return archive, attestation, attestation_value


def test_package_is_deterministic_and_binds_the_verified_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "publication"
    root.mkdir()
    projection = _publication(root)
    monkeypatch.setattr(
        packaging,
        "verify_diagnostic_publication_projection",
        lambda *_args, **_kwargs: projection,
    )

    archive_one, _, first = _package(root, tmp_path, index="1")
    archive_two, _, second = _package(root, tmp_path, index="2")

    assert sha256_file(archive_one) == sha256_file(archive_two)
    assert first.release_id == second.release_id
    assert first.archive.sha256 == sha256_file(archive_one)
    assert first.publication_id == second.publication_id
    assert first.case_count == 880
    assert first.uncompressed_size_bytes == projection.uncompressed_size_bytes
    assert first.archive.publication_manifest_sha256 == sha256_file(
        root / "publication.json"
    )
    # The archive inventory digest binds the exact sorted file inventory.
    assert len(first.inventory_sha256) == 64


def test_package_writes_an_immutable_release_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "publication"
    root.mkdir()
    projection = _publication(root)
    monkeypatch.setattr(
        packaging,
        "verify_diagnostic_publication_projection",
        lambda *_args, **_kwargs: projection,
    )
    store = tmp_path / "store"
    pub_id = publication_id(
        source_corpus_sha256=projection.source_corpus_sha256,
        publication_manifest_sha256=sha256_file(root / "publication.json"),
        uncompressed_size_bytes=projection.uncompressed_size_bytes,
        case_count=projection.case_count,
    )
    publication_manifest = DiagnosticPublicationLifecycleManifest(
        stage=DiagnosticLifecycleStage.PUBLICATION,
        stage_id=pub_id,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.PUBLICATION_RELEASE,
        source_revision="19f195a8",
        created_at="2026-01-01T00:00:00+00:00",
        source_corpus_sha256=projection.source_corpus_sha256,
        publication_manifest_sha256=sha256_file(root / "publication.json"),
        uncompressed_size_bytes=projection.uncompressed_size_bytes,
        case_count=projection.case_count,
    )
    atomic_write_json_value(
        publication_registry_dir(store) / pub_id / "manifest.json",
        publication_manifest.model_dump(mode="json"),
    )

    _, attestation_path, attestation = _package(
        root,
        tmp_path,
        store_root=store,
    )

    directory = releases_dir(store) / attestation.release_id
    assert directory.is_dir()
    manifest = load_json_file(
        DiagnosticReleaseLifecycleManifest,
        directory / "manifest.json",
    )
    assert manifest.archive_sha256 == attestation.archive.sha256
    assert manifest.attestation_sha256 == sha256_file(attestation_path)
    assert manifest.published is False

    # A second release object for the same identity is refused.
    with pytest.raises(FileExistsError):
        _package(root, tmp_path, store_root=store, index="dup")


def test_verify_round_trip_accepts_the_packaged_archive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "publication"
    root.mkdir()
    projection = _publication(root)
    monkeypatch.setattr(
        packaging,
        "verify_diagnostic_publication_projection",
        lambda *_args, **_kwargs: projection,
    )

    archive, _, attestation = _package(root, tmp_path)

    verified = release_module.verify_diagnostic_release_archive(
        archive_path=archive,
        semantic_loader=_noop_semantic_loader,
        solar_verifier=lambda path: None,
        expected_sha256=attestation.archive.sha256,
    )
    assert verified.case_count == 880
    assert (
        verified.uncompressed_size_bytes == projection.uncompressed_size_bytes
    )


def test_verify_rejects_a_wrong_expected_checksum(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "publication"
    root.mkdir()
    projection = _publication(root)
    monkeypatch.setattr(
        packaging,
        "verify_diagnostic_publication_projection",
        lambda *_args, **_kwargs: projection,
    )

    archive, _, attestation = _package(root, tmp_path)

    with pytest.raises(ValueError, match="does not match expectation"):
        release_module.verify_diagnostic_release_archive(
            archive_path=archive,
            semantic_loader=_noop_semantic_loader,
            solar_verifier=lambda path: None,
            expected_sha256="0" * 64,
        )
    assert attestation.archive.sha256 != "0" * 64


def test_package_refuses_an_unverified_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "publication"
    root.mkdir()

    def failing_verify(*_args, **_kwargs):
        raise ValueError("publication tree is not verified")

    monkeypatch.setattr(
        packaging,
        "verify_diagnostic_publication_projection",
        failing_verify,
    )
    with pytest.raises(ValueError, match="not verified"):
        _package(root, tmp_path)


def _write_malicious_archive(
    path: Path, member: tarfile.TarInfo, content: bytes = b"x"
) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        archive.addfile(member, io.BytesIO(content) if member.isreg() else None)
    path.write_bytes(zstandard.ZstdCompressor().compress(buffer.getvalue()))


def test_verify_rejects_archive_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "traversal.tar.zst"
    member = tarfile.TarInfo("../publication.json")
    member.size = 1
    _write_malicious_archive(archive, member)

    with pytest.raises(ValueError, match="unsafe release archive member"):
        release_module.verify_diagnostic_release_archive(
            archive_path=archive,
            semantic_loader=_noop_semantic_loader,
            solar_verifier=lambda path: None,
        )


def test_verify_rejects_archive_links(tmp_path: Path) -> None:
    archive = tmp_path / "link.tar.zst"
    member = tarfile.TarInfo("publication/publication.json")
    member.type = tarfile.SYMTYPE
    member.linkname = "/etc/passwd"
    _write_malicious_archive(archive, member)

    with pytest.raises(ValueError, match="unsupported release archive member"):
        release_module.verify_diagnostic_release_archive(
            archive_path=archive,
            semantic_loader=_noop_semantic_loader,
            solar_verifier=lambda path: None,
        )
