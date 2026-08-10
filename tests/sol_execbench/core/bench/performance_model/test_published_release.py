from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from sol_execbench.core.bench.performance_model.lifecycle import (
    BlobStore,
    DiagnosticEvidencePurpose,
    DiagnosticLifecycleArtifact,
    DiagnosticLifecycleParent,
    DiagnosticLifecycleStage,
    DiagnosticReleaseLifecycleManifest,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
    published_releases_dir,
    release_id,
    releases_dir,
)
from sol_execbench.core.bench.performance_model.release import (
    DiagnosticReleaseArchive,
    DiagnosticReleaseAttestation,
    ingest_github_published_release,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file

_TAG = "diagnostic-lifecycle-p0-conformance-v1"
_PRODUCTION_TAG = "gfx1200-diagnostics-v7-production-v1"
_RELEASE_NAMES = {
    _TAG: "Diagnostic lifecycle P0 conformance v1",
    _PRODUCTION_TAG: "gfx1200 diagnostics v7 production v1",
}


class _Runner:
    def __init__(
        self, assets: Path, *, tag: str = _TAG, draft: bool = False
    ) -> None:
        self.assets = assets
        self.tag = tag
        self.draft = draft

    def __call__(
        self, arguments: list[str]
    ) -> subprocess.CompletedProcess[str]:
        if arguments[0] == "api":
            return subprocess.CompletedProcess(arguments, 0, "a" * 40, "")
        if arguments[1] == "view":
            attestation = self.assets / f"{self.tag}.attestation.json"
            archive = self.assets / f"{self.tag}.tar.zst"
            payload = {
                "assets": [
                    {
                        "apiUrl": "https://api.example.invalid/assets/1",
                        "id": "asset-attestation",
                        "name": attestation.name,
                        "size": attestation.stat().st_size,
                        "state": "uploaded",
                        "url": "https://example.invalid/attestation",
                    },
                    {
                        "apiUrl": "https://api.example.invalid/assets/2",
                        "id": "asset-archive",
                        "name": archive.name,
                        "size": archive.stat().st_size,
                        "state": "uploaded",
                        "url": "https://example.invalid/archive",
                    },
                ],
                "apiUrl": "https://api.example.invalid/releases/1",
                "body": json.dumps(
                    {
                        "attestation_sha256": sha256_file(attestation),
                        "workflow_run_attempt": 1,
                        "workflow_run_id": 123,
                        "workflow_url": "https://example.invalid/actions/123",
                    }
                ),
                "id": "release-node-id",
                "isDraft": self.draft,
                "isPrerelease": False,
                "name": _RELEASE_NAMES[self.tag],
                "publishedAt": "2026-08-09T00:01:00Z",
                "tagName": self.tag,
                "url": "https://example.invalid/release",
                "targetCommitish": "main",
            }
            return subprocess.CompletedProcess(
                arguments, 0, json.dumps(payload), ""
            )
        destination = Path(arguments[arguments.index("--dir") + 1])
        for source in self.assets.iterdir():
            shutil.copyfile(source, destination / source.name)
        return subprocess.CompletedProcess(arguments, 0, "", "")


def _assets(
    root: Path,
    *,
    tag: str = _TAG,
    purpose: DiagnosticEvidencePurpose = (
        DiagnosticEvidencePurpose.CONTROL_PLANE_CONFORMANCE
    ),
) -> Path:
    root.mkdir()
    archive = root / f"{tag}.tar.zst"
    archive.write_bytes(b"archive")
    publication_id = "2" * 64
    archive_digest = sha256_file(archive)
    release_identity = release_id(
        publication_id=publication_id,
        archive_sha256=archive_digest,
        source_revision="a" * 40,
        producer_version="4.0.0",
        archive_size_bytes=archive.stat().st_size,
        purpose=purpose,
    )
    attestation = DiagnosticReleaseAttestation(
        release_id=release_identity,
        purpose=purpose,
        publication_id=publication_id,
        archive=DiagnosticReleaseArchive(
            name="release.tar.zst",
            sha256=archive_digest,
            size_bytes=archive.stat().st_size,
            publication_manifest_sha256="3" * 64,
            source_revision="a" * 40,
        ),
        uncompressed_size_bytes=1,
        case_count=440,
        inventory_sha256="4" * 64,
        source_revision="a" * 40,
        created_at="2026-08-09T00:00:00+00:00",
    )
    atomic_write_json_value(
        root / f"{tag}.attestation.json",
        attestation.model_dump(mode="json"),
    )
    return root


def _seed_local_candidate(
    store: Path, assets: Path, *, tag: str = _TAG
) -> None:
    archive = assets / f"{tag}.tar.zst"
    attestation_path = assets / f"{tag}.attestation.json"
    attestation = DiagnosticReleaseAttestation.model_validate_json(
        attestation_path.read_text(encoding="utf-8")
    )
    blob_store = BlobStore(store)
    blob_store.put_file(archive)
    blob_store.put_file(attestation_path)
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
                sha256="5" * 64,
            ),
        ),
        exact_inventory=(
            DiagnosticLifecycleArtifact(
                relative_path=attestation.archive.name,
                sha256=attestation.archive.sha256,
                size_bytes=archive.stat().st_size,
            ),
            DiagnosticLifecycleArtifact(
                relative_path=attestation_path.name,
                sha256=sha256_file(attestation_path),
                size_bytes=attestation_path.stat().st_size,
            ),
        ),
        created_at="2026-08-09T00:00:00+00:00",
        archive_sha256=attestation.archive.sha256,
        archive_size_bytes=attestation.archive.size_bytes,
        attestation_sha256=sha256_file(attestation_path),
    )
    destination = releases_dir(store) / attestation.release_id / "manifest.json"
    destination.parent.mkdir(parents=True)
    atomic_write_json_value(destination, manifest.model_dump(mode="json"))


def test_ingest_published_release_reconstructs_remote_receipt(
    tmp_path: Path,
) -> None:
    assets = _assets(tmp_path / "assets")
    store = tmp_path / "store"
    _seed_local_candidate(store, assets)

    receipt = ingest_github_published_release(
        repository="owner/repository",
        tag=_TAG,
        purpose=DiagnosticEvidencePurpose.CONTROL_PLANE_CONFORMANCE,
        store_root_path=store,
        runner=_Runner(assets),
    )

    assert receipt.published is True
    assert (
        receipt.purpose is DiagnosticEvidencePurpose.CONTROL_PLANE_CONFORMANCE
    )
    assert receipt.repository == "owner/repository"
    assert receipt.github_release_id == "release-node-id"
    assert receipt.target_commit_sha == "a" * 40
    assert receipt.workflow_run_id == 123
    assert receipt.round_trip_verified is True
    assert {asset.github_id for asset in receipt.assets} == {
        "asset-archive",
        "asset-attestation",
    }
    assert (
        published_releases_dir(store) / receipt.release_id / "receipt.json"
    ).is_file()


def test_ingest_refuses_a_draft_release(tmp_path: Path) -> None:
    assets = _assets(tmp_path / "assets")
    _seed_local_candidate(tmp_path / "store", assets)
    with pytest.raises(ValueError, match="expected published tag"):
        ingest_github_published_release(
            repository="owner/repository",
            tag=_TAG,
            purpose=DiagnosticEvidencePurpose.CONTROL_PLANE_CONFORMANCE,
            store_root_path=tmp_path / "store",
            runner=_Runner(assets, draft=True),
        )


def test_ingest_accepts_supported_production_release(tmp_path: Path) -> None:
    assets = _assets(
        tmp_path / "assets",
        tag=_PRODUCTION_TAG,
        purpose=DiagnosticEvidencePurpose.PRODUCTION,
    )
    store = tmp_path / "store"
    _seed_local_candidate(store, assets, tag=_PRODUCTION_TAG)

    receipt = ingest_github_published_release(
        repository="owner/repository",
        tag=_PRODUCTION_TAG,
        purpose=DiagnosticEvidencePurpose.PRODUCTION,
        store_root_path=store,
        runner=_Runner(assets, tag=_PRODUCTION_TAG),
    )

    assert receipt.tag == _PRODUCTION_TAG
    assert receipt.purpose is DiagnosticEvidencePurpose.PRODUCTION
