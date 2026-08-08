from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from sol_execbench.core.bench.performance_model.lifecycle import (
    DiagnosticEvidencePurpose,
    published_releases_dir,
)
from sol_execbench.core.bench.performance_model.release import (
    DiagnosticReleaseArchive,
    DiagnosticReleaseAttestation,
    ingest_github_published_release,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file

_TAG = "diagnostic-lifecycle-p0-conformance-v1"


class _Runner:
    def __init__(self, assets: Path, *, draft: bool = False) -> None:
        self.assets = assets
        self.draft = draft

    def __call__(
        self, arguments: list[str]
    ) -> subprocess.CompletedProcess[str]:
        if arguments[1] == "view":
            payload = {
                "assets": [
                    {"name": f"{_TAG}.tar.zst"},
                    {"name": f"{_TAG}.attestation.json"},
                ],
                "isDraft": self.draft,
                "tagName": _TAG,
                "url": "https://example.invalid/release",
                "targetCommitish": "a" * 40,
            }
            return subprocess.CompletedProcess(
                arguments, 0, json.dumps(payload), ""
            )
        destination = Path(arguments[arguments.index("--dir") + 1])
        for source in self.assets.iterdir():
            shutil.copyfile(source, destination / source.name)
        return subprocess.CompletedProcess(arguments, 0, "", "")


def _assets(root: Path) -> Path:
    root.mkdir()
    archive = root / f"{_TAG}.tar.zst"
    archive.write_bytes(b"archive")
    attestation = DiagnosticReleaseAttestation(
        release_id="1" * 64,
        purpose=DiagnosticEvidencePurpose.CONTROL_PLANE_CONFORMANCE,
        publication_id="2" * 64,
        archive=DiagnosticReleaseArchive(
            name=archive.name,
            sha256=sha256_file(archive),
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
        root / f"{_TAG}.attestation.json",
        attestation.model_dump(mode="json"),
    )
    return root


def test_ingest_published_release_reconstructs_remote_receipt(
    tmp_path: Path,
) -> None:
    assets = _assets(tmp_path / "assets")
    store = tmp_path / "store"

    receipt = ingest_github_published_release(
        repository="owner/repository",
        store_root_path=store,
        runner=_Runner(assets),
    )

    assert receipt.published is True
    assert (
        receipt.purpose is DiagnosticEvidencePurpose.CONTROL_PLANE_CONFORMANCE
    )
    assert (
        published_releases_dir(store) / receipt.release_id / "receipt.json"
    ).is_file()


def test_ingest_refuses_a_draft_release(tmp_path: Path) -> None:
    assets = _assets(tmp_path / "assets")
    with pytest.raises(ValueError, match="expected published tag"):
        ingest_github_published_release(
            repository="owner/repository",
            store_root_path=tmp_path / "store",
            runner=_Runner(assets, draft=True),
        )
