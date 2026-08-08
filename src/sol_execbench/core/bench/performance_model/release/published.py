# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed observation of an externally published diagnostic release."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from sol_execbench.core.bench.performance_model.lifecycle.blob_store import (
    BlobStore,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import (
    published_releases_dir,
)
from sol_execbench.core.bench.performance_model.release.packaging import (
    DiagnosticReleaseAttestation,
)
from sol_execbench.core.data.base_model import CurrentFrozenSchemaModel
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
)
from sol_execbench.core.integrity import SHA256Digest, sha256_file
from sol_execbench.core.integrity.schema_versions import SchemaVersion
from sol_execbench.core.platform.runtime import resolve_tool_path
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

_GH_TIMEOUT_SECONDS = 60.0
_MAX_GH_OUTPUT_BYTES = 256 * 1024
_TAG = "diagnostic-lifecycle-p0-conformance-v1"
_ARCHIVE = f"{_TAG}.tar.zst"
_ATTESTATION = f"{_TAG}.attestation.json"


class DiagnosticPublishedRelease(CurrentFrozenSchemaModel):
    """Immutable receipt reconstructed from downloaded public assets."""

    current_schema_version = SchemaVersion.DIAGNOSTIC_PUBLISHED_RELEASE

    schema_version: Literal[SchemaVersion.DIAGNOSTIC_PUBLISHED_RELEASE] = (
        SchemaVersion.DIAGNOSTIC_PUBLISHED_RELEASE
    )
    release_id: SHA256Digest
    purpose: Literal[DiagnosticEvidencePurpose.CONTROL_PLANE_CONFORMANCE]
    tag: Literal["diagnostic-lifecycle-p0-conformance-v1"]
    url: str = Field(min_length=1)
    source_revision: str = Field(min_length=40)
    archive_sha256: SHA256Digest
    attestation_sha256: SHA256Digest
    observed_at: str = Field(min_length=1)
    published: Literal[True] = True


GitHubRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _run_gh(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    gh = resolve_tool_path("gh")
    if gh is None:
        raise RuntimeError("GitHub CLI is required to observe a release")
    return run_in_process_group_bounded(
        [str(gh), *arguments],
        timeout=_GH_TIMEOUT_SECONDS,
        max_capture_bytes=_MAX_GH_OUTPUT_BYTES,
    )


def _require_success(
    completed: subprocess.CompletedProcess[str], operation: str
) -> str:
    if completed.returncode != 0:
        raise ValueError(f"GitHub release {operation} failed")
    return completed.stdout


def ingest_github_published_release(
    *,
    repository: str,
    store_root_path: Path,
    runner: GitHubRunner = _run_gh,
) -> DiagnosticPublishedRelease:
    """Download, verify, and persist the fixed public conformance release."""
    metadata = json.loads(
        _require_success(
            runner(
                [
                    "release",
                    "view",
                    _TAG,
                    "--repo",
                    repository,
                    "--json",
                    "assets,isDraft,tagName,url,targetCommitish",
                ]
            ),
            "metadata read",
        )
    )
    _validate_metadata(metadata)
    with tempfile.TemporaryDirectory(
        prefix="diagnostic-release-ingest-"
    ) as name:
        root = Path(name)
        _require_success(
            runner(
                [
                    "release",
                    "download",
                    _TAG,
                    "--repo",
                    repository,
                    "--dir",
                    str(root),
                ]
            ),
            "asset download",
        )
        receipt = _receipt_from_download(root, metadata)
        _persist_receipt(store_root_path.resolve(), receipt)
        return receipt


def _validate_metadata(metadata: object) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("GitHub release metadata is not an object")
    if metadata.get("isDraft") is not False or metadata.get("tagName") != _TAG:
        raise ValueError("GitHub release is not the expected published tag")
    assets = metadata.get("assets")
    if not isinstance(assets, list):
        raise ValueError("GitHub release assets are unavailable")
    names = {item.get("name") for item in assets if isinstance(item, dict)}
    if names != {_ARCHIVE, _ATTESTATION}:
        raise ValueError("GitHub release must contain exactly two fixed assets")


def _receipt_from_download(
    root: Path, metadata: dict[str, object]
) -> DiagnosticPublishedRelease:
    archive = root / _ARCHIVE
    attestation_path = root / _ATTESTATION
    if {path.name for path in root.iterdir()} != {_ARCHIVE, _ATTESTATION}:
        raise ValueError("downloaded release asset set differs from metadata")
    attestation = load_json_file(DiagnosticReleaseAttestation, attestation_path)
    archive_digest = sha256_file(archive)
    if archive_digest != attestation.archive.sha256:
        raise ValueError("downloaded release archive digest mismatch")
    if (
        attestation.purpose
        is not DiagnosticEvidencePurpose.CONTROL_PLANE_CONFORMANCE
    ):
        raise ValueError(
            "published diagnostic release has production authority"
        )
    return DiagnosticPublishedRelease(
        release_id=attestation.release_id,
        purpose=attestation.purpose,
        tag=_TAG,
        url=str(metadata["url"]),
        source_revision=attestation.source_revision,
        archive_sha256=archive_digest,
        attestation_sha256=sha256_file(attestation_path),
        observed_at=datetime.now(UTC).isoformat(),
    )


def _persist_receipt(root: Path, receipt: DiagnosticPublishedRelease) -> None:
    directory = published_releases_dir(root) / receipt.release_id
    path = directory / "receipt.json"
    if path.exists():
        existing = load_json_file(DiagnosticPublishedRelease, path)
        if existing != receipt:
            raise ValueError("published release receipt already differs")
        return
    directory.mkdir(parents=True)
    atomic_write_json_value(path, receipt.model_dump(mode="json"))
    BlobStore(root).put_file(path)


__all__ = [
    "DiagnosticPublishedRelease",
    "GitHubRunner",
    "ingest_github_published_release",
]
