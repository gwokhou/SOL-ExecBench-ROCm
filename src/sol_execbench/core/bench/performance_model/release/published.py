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
from typing import Literal, TypedDict, cast

from pydantic import Field, model_validator

from sol_execbench.core.bench.performance_model.lifecycle.blob_store import (
    BlobStore,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
)
from sol_execbench.core.bench.performance_model.lifecycle.identity import (
    release_id as lifecycle_release_id,
)
from sol_execbench.core.bench.performance_model.lifecycle.models import (
    DiagnosticReleaseLifecycleManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import (
    published_releases_dir,
    releases_dir,
)
from sol_execbench.core.bench.performance_model.release.packaging import (
    DiagnosticReleaseAttestation,
)
from sol_execbench.core.data.base_model import (
    CurrentFrozenSchemaModel,
    FrozenArtifactModel,
)
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
_SUPPORTED_RELEASES = {
    "diagnostic-lifecycle-p0-conformance-v1": (
        DiagnosticEvidencePurpose.CONTROL_PLANE_CONFORMANCE,
        "Diagnostic lifecycle P0 conformance v1",
    ),
    "gfx1200-diagnostics-v7-production-v1": (
        DiagnosticEvidencePurpose.PRODUCTION,
        "gfx1200 diagnostics v7 production v1",
    ),
}


def _asset_names(tag: str) -> tuple[str, str]:
    return f"{tag}.tar.zst", f"{tag}.attestation.json"


class _ReleaseObservation(TypedDict):
    attestation_sha256: str
    workflow_run_attempt: int
    workflow_run_id: int
    workflow_url: str


class DiagnosticPublishedReleaseAsset(FrozenArtifactModel):
    """One exact GitHub asset observed and downloaded during ingestion."""

    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9.-]+$")
    github_id: str = Field(min_length=1)
    api_url: str = Field(min_length=1)
    download_url: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: SHA256Digest


class DiagnosticPublishedRelease(CurrentFrozenSchemaModel):
    """Immutable receipt reconstructed from downloaded public assets."""

    current_schema_version = SchemaVersion.DIAGNOSTIC_PUBLISHED_RELEASE

    schema_version: Literal[SchemaVersion.DIAGNOSTIC_PUBLISHED_RELEASE] = (
        SchemaVersion.DIAGNOSTIC_PUBLISHED_RELEASE
    )
    release_id: SHA256Digest
    purpose: DiagnosticEvidencePurpose
    repository: str = Field(min_length=3)
    tag: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9.-]+$")
    url: str = Field(min_length=1)
    github_release_id: str = Field(min_length=1)
    github_release_api_url: str = Field(min_length=1)
    target_commit_sha: str = Field(
        min_length=40,
        max_length=40,
        pattern=r"^[0-9a-f]{40}$",
    )
    source_revision: str = Field(
        min_length=40,
        max_length=40,
        pattern=r"^[0-9a-f]{40}$",
    )
    published_at: str = Field(min_length=1)
    workflow_run_id: int = Field(gt=0)
    workflow_run_attempt: int = Field(gt=0)
    workflow_url: str = Field(min_length=1)
    assets: tuple[DiagnosticPublishedReleaseAsset, ...] = Field(
        min_length=2,
        max_length=2,
    )
    observed_at: str = Field(min_length=1)
    published: Literal[True] = True
    round_trip_verified: Literal[True] = True

    @model_validator(mode="after")
    def _exact_assets(self) -> DiagnosticPublishedRelease:
        names = tuple(asset.name for asset in self.assets)
        expected = _SUPPORTED_RELEASES.get(self.tag)
        if expected is None or expected[0] is not self.purpose:
            raise ValueError("published release tag/purpose is unsupported")
        if names != tuple(sorted(_asset_names(self.tag))):
            raise ValueError("published release asset inventory is not exact")
        if self.target_commit_sha != self.source_revision:
            raise ValueError(
                "published tag target differs from source revision"
            )
        return self


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
    tag: str,
    purpose: DiagnosticEvidencePurpose,
    store_root_path: Path,
    runner: GitHubRunner = _run_gh,
) -> DiagnosticPublishedRelease:
    """Download, verify, and persist one explicitly supported release."""
    expected = _SUPPORTED_RELEASES.get(tag)
    if expected is None or expected[0] is not purpose:
        raise ValueError("unsupported published release tag/purpose")
    metadata = json.loads(
        _require_success(
            runner(
                [
                    "release",
                    "view",
                    tag,
                    "--repo",
                    repository,
                    "--json",
                    (
                        "apiUrl,assets,body,id,isDraft,isPrerelease,name,"
                        "publishedAt,tagName,targetCommitish,url"
                    ),
                ]
            ),
            "metadata read",
        )
    )
    _validate_metadata(metadata, tag=tag, purpose=purpose)
    target_commit_sha = _require_success(
        runner(
            [
                "api",
                f"repos/{repository}/commits/{tag}",
                "--jq",
                ".sha",
            ]
        ),
        "tag target read",
    ).strip()
    with tempfile.TemporaryDirectory(
        prefix="diagnostic-release-ingest-"
    ) as name:
        root = Path(name)
        _require_success(
            runner(
                [
                    "release",
                    "download",
                    tag,
                    "--repo",
                    repository,
                    "--dir",
                    str(root),
                ]
            ),
            "asset download",
        )
        receipt = _receipt_from_download(
            root,
            cast(dict[str, object], metadata),
            repository=repository,
            tag=tag,
            purpose=purpose,
            target_commit_sha=target_commit_sha,
        )
        _persist_receipt(store_root_path.resolve(), receipt)
        return receipt


def _validate_metadata(
    metadata: object,
    *,
    tag: str,
    purpose: DiagnosticEvidencePurpose,
) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("GitHub release metadata is not an object")
    if (
        metadata.get("isDraft") is not False
        or metadata.get("isPrerelease") is not False
        or metadata.get("tagName") != tag
        or metadata.get("name") != _SUPPORTED_RELEASES[tag][1]
    ):
        raise ValueError("GitHub release is not the expected published tag")
    for field in ("apiUrl", "body", "id", "publishedAt", "url"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"GitHub release {field} is unavailable")
    assets = metadata.get("assets")
    if not isinstance(assets, list):
        raise ValueError("GitHub release assets are unavailable")
    names = {item.get("name") for item in assets if isinstance(item, dict)}
    if names != set(_asset_names(tag)):
        raise ValueError("GitHub release must contain exactly two fixed assets")
    if len(assets) != 2:
        raise ValueError("GitHub release asset inventory is ambiguous")
    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("GitHub release asset metadata is malformed")
        for field in ("apiUrl", "id", "name", "state", "url"):
            value = asset.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"GitHub asset {field} is unavailable")
        if not isinstance(asset.get("size"), int):
            raise ValueError("GitHub asset size is unavailable")


def _receipt_from_download(
    root: Path,
    metadata: dict[str, object],
    *,
    repository: str,
    tag: str,
    purpose: DiagnosticEvidencePurpose,
    target_commit_sha: str,
) -> DiagnosticPublishedRelease:
    _archive_name, attestation_name = _asset_names(tag)
    attestation, local_assets = _verified_downloaded_assets(
        root,
        tag=tag,
        purpose=purpose,
        target_commit_sha=target_commit_sha,
    )
    observation = _release_observation(metadata)
    if observation["attestation_sha256"] != local_assets[attestation_name][1]:
        raise ValueError("workflow attestation digest differs from asset")
    assets = _published_assets(local_assets, metadata)
    return DiagnosticPublishedRelease(
        release_id=attestation.release_id,
        purpose=purpose,
        repository=repository,
        tag=tag,
        url=str(metadata["url"]),
        github_release_id=str(metadata["id"]),
        github_release_api_url=str(metadata["apiUrl"]),
        target_commit_sha=target_commit_sha,
        source_revision=attestation.source_revision,
        published_at=str(metadata["publishedAt"]),
        workflow_run_id=int(observation["workflow_run_id"]),
        workflow_run_attempt=int(observation["workflow_run_attempt"]),
        workflow_url=str(observation["workflow_url"]),
        assets=assets,
        observed_at=datetime.now(UTC).isoformat(),
    )


def _verified_downloaded_assets(
    root: Path,
    *,
    tag: str,
    purpose: DiagnosticEvidencePurpose,
    target_commit_sha: str,
) -> tuple[DiagnosticReleaseAttestation, dict[str, tuple[Path, SHA256Digest]]]:
    archive_name, attestation_name = _asset_names(tag)
    archive = root / archive_name
    attestation_path = root / attestation_name
    if {path.name for path in root.iterdir()} != {
        archive_name,
        attestation_name,
    }:
        raise ValueError("downloaded release asset set differs from metadata")
    attestation = load_json_file(DiagnosticReleaseAttestation, attestation_path)
    archive_digest = sha256_file(archive)
    attestation_digest = sha256_file(attestation_path)
    if archive_digest != attestation.archive.sha256:
        raise ValueError("downloaded release archive digest mismatch")
    if archive.stat().st_size != attestation.archive.size_bytes:
        raise ValueError("downloaded release archive size mismatch")
    if attestation.purpose is not purpose:
        raise ValueError("published diagnostic release purpose mismatch")
    if attestation.source_revision != target_commit_sha:
        raise ValueError("published tag target differs from source revision")
    expected_release_id = lifecycle_release_id(
        publication_id=attestation.publication_id,
        archive_sha256=attestation.archive.sha256,
        source_revision=attestation.source_revision,
        producer_version=attestation.producer_version,
        archive_size_bytes=attestation.archive.size_bytes,
        purpose=attestation.purpose,
    )
    if expected_release_id != attestation.release_id:
        raise ValueError("published release identity does not recompute")
    return attestation, {
        archive_name: (archive, archive_digest),
        attestation_name: (attestation_path, attestation_digest),
    }


def _published_assets(
    local_assets: dict[str, tuple[Path, SHA256Digest]],
    metadata: dict[str, object],
) -> tuple[DiagnosticPublishedReleaseAsset, ...]:
    metadata_assets = {
        str(item["name"]): item
        for item in cast(list[dict[str, object]], metadata["assets"])
    }
    assets: list[DiagnosticPublishedReleaseAsset] = []
    for name in sorted(local_assets):
        path, digest = local_assets[name]
        item = metadata_assets[name]
        if (
            item.get("state") != "uploaded"
            or item.get("size") != path.stat().st_size
        ):
            raise ValueError(f"GitHub asset metadata differs for {name}")
        assets.append(
            DiagnosticPublishedReleaseAsset(
                name=name,
                github_id=str(item["id"]),
                api_url=str(item["apiUrl"]),
                download_url=str(item["url"]),
                size_bytes=path.stat().st_size,
                sha256=digest,
            )
        )
    return tuple(assets)


def _release_observation(metadata: dict[str, object]) -> _ReleaseObservation:
    try:
        value = json.loads(str(metadata["body"]))
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "GitHub release body is not a workflow receipt"
        ) from error
    if not isinstance(value, dict):
        raise ValueError("GitHub release body is not a workflow receipt")
    required = {
        "attestation_sha256",
        "workflow_run_attempt",
        "workflow_run_id",
        "workflow_url",
    }
    if set(value) != required:
        raise ValueError("GitHub release workflow receipt fields are not exact")
    if (
        not isinstance(value["attestation_sha256"], str)
        or not isinstance(value["workflow_url"], str)
        or type(value["workflow_run_attempt"]) is not int
        or type(value["workflow_run_id"]) is not int
    ):
        raise ValueError("GitHub release workflow receipt values are malformed")
    return cast(_ReleaseObservation, value)


def _persist_receipt(root: Path, receipt: DiagnosticPublishedRelease) -> None:
    candidate_path = releases_dir(root) / receipt.release_id / "manifest.json"
    if not candidate_path.is_file():
        raise ValueError("published release has no local release candidate")
    candidate = load_json_file(
        DiagnosticReleaseLifecycleManifest,
        candidate_path,
    )
    assets = {asset.name: asset for asset in receipt.assets}
    archive_name, attestation_name = _asset_names(receipt.tag)
    if (
        candidate.source_revision != receipt.source_revision
        or candidate.archive_sha256 != assets[archive_name].sha256
        or candidate.archive_size_bytes != assets[archive_name].size_bytes
        or candidate.attestation_sha256 != assets[attestation_name].sha256
    ):
        raise ValueError("published assets differ from local release candidate")
    blob_store = BlobStore(root)
    for asset in receipt.assets:
        if not blob_store.contains(asset.sha256):
            raise ValueError(
                f"published asset is absent from local CAS: {asset.name}"
            )
    directory = published_releases_dir(root) / receipt.release_id
    path = directory / "receipt.json"
    if path.exists():
        existing = load_json_file(DiagnosticPublishedRelease, path)
        if existing != receipt:
            raise ValueError("published release receipt already differs")
        return
    directory.mkdir(parents=True)
    atomic_write_json_value(path, receipt.model_dump(mode="json"))
    blob_store.put_file(path)


__all__ = [
    "DiagnosticPublishedRelease",
    "DiagnosticPublishedReleaseAsset",
    "GitHubRunner",
    "ingest_github_published_release",
]
