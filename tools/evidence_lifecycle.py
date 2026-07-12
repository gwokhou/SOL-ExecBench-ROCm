#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Enforce the Git-tracked state machine for published evidence bundles.

State machine (records are append-only; published Release assets are never deleted):

    [round-trip verified bundle]
                 |
                 | record_published()
                 v
            published ---------------------> revoked
                 |                              ^
                 | record_published(           | revoke_published(reason)
                 |     supersedes=<old>)        |
                 v                              |
            superseded -------------------------+

``published`` is the only active state.  A successor publication atomically adds
the new ``published`` record and changes its predecessor to ``superseded``.  A
revocation is terminal and requires a public reason.  ``superseded`` is terminal
too: it remains downloadable and auditable but cannot return to ``published``.
The validator rejects deletion-by-omission, duplicate release identities, dangling
replacement links, unchecksummed bundles, and any transition outside this graph.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from sol_execbench.core.scoring.release_baseline import (
    load_evidence_publication_manifest,
)


SCHEMA_VERSION = "sol_execbench.evidence_lifecycle.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
_STATUSES = frozenset({"published", "superseded", "revoked"})
_ALLOWED_TRANSITIONS = {
    "published": frozenset({"superseded", "revoked"}),
    "superseded": frozenset(),
    "revoked": frozenset(),
}
_REQUIRED_FIELDS = frozenset(
    {
        "release",
        "status",
        "manifest",
        "bundle_asset",
        "bundle_sha256",
        "release_url",
        "source_revision",
        "published_at",
        "supersedes",
        "replaced_by",
        "revocation_reason",
    }
)


def _safe_repo_path(value: object, root: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes repository root") from exc
    return resolved


def _non_empty_string(record: dict[str, Any], field: str) -> str:
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"lifecycle record {field} must be a non-empty string")
    return value


def validate_lifecycle_registry(registry_path: Path, repo_root: Path) -> None:
    """Validate release-state transitions and manifest identity invariants."""
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid evidence lifecycle registry: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported evidence lifecycle registry schema")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("evidence lifecycle registry requires non-empty records")

    by_release: dict[str, dict[str, Any]] = {}
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise ValueError("evidence lifecycle records must be objects")
        if set(raw_record) != _REQUIRED_FIELDS:
            raise ValueError("evidence lifecycle record has unknown or missing fields")
        release = _non_empty_string(raw_record, "release")
        if release in by_release:
            raise ValueError(f"duplicate evidence lifecycle release: {release}")
        status = raw_record["status"]
        if status not in _STATUSES:
            raise ValueError(f"unsupported evidence lifecycle status: {status!r}")
        bundle_asset = _non_empty_string(raw_record, "bundle_asset")
        if Path(bundle_asset).name != bundle_asset or not bundle_asset.endswith(
            ".tar.gz"
        ):
            raise ValueError("bundle_asset must be a tar.gz filename")
        bundle_sha256 = _non_empty_string(raw_record, "bundle_sha256")
        if _SHA256_RE.fullmatch(bundle_sha256) is None:
            raise ValueError("bundle_sha256 must be a lowercase sha256 digest")
        release_url = _non_empty_string(raw_record, "release_url")
        if not release_url.startswith("https://github.com/"):
            raise ValueError("release_url must be a GitHub HTTPS URL")
        source_revision = _non_empty_string(raw_record, "source_revision")
        if _GIT_REVISION_RE.fullmatch(source_revision) is None:
            raise ValueError("source_revision must be a full Git object id")
        _non_empty_string(raw_record, "published_at")

        manifest_path = _safe_repo_path(raw_record["manifest"], repo_root, "manifest")
        manifest = load_evidence_publication_manifest(manifest_path)
        if manifest.release != release:
            raise ValueError("lifecycle release does not match manifest release")
        if manifest.source_revision != source_revision:
            raise ValueError("lifecycle source revision does not match manifest")
        expected_asset_url = f"{manifest.artifact_base_uri}{bundle_asset}"
        if expected_asset_url != (
            f"https://github.com/{release_url.split('github.com/', 1)[1].split('/releases/', 1)[0]}"
            f"/releases/download/{release}/{bundle_asset}"
        ):
            raise ValueError("manifest artifact URI and lifecycle release URL disagree")

        supersedes = raw_record["supersedes"]
        replaced_by = raw_record["replaced_by"]
        revocation_reason = raw_record["revocation_reason"]
        for field, value in (("supersedes", supersedes), ("replaced_by", replaced_by)):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field} must be null or a release name")
        if status == "published" and (
            replaced_by is not None or revocation_reason is not None
        ):
            raise ValueError(
                "published evidence cannot have replacement or revocation data"
            )
        if status == "superseded" and (
            not isinstance(replaced_by, str) or revocation_reason is not None
        ):
            raise ValueError("superseded evidence requires replaced_by only")
        if status == "revoked" and (
            not isinstance(revocation_reason, str) or not revocation_reason.strip()
        ):
            raise ValueError("revoked evidence requires revocation_reason")
        by_release[release] = raw_record

    for release, record in by_release.items():
        supersedes = record["supersedes"]
        replaced_by = record["replaced_by"]
        if supersedes is not None:
            predecessor = by_release.get(supersedes)
            if predecessor is None or predecessor["status"] != "superseded":
                raise ValueError(
                    "supersedes must reference a superseded lifecycle record"
                )
            if predecessor["replaced_by"] != release:
                raise ValueError("supersedes and replaced_by must form a matching pair")
        if replaced_by is not None and replaced_by not in by_release:
            raise ValueError("replaced_by must reference a known lifecycle record")


def _load_registry(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid evidence lifecycle registry: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("invalid evidence lifecycle registry")
    return payload


def _write_validated_registry(
    path: Path, payload: dict[str, Any], repo_root: Path
) -> None:
    """Validate a transition in a sibling temporary file before replacing state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary_path = Path(temporary.name)
    try:
        validate_lifecycle_registry(temporary_path, repo_root)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def record_published(
    registry_path: Path,
    repo_root: Path,
    *,
    release: str,
    bundle_sha256: str,
    release_url: str,
    published_at: str,
    supersedes: str | None,
) -> None:
    """Add an active release, optionally transitioning one active predecessor.

    This operation is valid only after the publisher has downloaded the public
    Release asset and passed ``baseline publication verify``.  When ``supersedes``
    is provided, this is the only permitted ``published -> superseded`` transition;
    both sides of the replacement edge are written together before validation.
    """
    if _SHA256_RE.fullmatch(bundle_sha256) is None:
        raise ValueError("bundle_sha256 must be a lowercase sha256 digest")
    if not release_url.startswith("https://github.com/"):
        raise ValueError("release_url must be a GitHub HTTPS URL")
    payload = _load_registry(registry_path)
    records = payload["records"]
    assert isinstance(records, list)
    if any(
        isinstance(record, dict) and record.get("release") == release
        for record in records
    ):
        raise ValueError("lifecycle release already exists")
    manifest_relative_path = f"docs/releases/{release}.evidence.json"
    manifest_path = _safe_repo_path(manifest_relative_path, repo_root, "manifest")
    manifest = load_evidence_publication_manifest(manifest_path)
    if manifest.release != release:
        raise ValueError("manifest release does not match requested lifecycle release")
    record: dict[str, Any] = {
        "release": release,
        "status": "published",
        "manifest": manifest_relative_path,
        "bundle_asset": f"{release}-evidence.tar.gz",
        "bundle_sha256": bundle_sha256,
        "release_url": release_url,
        "source_revision": manifest.source_revision,
        "published_at": published_at,
        "supersedes": supersedes,
        "replaced_by": None,
        "revocation_reason": None,
    }
    if supersedes is not None:
        predecessor = next(
            (
                candidate
                for candidate in records
                if isinstance(candidate, dict)
                and candidate.get("release") == supersedes
            ),
            None,
        )
        if predecessor is None or predecessor.get("status") != "published":
            raise ValueError("supersedes must reference a published lifecycle release")
        if "superseded" not in _ALLOWED_TRANSITIONS["published"]:
            raise ValueError("published evidence cannot transition to superseded")
        predecessor["status"] = "superseded"
        predecessor["replaced_by"] = release
    records.append(record)
    _write_validated_registry(registry_path, payload, repo_root)


def revoke_published(
    registry_path: Path,
    repo_root: Path,
    *,
    release: str,
    reason: str,
) -> None:
    """Perform the terminal ``published -> revoked`` transition without deletion."""
    if not reason.strip():
        raise ValueError("revocation reason must be non-empty")
    payload = _load_registry(registry_path)
    records = payload["records"]
    assert isinstance(records, list)
    record = next(
        (
            candidate
            for candidate in records
            if isinstance(candidate, dict) and candidate.get("release") == release
        ),
        None,
    )
    if record is None or record.get("status") != "published":
        raise ValueError("only a published lifecycle release can be revoked")
    if "revoked" not in _ALLOWED_TRANSITIONS["published"]:
        raise ValueError("published evidence cannot transition to revoked")
    record["status"] = "revoked"
    record["revocation_reason"] = reason
    _write_validated_registry(registry_path, payload, repo_root)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    command = parser.add_mutually_exclusive_group()
    command.add_argument("--record-published", action="store_true")
    command.add_argument("--revoke", action="store_true")
    parser.add_argument("--release")
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--release-url")
    parser.add_argument("--published-at")
    parser.add_argument("--supersedes")
    parser.add_argument("--reason")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.record_published:
            if not all(
                (args.release, args.bundle_sha256, args.release_url, args.published_at)
            ):
                raise ValueError(
                    "--record-published requires --release, --bundle-sha256, "
                    "--release-url, and --published-at"
                )
            record_published(
                args.registry,
                args.repo_root,
                release=args.release,
                bundle_sha256=args.bundle_sha256,
                release_url=args.release_url,
                published_at=args.published_at,
                supersedes=args.supersedes,
            )
        elif args.revoke:
            if not args.release or not args.reason:
                raise ValueError("--revoke requires --release and --reason")
            revoke_published(
                args.registry,
                args.repo_root,
                release=args.release,
                reason=args.reason,
            )
        else:
            validate_lifecycle_registry(args.registry, args.repo_root)
    except ValueError as exc:
        print(f"evidence lifecycle validation failed: {exc}")
        return 1
    print(f"Validated evidence lifecycle registry: {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
