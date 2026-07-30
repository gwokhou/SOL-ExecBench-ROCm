"""Pinned Orojenesis artifact and provenance validation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from solar.analysis.orojenesis.errors import OrojenesisError
from solar.schema_versions import OROJENESIS_PROVENANCE_SCHEMA_VERSION
from solar.types import DynamicValue


@dataclass(frozen=True)
class OrojenesisIdentityPolicy:
    """Expected source, artifact, and reproducible-build identities."""

    repository: str
    commit: str
    tree_oid: str
    source_archive_sha256: str
    compiler_wrapper_sha256: str
    builder_image: str
    ubuntu_snapshot: str
    openssl_sha256: str
    ca_certificates_sha256: str
    source_date_epoch: int
    provenance_filename: str
    trusted_mapper_sha256: frozenset[str]


def validate_toolchain_identity(
    home: Path,
    mapper: Path,
    policy: OrojenesisIdentityPolicy,
) -> dict[str, DynamicValue]:
    """Validate a mapper and return its authenticated provenance record."""
    if not mapper.is_file() or not os.access(mapper, os.X_OK):
        raise OrojenesisError(f"missing executable: {mapper}")
    binary_sha256 = _sha256(mapper)
    if binary_sha256 not in policy.trusted_mapper_sha256:
        raise OrojenesisError(
            "Orojenesis mapper artifact is not trusted by this SOLAR release",
        )
    provenance_path = home / policy.provenance_filename
    provenance = _load_provenance(provenance_path)
    _validate_source(provenance.get("source") or {}, policy)
    _validate_artifact(
        provenance.get("artifact") or {},
        home,
        mapper,
        binary_sha256,
    )
    _validate_build(provenance.get("build") or {}, policy)
    return {
        **provenance,
        "verification_mode": "provenance_manifest",
        "provenance_sha256": _sha256(provenance_path),
    }


def _load_provenance(path: Path) -> dict[str, DynamicValue]:
    if not path.is_file():
        raise OrojenesisError(
            "Orojenesis provenance manifest is required for formal analysis",
        )
    try:
        provenance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrojenesisError("cannot parse Orojenesis provenance") from exc
    if not isinstance(provenance, dict):
        raise OrojenesisError("Orojenesis provenance must be an object")
    if provenance.get("schema_version") != OROJENESIS_PROVENANCE_SCHEMA_VERSION:
        raise OrojenesisError("unsupported Orojenesis provenance schema")
    return provenance


def _validate_source(
    source: dict[str, DynamicValue],
    policy: OrojenesisIdentityPolicy,
) -> None:
    if source.get("repository") != policy.repository:
        raise OrojenesisError("Orojenesis provenance repository mismatch")
    if source.get("commit") != policy.commit:
        raise OrojenesisError(
            "Orojenesis provenance revision mismatch: expected "
            f"{policy.commit}, got {source.get('commit')}",
        )
    if str(source.get("tree_git_oid", "")) != policy.tree_oid:
        raise OrojenesisError("Orojenesis provenance source tree mismatch")
    if str(source.get("archive_sha256", "")) != policy.source_archive_sha256:
        raise OrojenesisError(
            "Orojenesis provenance source archive mismatch",
        )


def _validate_artifact(
    artifact: dict[str, DynamicValue],
    home: Path,
    mapper: Path,
    binary_sha256: str,
) -> None:
    artifact_path = Path(str(artifact.get("path", "")))
    if (
        artifact_path.is_absolute()
        or ".." in artifact_path.parts
        or (home / artifact_path).resolve() != mapper
    ):
        raise OrojenesisError(
            "Orojenesis provenance artifact path mismatch",
        )
    recorded_binary = str(artifact.get("sha256", ""))
    if len(recorded_binary) != 64 or any(
        character not in "0123456789abcdef" for character in recorded_binary
    ):
        raise OrojenesisError(
            "Orojenesis provenance lacks a binary SHA-256",
        )
    if recorded_binary != binary_sha256:
        raise OrojenesisError("Orojenesis mapper binary hash mismatch")


def _validate_build(
    build: dict[str, DynamicValue],
    policy: OrojenesisIdentityPolicy,
) -> None:
    if (
        str(build.get("compiler_wrapper_sha256", ""))
        != policy.compiler_wrapper_sha256
    ):
        raise OrojenesisError(
            "Orojenesis provenance compiler-wrapper mismatch",
        )
    if build.get("builder_image") != policy.builder_image:
        raise OrojenesisError("Orojenesis provenance builder image mismatch")
    if build.get("ubuntu_snapshot") != policy.ubuntu_snapshot:
        raise OrojenesisError("Orojenesis provenance Ubuntu snapshot mismatch")
    packages = build.get("bootstrap_packages") or {}
    if (
        packages.get("openssl") != policy.openssl_sha256
        or packages.get("ca-certificates") != policy.ca_certificates_sha256
    ):
        raise OrojenesisError(
            "Orojenesis provenance bootstrap package mismatch",
        )
    if build.get("package_source_mode") != "snapshot_only":
        raise OrojenesisError(
            "Orojenesis provenance package source mismatch",
        )
    if build.get("source_date_epoch") != policy.source_date_epoch:
        raise OrojenesisError("Orojenesis provenance source epoch mismatch")
    if not str(build.get("compiler", "")):
        raise OrojenesisError("Orojenesis provenance lacks build identity")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
