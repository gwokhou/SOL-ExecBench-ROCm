# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Pinned release trust roots and detached Ed25519 verification."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.data.base_model import StrictArtifactModel
from sol_execbench.core.integrity import verify_artifact_file
from sol_execbench.core.platform.runtime import resolve_tool_path
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

from .release_models import ArtifactReference, AuthorityRole, SignedStatement


class AuthorityKey(StrictArtifactModel):
    """One repository-pinned public key for a single authority role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key_id: str
    role: AuthorityRole
    public_key: ArtifactReference

    @model_validator(mode="after")
    def _identity(self) -> "AuthorityKey":
        if self.key_id != self.public_key.sha256:
            raise ValueError("authority key id must equal its public-key SHA-256")
        if not 0 < self.public_key.size_bytes <= 16 * 1024:
            raise ValueError("authority public key size is invalid")
        return self


class ReleaseAuthority(StrictArtifactModel):
    """Threshold-free four-role trust policy for one corpus release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    keys: tuple[AuthorityKey, ...] = Field(min_length=4)
    max_rerun_relative_delta: float = Field(gt=0.0, le=0.25)

    @model_validator(mode="after")
    def _contract(self) -> "ReleaseAuthority":
        identities = [item.key_id for item in self.keys]
        roles = [item.role for item in self.keys]
        if self.schema_version != 1:
            raise ValueError("release authority schema mismatch")
        if len(identities) != len(set(identities)):
            raise ValueError("release authority key ids must be unique")
        if len(roles) != len(set(roles)) or set(roles) != set(AuthorityRole):
            raise ValueError("release authority requires one key for every role")
        return self

    def key_for(self, statement: SignedStatement) -> AuthorityKey:
        matches = [
            item
            for item in self.keys
            if item.key_id == statement.key_id and item.role == statement.role
        ]
        if len(matches) != 1:
            raise ValueError("signed statement uses an untrusted authority key")
        return matches[0]


def load_release_authority(manifest_path: Path) -> ReleaseAuthority:
    """Load the strict trust policy embedded in a corpus manifest."""
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("could not parse corpus release authority") from exc
    if not isinstance(payload, dict) or not isinstance(
        payload.get("release_authority"), dict
    ):
        raise ValueError("corpus manifest has no release authority")
    return ReleaseAuthority.model_validate(payload["release_authority"])


def verify_signed_statement(
    statement: SignedStatement,
    *,
    bundle_root: Path,
    corpus_root: Path,
    authority: ReleaseAuthority,
) -> Path:
    """Verify one statement identity and detached Ed25519 signature."""
    payload = verify_artifact_file(
        bundle_root,
        statement.payload.path,
        expected_sha256=statement.payload.sha256,
        expected_size_bytes=statement.payload.size_bytes,
    )
    signature = verify_artifact_file(
        bundle_root,
        statement.signature.path,
        expected_sha256=statement.signature.sha256,
        expected_size_bytes=statement.signature.size_bytes,
    )
    key = authority.key_for(statement)
    public_key = verify_artifact_file(
        corpus_root,
        key.public_key.path,
        expected_sha256=key.public_key.sha256,
        expected_size_bytes=key.public_key.size_bytes,
    )
    _verify_ed25519(payload, signature, public_key)
    return payload


def _verify_ed25519(payload: Path, signature: Path, public_key: Path) -> None:
    openssl = resolve_tool_path("openssl")
    if openssl is None:
        raise RuntimeError("OpenSSL is required to verify release attestations")
    completed = run_in_process_group_bounded(
        [
            str(openssl),
            "pkeyutl",
            "-verify",
            "-pubin",
            "-inkey",
            str(public_key),
            "-rawin",
            "-in",
            str(payload),
            "-sigfile",
            str(signature),
        ],
        timeout=30,
        max_capture_bytes=4096,
    )
    if completed.returncode != 0:
        raise ValueError("release attestation signature verification failed")


__all__ = [
    "AuthorityKey",
    "ReleaseAuthority",
    "load_release_authority",
    "verify_signed_statement",
]
