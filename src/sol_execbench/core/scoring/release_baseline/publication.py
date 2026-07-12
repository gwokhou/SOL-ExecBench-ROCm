# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Git-tracked index for externally published release evidence.

Benchmark traces and profiler output are deliberately not source-controlled.
This module keeps their immutable identity in Git: a release manifest lists
every required artifact, its relative name, digest, source revision, and the
candidate implementation that produced the score.  Consumers download the
release bundle separately and verify it locally before making a published
authority claim.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from sol_execbench.core.integrity.checksums import sha256_file


EVIDENCE_PUBLICATION_MANIFEST_SCHEMA_VERSION = (
    "sol_execbench.evidence_publication_manifest.v1"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
_REQUIRED_ARTIFACT_ROLES = frozenset(
    {
        "candidate_solution",
        "candidate_trace",
        "candidate_timing",
        "scoring_baseline",
        "release_baseline_bundle",
        "release_baseline_verification",
        "suite_manifest",
        "official_score_evidence",
    }
)


def _non_empty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a 64-character lowercase sha256 digest")


def _relative_path(value: str, field: str) -> None:
    _non_empty(value, field)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path")


@dataclass(frozen=True)
class PublishedArtifact:
    """One immutable file in an externally published evidence bundle."""

    role: str
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        _non_empty(self.role, "artifact role")
        _relative_path(self.relative_path, "artifact relative_path")
        _sha256(self.sha256, "artifact sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class CandidateIdentity:
    """Content-addressed candidate implementation used by this evidence release."""

    solution_ref: str
    solution_sha256: str
    trace_relative_path: str
    trace_sha256: str
    timing_relative_path: str
    timing_sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.solution_ref, "candidate solution_ref")
        _sha256(self.solution_sha256, "candidate solution_sha256")
        _relative_path(self.trace_relative_path, "candidate trace_relative_path")
        _sha256(self.trace_sha256, "candidate trace_sha256")
        _relative_path(self.timing_relative_path, "candidate timing_relative_path")
        _sha256(self.timing_sha256, "candidate timing_sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "solution_ref": self.solution_ref,
            "solution_sha256": self.solution_sha256,
            "trace_relative_path": self.trace_relative_path,
            "trace_sha256": self.trace_sha256,
            "timing_relative_path": self.timing_relative_path,
            "timing_sha256": self.timing_sha256,
        }


@dataclass(frozen=True)
class EvidencePublicationManifest:
    """Versioned, Git-trackable locator and integrity contract for a release."""

    release: str
    scope: str
    source_repository: str
    source_revision: str
    container_image_digest: str
    artifact_base_uri: str
    candidate: CandidateIdentity
    artifacts: tuple[PublishedArtifact, ...]
    manifest_sha256: str | None = None
    schema_version: str = EVIDENCE_PUBLICATION_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in ("release", "scope", "source_repository", "artifact_base_uri"):
            _non_empty(getattr(self, field), field)
        if self.schema_version != EVIDENCE_PUBLICATION_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported evidence publication manifest schema")
        if _GIT_REVISION_RE.fullmatch(self.source_revision) is None:
            raise ValueError("source_revision must be a full lowercase Git object id")
        if not self.container_image_digest.startswith("sha256:"):
            raise ValueError("container_image_digest must use a sha256 digest")
        _sha256(
            self.container_image_digest.removeprefix("sha256:"),
            "container_image_digest",
        )
        parsed = urlparse(self.artifact_base_uri)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("artifact_base_uri must be an https URI")
        paths = [artifact.relative_path for artifact in self.artifacts]
        if not paths or len(paths) != len(set(paths)):
            raise ValueError("artifacts must be non-empty with unique relative paths")
        roles = [artifact.role for artifact in self.artifacts]
        if len(roles) != len(set(roles)):
            raise ValueError("artifacts must have unique roles")
        missing_roles = _REQUIRED_ARTIFACT_ROLES - set(roles)
        if missing_roles:
            raise ValueError(
                "publication manifest missing required artifact roles: "
                + ", ".join(sorted(missing_roles))
            )
        if self.candidate.solution_ref not in paths:
            raise ValueError("candidate solution_ref must be listed in artifacts")
        if self.candidate.trace_relative_path not in paths:
            raise ValueError(
                "candidate trace_relative_path must be listed in artifacts"
            )
        digests = {
            artifact.relative_path: artifact.sha256 for artifact in self.artifacts
        }
        if digests[self.candidate.solution_ref] != self.candidate.solution_sha256:
            raise ValueError("candidate solution digest must match its artifact")
        if digests[self.candidate.trace_relative_path] != self.candidate.trace_sha256:
            raise ValueError("candidate trace digest must match its artifact")
        if self.candidate.timing_relative_path not in paths:
            raise ValueError(
                "candidate timing_relative_path must be listed in artifacts"
            )
        if digests[self.candidate.timing_relative_path] != self.candidate.timing_sha256:
            raise ValueError("candidate timing digest must match its artifact")
        if self.manifest_sha256 is not None:
            _sha256(self.manifest_sha256, "manifest_sha256")
            if self.manifest_sha256 != self._checksum():
                raise ValueError("evidence publication manifest checksum mismatch")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "release": self.release,
            "scope": self.scope,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "container_image_digest": self.container_image_digest,
            "artifact_base_uri": self.artifact_base_uri,
            "candidate": self.candidate.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "manifest_sha256": self.manifest_sha256,
        }

    def with_checksum(self) -> "EvidencePublicationManifest":
        return replace(self, manifest_sha256=self._checksum())

    def _checksum(self) -> str:
        payload = self.payload()
        payload["manifest_sha256"] = None
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return self.with_checksum().payload()

    def verify_artifact_root(self, artifact_root: Path) -> None:
        """Ensure a downloaded release bundle exactly matches this manifest."""
        root = Path(artifact_root)
        failures: list[str] = []
        for artifact in self.artifacts:
            path = root / artifact.relative_path
            if not path.is_file():
                failures.append(f"missing:{artifact.relative_path}")
            elif sha256_file(path) != artifact.sha256:
                failures.append(f"checksum_mismatch:{artifact.relative_path}")
        if failures:
            raise ValueError(
                "published evidence verification failed: " + ", ".join(failures)
            )
        self._verify_authority_contract(root)

    def _verify_authority_contract(self, root: Path) -> None:
        """Verify that all published score inputs form one complete release."""
        by_role = {artifact.role: artifact for artifact in self.artifacts}
        try:
            payloads = {
                role: json.loads(
                    (root / artifact.relative_path).read_text(encoding="utf-8")
                )
                for role, artifact in by_role.items()
                if role in _REQUIRED_ARTIFACT_ROLES
            }
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"published authority artifact is not valid JSON: {exc}"
            ) from exc
        bundle = payloads["release_baseline_bundle"]
        verification = payloads["release_baseline_verification"]
        evidence = payloads["official_score_evidence"]
        if (
            not isinstance(bundle, dict)
            or _field(bundle, "schema_version")
            != "sol_execbench.release_baseline_bundle.v1"
        ):
            raise ValueError("published release baseline bundle has an invalid schema")
        if (
            not isinstance(verification, dict)
            or _field(verification, "schema_version")
            != "sol_execbench.release_baseline_verification.v1"
        ):
            raise ValueError(
                "published release baseline verification has an invalid schema"
            )
        if (
            not isinstance(evidence, dict)
            or _field(evidence, "schema_version")
            != "sol_execbench.official_score_evidence.v1"
        ):
            raise ValueError("published official score evidence has an invalid schema")
        if (
            _field(bundle, "release") != self.release
            or _field(bundle, "scope") != self.scope
        ):
            raise ValueError(
                "published bundle release or scope does not match manifest"
            )
        if evidence.get("scope") != self.scope or not evidence.get("score_authority"):
            raise ValueError(
                "published official score evidence is not authoritative for manifest scope"
            )
        if verification.get("release") != self.release:
            raise ValueError("published verification release does not match manifest")
        if bundle.get("baseline_artifact_sha256") != by_role["scoring_baseline"].sha256:
            raise ValueError("published scoring baseline does not match release bundle")
        if bundle.get("suite_manifest_sha256") != by_role["suite_manifest"].sha256:
            raise ValueError("published suite manifest does not match release bundle")
        if (
            verification.get("bundle_sha256")
            != by_role["release_baseline_bundle"].sha256
        ):
            raise ValueError("published verification does not match release bundle")
        candidate = evidence.get("candidate_evidence")
        if (
            not isinstance(candidate, dict)
            or candidate.get("solution_sha256") != self.candidate.solution_sha256
            or candidate.get("trace_sha256") != self.candidate.trace_sha256
            or candidate.get("timing_sha256") != self.candidate.timing_sha256
        ):
            raise ValueError(
                "published official score evidence does not bind manifest candidate"
            )

        digests = {artifact.sha256 for artifact in self.artifacts}
        if verification.get("rerun_trace_sha256") not in digests:
            raise ValueError("published rerun trace is absent from artifact manifest")
        workloads = bundle.get("workloads")
        if not isinstance(workloads, list):
            raise ValueError("published release bundle has no workload list")
        for row in workloads:
            if not isinstance(row, dict) or row.get("classification") != "official":
                continue
            for field in ("trace_sha256", "bound_sha256", "hardware_model_sha256"):
                if row.get(field) not in digests:
                    raise ValueError(
                        f"published official workload input is absent: {field}"
                    )
            bound_digest = row.get("bound_sha256")
            bound_artifact = next(
                (
                    artifact
                    for artifact in self.artifacts
                    if artifact.sha256 == bound_digest
                ),
                None,
            )
            if bound_artifact is None:
                continue
            try:
                bound = json.loads(
                    (root / bound_artifact.relative_path).read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("published AMD SOL bound is not valid JSON") from exc
            if (
                isinstance(bound, dict)
                and bound.get("schema_version") == "sol_execbench.amd_sol_bound.v4"
                and bound.get("fusion_validation_sha256") not in digests
            ):
                raise ValueError(
                    "published v4 bound fusion evidence is absent from artifact manifest"
                )


def evidence_publication_manifest_from_dict(
    payload: Mapping[str, Any],
) -> EvidencePublicationManifest:
    """Parse a strict publication manifest from JSON data."""
    artifacts_raw = payload.get("artifacts")
    candidate_raw = payload.get("candidate")
    if not isinstance(artifacts_raw, list) or not isinstance(candidate_raw, Mapping):
        raise ValueError("publication manifest requires candidate and artifacts")
    try:
        manifest = EvidencePublicationManifest(
            release=str(payload["release"]),
            scope=str(payload["scope"]),
            source_repository=str(payload["source_repository"]),
            source_revision=str(payload["source_revision"]),
            container_image_digest=str(payload["container_image_digest"]),
            artifact_base_uri=str(payload["artifact_base_uri"]),
            candidate=CandidateIdentity(
                solution_ref=str(candidate_raw["solution_ref"]),
                solution_sha256=str(candidate_raw["solution_sha256"]),
                trace_relative_path=str(candidate_raw["trace_relative_path"]),
                trace_sha256=str(candidate_raw["trace_sha256"]),
                timing_relative_path=str(candidate_raw["timing_relative_path"]),
                timing_sha256=str(candidate_raw["timing_sha256"]),
            ),
            artifacts=tuple(
                PublishedArtifact(
                    role=str(item["role"]),
                    relative_path=str(item["relative_path"]),
                    sha256=str(item["sha256"]),
                )
                for item in artifacts_raw
                if isinstance(item, Mapping)
            ),
            manifest_sha256=(
                str(payload["manifest_sha256"])
                if payload.get("manifest_sha256") is not None
                else None
            ),
            schema_version=str(payload.get("schema_version")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid evidence publication manifest: {exc}") from exc
    if len(manifest.artifacts) != len(artifacts_raw):
        raise ValueError("publication manifest artifacts must be objects")
    return manifest


def load_evidence_publication_manifest(path: Path) -> EvidencePublicationManifest:
    """Load and verify a Git-tracked evidence publication manifest."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid evidence publication manifest JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("evidence publication manifest must be a JSON object")
    return evidence_publication_manifest_from_dict(payload)


def _field(payload: Mapping[str, Any], name: str) -> Any:
    """Return a required JSON field without silently accepting its absence."""
    return payload[name] if name in payload else None
