# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed development and held-out diagnostic corpora."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.performance_model.diagnostic_schema_versions import (
    DiagnosticArtifactSchema,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    StrictArtifactModel,
)
from sol_execbench.core.integrity import (
    SHA256Digest,
    sha256_file,
    stable_json_checksum,
    validate_relative_artifact_path,
)

MINIMUM_CASES_PER_FAMILY = 20
MINIMUM_CORPUS_CASES = 220
_SUPPORTED_FAMILIES = frozenset(
    {
        WorkloadKind.ELEMENTWISE,
        WorkloadKind.TRANSPOSE,
        WorkloadKind.REDUCTION,
        WorkloadKind.MATMUL,
        WorkloadKind.SOFTMAX,
        WorkloadKind.CROSS_ENTROPY,
        WorkloadKind.INDEXED_READ,
        WorkloadKind.INDEXED_UPDATE,
        WorkloadKind.COMPOSITE,
        WorkloadKind.TRANSFORMER,
        WorkloadKind.CONCURRENT,
    }
)
_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ValidationArtifactReference(StrictArtifactModel):
    """One immutable tree-backed artifact input to a validation case.

    Tree-backed references resolve relative to the corpus root and are used by
    self-contained compact publications, where the case files live beside the
    corpus.
    """

    model_config = _CONFIG

    path: str = Field(min_length=1)
    sha256: SHA256Digest
    size_bytes: int = Field(ge=0)
    blob_backed: Literal[False] = False


class BlobArtifactReference(StrictArtifactModel):
    """One immutable blob-backed artifact input to a validation case.

    Blob-backed references carry no path; the SHA-256 key in the lifecycle
    blob store is the durable identity. Promotion targets these so historical
    path trees can be retired once unreachable.
    """

    model_config = _CONFIG

    sha256: SHA256Digest
    size_bytes: int = Field(ge=0)
    tree_manifest_sha256: SHA256Digest
    blob_backed: Literal[True] = True


CorpusArtifactReference = Annotated[
    BlobArtifactReference | ValidationArtifactReference,
    Field(discriminator="blob_backed"),
]


class DiagnosticValidationCase(StrictArtifactModel):
    """Labels and source references, deliberately excluding model output."""

    model_config = _CONFIG

    case_id: str = Field(min_length=1)
    pair_id: SHA256Digest
    workload_kind: WorkloadKind
    evidence_manifest: CorpusArtifactReference
    solar_manifest: CorpusArtifactReference
    gold_action_codes: list[str] = Field(default_factory=list)


class DiagnosticValidationCorpus(CurrentSchemaModel):
    """Frozen development or held-out cases for every admitted family."""

    model_config = _CONFIG
    current_schema_version = (
        DiagnosticArtifactSchema.DIAGNOSTIC_VALIDATION_CORPUS
    )

    schema_version: Literal[
        DiagnosticArtifactSchema.DIAGNOSTIC_VALIDATION_CORPUS
    ] = DiagnosticArtifactSchema.DIAGNOSTIC_VALIDATION_CORPUS
    purpose: DiagnosticEvidencePurpose = DiagnosticEvidencePurpose.PRODUCTION
    role: Literal["development", "held_out"]
    cases: list[DiagnosticValidationCase] = Field(
        min_length=MINIMUM_CORPUS_CASES
    )

    @model_validator(mode="after")
    def cases_are_complete_and_unique(self) -> DiagnosticValidationCorpus:
        """Require unique cases and at least twenty from each family."""
        case_ids = [case.case_id for case in self.cases]
        pair_ids = [case.pair_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("validation corpus repeats case_id")
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError(
                "validation corpus repeats workload/candidate pair"
            )
        evidence_hashes = [case.evidence_manifest.sha256 for case in self.cases]
        if len(evidence_hashes) != len(set(evidence_hashes)):
            raise ValueError("validation corpus repeats evidence manifest")
        kinds = {case.workload_kind for case in self.cases}
        if not kinds <= _SUPPORTED_FAMILIES:
            raise ValueError("validation corpus contains unsupported family")
        for kind in _SUPPORTED_FAMILIES:
            count = sum(case.workload_kind is kind for case in self.cases)
            if count < MINIMUM_CASES_PER_FAMILY:
                raise ValueError(f"validation corpus lacks {kind} coverage")
        return self


def verify_tree_reference(
    corpus_root: Path,
    reference: ValidationArtifactReference,
) -> Path:
    """Resolve and verify one tree-backed reference below the corpus root."""
    root = corpus_root.resolve()
    relative = validate_relative_artifact_path(reference.path)
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            "validation corpus artifact is missing or escapes root",
        )
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            "validation corpus artifact escapes its corpus root",
        )
    if resolved.stat().st_size != reference.size_bytes:
        raise ValueError("validation corpus artifact size mismatch")
    if sha256_file(resolved) != reference.sha256:
        raise ValueError("validation corpus artifact SHA-256 mismatch")
    return resolved


def require_disjoint_corpora(
    development: DiagnosticValidationCorpus,
    held_out: DiagnosticValidationCorpus,
) -> None:
    """Reject role mistakes or repeated workload/candidate pairs."""
    if development.role != "development" or held_out.role != "held_out":
        raise ValueError("validation corpus roles are invalid")
    pair_overlap = {case.pair_id for case in development.cases} & {
        case.pair_id for case in held_out.cases
    }
    if pair_overlap:
        raise ValueError("development and held-out corpora overlap")
    evidence_overlap = {
        case.evidence_manifest.sha256 for case in development.cases
    } & {case.evidence_manifest.sha256 for case in held_out.cases}
    if evidence_overlap:
        raise ValueError("development and held-out evidence manifests overlap")


def validation_pair_id(
    *,
    workload_sha256: str,
    candidate_sha256: str,
) -> SHA256Digest:
    """Derive the only admitted workload/candidate pair identity."""
    return stable_json_checksum(
        {
            "candidate_sha256": candidate_sha256,
            "workload_sha256": workload_sha256,
        }
    )


__all__ = [
    "MINIMUM_CASES_PER_FAMILY",
    "MINIMUM_CORPUS_CASES",
    "BlobArtifactReference",
    "CorpusArtifactReference",
    "DiagnosticValidationCase",
    "DiagnosticValidationCorpus",
    "ValidationArtifactReference",
    "require_disjoint_corpora",
    "validation_pair_id",
]
