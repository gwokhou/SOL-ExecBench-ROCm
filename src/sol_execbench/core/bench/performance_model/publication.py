# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Governed compact publication of diagnostic model inputs."""

from __future__ import annotations

import shutil
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, field_validator, model_validator

from sol_execbench.core.bench.diagnostic_sidecar import (
    CurrentDiagnosticSidecarAuthority,
)
from sol_execbench.core.bench.performance_model.authoring import (
    fit_diagnostic_inference_profile,
)
from sol_execbench.core.bench.performance_model.builder import (
    SemanticCharacterizationLoader,
)
from sol_execbench.core.bench.performance_model.evidence_manifest import (
    PerformanceEvidenceArtifact,
    PerformanceEvidenceArtifactKind,
    PerformanceEvidenceManifest,
    load_and_verify_performance_evidence_manifest,
)
from sol_execbench.core.bench.performance_model.inference import (
    DiagnosticInferenceProfile,
)
from sol_execbench.core.bench.performance_model.lifecycle.resolver import (
    ReferenceResolver,
    resolve_corpus_reference,
)
from sol_execbench.core.bench.performance_model.models import (
    DiagnosticCalibrationProfile,
)
from sol_execbench.core.bench.performance_model.validation_corpus import (
    CorpusArtifactReference,
    DiagnosticValidationCase,
    DiagnosticValidationCorpus,
    ValidationArtifactReference,
    validation_pair_id,
)
from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceKernel,
    StaticKernelEvidenceSidecar,
    StaticResourceFootprint,
)
from sol_execbench.core.data.base_model import FrozenArtifactModel
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
)
from sol_execbench.core.integrity import (
    SHA256Digest,
    sha256_file,
    validate_relative_artifact_path,
    verify_artifact_file,
)
from sol_execbench.core.integrity.schema_versions import SchemaVersion

PUBLICATION_MANIFEST_NAME = "publication.json"
_SOURCE_INFERENCE_PATH = "source-inference.json"
_PROJECTED_INFERENCE_PATH = "inference.json"
_CORPUS_PATH = "development.json"
_CALIBRATION_PATH = "calibration/profile.json"
_CALIBRATION_AUDIT_PATH = "calibration/profile.audit.json"


class SolarManifestProjector(Protocol):
    """Boundary adapter for compacting one verified SOLAR manifest."""

    def __call__(
        self,
        source: Path,
        destination: Path,
        *,
        expected_definition: str,
        expected_workload_uuid: str,
    ) -> Path:
        """Project one source manifest into *destination*."""
        ...


class SolarManifestProjectionVerifier(Protocol):
    """Boundary adapter for verifying one compact SOLAR manifest."""

    def __call__(self, path: Path) -> None:
        """Verify the projected manifest and every cited artifact."""
        ...


class DiagnosticPublicationPolicy(StrEnum):
    """Closed publication projection policies."""

    COMPACT_MODEL_INPUTS_V1 = "compact_model_inputs_v1"


class DiagnosticPublicationArtifact(FrozenArtifactModel):
    """One regular file in a diagnostic publication tree."""

    path: str
    sha256: SHA256Digest
    size_bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def _safe_path(cls, value: str) -> str:
        return validate_relative_artifact_path(value, "publication path")


class DiagnosticPublicationProjection(CurrentDiagnosticSidecarAuthority):
    """Exact compact development corpus and reproducible inference inputs."""

    current_schema_version = SchemaVersion.DIAGNOSTIC_PUBLICATION_PROJECTION

    schema_version: Literal[SchemaVersion.DIAGNOSTIC_PUBLICATION_PROJECTION] = (
        SchemaVersion.DIAGNOSTIC_PUBLICATION_PROJECTION
    )
    policy: Literal[DiagnosticPublicationPolicy.COMPACT_MODEL_INPUTS_V1] = (
        DiagnosticPublicationPolicy.COMPACT_MODEL_INPUTS_V1
    )
    role: Literal["development"] = "development"
    case_count: int = Field(ge=220)
    source_corpus_sha256: SHA256Digest
    corpus: DiagnosticPublicationArtifact
    calibration_profile: DiagnosticPublicationArtifact
    calibration_audit: DiagnosticPublicationArtifact
    source_inference_profile: DiagnosticPublicationArtifact
    inference_profile: DiagnosticPublicationArtifact
    omitted_performance_artifact_kinds: tuple[Literal["rocpd"]] = ("rocpd",)
    omitted_solar_artifact_prefixes: tuple[Literal["orojenesis/"]] = (
        "orojenesis/",
    )
    artifacts: tuple[DiagnosticPublicationArtifact, ...] = Field(min_length=1)
    uncompressed_size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _inventory_is_exact(self) -> DiagnosticPublicationProjection:
        paths = [item.path for item in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("diagnostic publication repeats artifact path")
        if paths != sorted(paths):
            raise ValueError("diagnostic publication inventory is not sorted")
        indexed = {item.path: item for item in self.artifacts}
        for required, canonical_path in (
            (self.corpus, _CORPUS_PATH),
            (self.calibration_profile, _CALIBRATION_PATH),
            (self.calibration_audit, _CALIBRATION_AUDIT_PATH),
            (self.source_inference_profile, _SOURCE_INFERENCE_PATH),
            (self.inference_profile, _PROJECTED_INFERENCE_PATH),
        ):
            if required.path != canonical_path:
                raise ValueError("publication input uses a noncanonical path")
            if indexed.get(required.path) != required:
                raise ValueError("publication input is absent from inventory")
        if self.uncompressed_size_bytes != sum(
            item.size_bytes for item in self.artifacts
        ):
            raise ValueError("publication byte total disagrees with inventory")
        return self


def build_diagnostic_publication_projection(
    *,
    development_corpus_path: Path,
    calibration_profile_path: Path,
    source_inference_profile_path: Path,
    output_root: Path,
    semantic_loader: SemanticCharacterizationLoader,
    solar_projector: SolarManifestProjector,
    solar_verifier: SolarManifestProjectionVerifier,
    blob_resolver: ReferenceResolver | None = None,
) -> Path:
    """Build and atomically publish a compact, self-verifying corpus tree."""
    corpus_path = development_corpus_path.resolve()
    calibration_path = calibration_profile_path.resolve()
    source_inference_path = source_inference_profile_path.resolve()
    output = output_root.resolve()
    _require_new_output(
        output, (corpus_path, calibration_path, source_inference_path)
    )
    corpus, source_inference, audit_path = _load_source_inputs(
        corpus_path,
        calibration_path,
        source_inference_path,
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        _copy_publication_inputs(
            staging,
            calibration_path,
            audit_path,
            source_inference_path,
        )
        projected_corpus = DiagnosticValidationCorpus(
            role="development",
            cases=[
                _project_case(
                    case,
                    index,
                    corpus_path,
                    staging,
                    solar_projector=solar_projector,
                    blob_resolver=blob_resolver,
                )
                for index, case in enumerate(corpus.cases)
            ],
        )
        atomic_write_json_value(
            staging / _CORPUS_PATH,
            projected_corpus.model_dump(mode="json"),
        )
        projected_inference = _fit_projected_inference(
            staging,
            semantic_loader=semantic_loader,
            blob_resolver=blob_resolver,
        )
        _require_inference_equivalence(source_inference, projected_inference)
        _write_publication_manifest(
            staging,
            source_corpus_sha256=sha256_file(corpus_path),
            case_count=len(corpus.cases),
        )
        verify_diagnostic_publication_projection(
            staging / PUBLICATION_MANIFEST_NAME,
            semantic_loader=semantic_loader,
            solar_verifier=solar_verifier,
            blob_resolver=blob_resolver,
        )
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / PUBLICATION_MANIFEST_NAME


def verify_diagnostic_publication_projection(
    manifest_path: Path,
    *,
    semantic_loader: SemanticCharacterizationLoader,
    solar_verifier: SolarManifestProjectionVerifier,
    blob_resolver: ReferenceResolver | None = None,
) -> DiagnosticPublicationProjection:
    """Verify the exact tree and reproduce its projected inference profile."""
    path = manifest_path.resolve()
    root = path.parent
    projection = load_json_file(DiagnosticPublicationProjection, path)
    _verify_inventory(root, projection)
    corpus_path = _verify_publication_ref(root, projection.corpus)
    calibration_path = _verify_publication_ref(
        root, projection.calibration_profile
    )
    source_profile_path = _verify_publication_ref(
        root, projection.source_inference_profile
    )
    projected_profile_path = _verify_publication_ref(
        root, projection.inference_profile
    )
    corpus = load_json_file(DiagnosticValidationCorpus, corpus_path)
    if (
        corpus.role != "development"
        or len(corpus.cases) != projection.case_count
    ):
        raise ValueError("publication corpus role or case count mismatch")
    for case in corpus.cases:
        _verify_projected_case(
            case,
            corpus_path,
            solar_verifier=solar_verifier,
            blob_resolver=blob_resolver,
        )
    source_profile = load_json_file(
        DiagnosticInferenceProfile, source_profile_path
    )
    projected_profile = load_json_file(
        DiagnosticInferenceProfile, projected_profile_path
    )
    if (
        source_profile.development_corpus_sha256
        != projection.source_corpus_sha256
    ):
        raise ValueError("source inference does not bind source corpus")
    rebuilt = fit_diagnostic_inference_profile(
        development_corpus_path=corpus_path,
        calibration_profile_path=calibration_path,
        semantic_loader=semantic_loader,
        blob_resolver=blob_resolver,
    )
    if rebuilt != projected_profile:
        raise ValueError("publication inference does not reproduce")
    _require_inference_equivalence(source_profile, projected_profile)
    return projection


def _require_new_output(output: Path, inputs: tuple[Path, ...]) -> None:
    if output.exists():
        raise FileExistsError(
            f"diagnostic publication already exists: {output}"
        )
    for path in inputs:
        source_root = path.parent
        if output.is_relative_to(source_root) or source_root.is_relative_to(
            output
        ):
            raise ValueError(
                "publication output and process input directories must be isolated"
            )
    output.parent.mkdir(parents=True, exist_ok=True)


def _load_source_inputs(
    corpus_path: Path,
    calibration_path: Path,
    inference_path: Path,
) -> tuple[
    DiagnosticValidationCorpus,
    DiagnosticInferenceProfile,
    Path,
]:
    corpus = load_json_file(DiagnosticValidationCorpus, corpus_path)
    if corpus.role != "development":
        raise ValueError("diagnostic publication requires development corpus")
    load_json_file(DiagnosticCalibrationProfile, calibration_path)
    inference = load_json_file(DiagnosticInferenceProfile, inference_path)
    audit_path = calibration_path.with_name(
        f"{calibration_path.stem}.audit.json"
    )
    if not audit_path.is_file():
        raise ValueError("diagnostic calibration audit is missing")
    if inference.development_corpus_sha256 != sha256_file(corpus_path):
        raise ValueError("source inference does not bind development corpus")
    if inference.calibration_profile_sha256 != sha256_file(calibration_path):
        raise ValueError("source inference does not bind calibration profile")
    if inference.calibration_audit_sha256 != sha256_file(audit_path):
        raise ValueError("source inference does not bind calibration audit")
    return corpus, inference, audit_path


def _copy_publication_inputs(
    staging: Path,
    calibration_path: Path,
    audit_path: Path,
    source_inference_path: Path,
) -> None:
    _copy_regular_file(calibration_path, staging / _CALIBRATION_PATH)
    _copy_regular_file(audit_path, staging / _CALIBRATION_AUDIT_PATH)
    _copy_regular_file(source_inference_path, staging / _SOURCE_INFERENCE_PATH)


def _project_case(
    case: DiagnosticValidationCase,
    index: int,
    corpus_path: Path,
    staging: Path,
    *,
    solar_projector: SolarManifestProjector,
    blob_resolver: ReferenceResolver | None = None,
) -> DiagnosticValidationCase:
    source_evidence = _verified_corpus_reference(
        corpus_path, case.evidence_manifest, blob_resolver=blob_resolver
    )
    source_solar = _verified_corpus_reference(
        corpus_path, case.solar_manifest, blob_resolver=blob_resolver
    )
    destination = staging / "cases" / f"{index:04d}"
    evidence_path, evidence = _project_performance_manifest(
        source_evidence, destination / "performance"
    )
    solar_path = solar_projector(
        source_solar,
        destination / "solar",
        expected_definition=evidence.identity.definition,
        expected_workload_uuid=evidence.identity.workload_uuid,
    )
    return case.model_copy(
        update={
            "evidence_manifest": _validation_reference(staging, evidence_path),
            "solar_manifest": _validation_reference(staging, solar_path),
        }
    )


def _verified_corpus_reference(
    corpus_path: Path,
    reference: CorpusArtifactReference,
    *,
    blob_resolver: ReferenceResolver | None = None,
) -> Path:
    return resolve_corpus_reference(
        reference,
        resolver=blob_resolver,
        corpus_root=corpus_path.parent,
    )


def _project_performance_manifest(
    source: Path, destination: Path
) -> tuple[Path, PerformanceEvidenceManifest]:
    manifest = load_and_verify_performance_evidence_manifest(
        source, require_complete=True
    )
    destination.mkdir(parents=True, exist_ok=False)
    artifacts = [
        _project_performance_artifact(source, destination, artifact)
        for artifact in manifest.artifacts
        if artifact.kind is not PerformanceEvidenceArtifactKind.ROCPD
    ]
    projected = manifest.model_copy(update={"artifacts": artifacts})
    output = destination / "manifest.json"
    atomic_write_json_value(output, projected.model_dump(mode="json"))
    verified = load_and_verify_performance_evidence_manifest(
        output, require_complete=True
    )
    return output, verified


def _project_performance_artifact(
    manifest_path: Path,
    destination: Path,
    artifact: PerformanceEvidenceArtifact,
) -> PerformanceEvidenceArtifact:
    source = verify_artifact_file(
        manifest_path.parent,
        artifact.path,
        expected_sha256=artifact.sha256,
        expected_size_bytes=artifact.size_bytes,
    )
    output = destination / artifact.path
    if artifact.kind is PerformanceEvidenceArtifactKind.STATIC_EVIDENCE:
        _write_projected_static_evidence(source, output)
    else:
        _copy_regular_file(source, output)
    return artifact.model_copy(
        update={
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
        }
    )


def _write_projected_static_evidence(source: Path, output: Path) -> None:
    sidecar = load_json_file(StaticKernelEvidenceSidecar, source)
    projected = sidecar.model_copy(
        update={
            "artifacts": [
                _project_static_artifact(item) for item in sidecar.artifacts
            ],
            "tool_runs": [],
            "kernels": [
                _project_static_kernel(item) for item in sidecar.kernels
            ],
            "footprints": [
                _project_footprint(item) for item in sidecar.footprints
            ],
            "warnings": [],
            "source_references": [],
            "summary": None,
        }
    )
    atomic_write_json_value(output, projected.to_dict())


def _project_static_artifact(
    artifact: StaticKernelEvidenceArtifact,
) -> StaticKernelEvidenceArtifact:
    return artifact.model_copy(
        update={
            "source_path": None,
            "persisted_path": None,
            "source_references": [],
        }
    )


def _project_static_kernel(
    kernel: StaticKernelEvidenceKernel,
) -> StaticKernelEvidenceKernel:
    return kernel.model_copy(
        update={
            "source_references": [],
            "footprint": (
                _project_footprint(kernel.footprint)
                if kernel.footprint is not None
                else None
            ),
        }
    )


def _project_footprint(
    footprint: StaticResourceFootprint,
) -> StaticResourceFootprint:
    return footprint.model_copy(update={"source_references": []})


def _copy_regular_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"publication source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _validation_reference(
    root: Path, path: Path
) -> ValidationArtifactReference:
    return ValidationArtifactReference(
        path=path.relative_to(root).as_posix(),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _fit_projected_inference(
    staging: Path,
    *,
    semantic_loader: SemanticCharacterizationLoader,
    blob_resolver: ReferenceResolver | None = None,
) -> DiagnosticInferenceProfile:
    profile = fit_diagnostic_inference_profile(
        development_corpus_path=staging / _CORPUS_PATH,
        calibration_profile_path=staging / _CALIBRATION_PATH,
        semantic_loader=semantic_loader,
        blob_resolver=blob_resolver,
    )
    atomic_write_json_value(
        staging / _PROJECTED_INFERENCE_PATH, profile.model_dump(mode="json")
    )
    return profile


def _require_inference_equivalence(
    source: DiagnosticInferenceProfile,
    projected: DiagnosticInferenceProfile,
) -> None:
    expected = source.model_copy(
        update={
            "development_corpus_sha256": projected.development_corpus_sha256
        }
    )
    if expected != projected:
        raise ValueError(
            "compact projection changes frozen inference semantics"
        )


def _write_publication_manifest(
    staging: Path, *, source_corpus_sha256: str, case_count: int
) -> None:
    artifacts = _artifact_inventory(staging)
    indexed = {item.path: item for item in artifacts}
    projection = DiagnosticPublicationProjection(
        case_count=case_count,
        source_corpus_sha256=source_corpus_sha256,
        corpus=indexed[_CORPUS_PATH],
        calibration_profile=indexed[_CALIBRATION_PATH],
        calibration_audit=indexed[_CALIBRATION_AUDIT_PATH],
        source_inference_profile=indexed[_SOURCE_INFERENCE_PATH],
        inference_profile=indexed[_PROJECTED_INFERENCE_PATH],
        artifacts=artifacts,
        uncompressed_size_bytes=sum(item.size_bytes for item in artifacts),
    )
    atomic_write_json_value(
        staging / PUBLICATION_MANIFEST_NAME, projection.model_dump(mode="json")
    )


def _artifact_inventory(
    root: Path,
) -> tuple[DiagnosticPublicationArtifact, ...]:
    return tuple(
        DiagnosticPublicationArtifact(
            path=path.relative_to(root).as_posix(),
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / PUBLICATION_MANIFEST_NAME
    )


def _verify_inventory(
    root: Path, projection: DiagnosticPublicationProjection
) -> None:
    expected = {item.path for item in projection.artifacts}
    observed: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("diagnostic publication contains a symlink")
        if path.is_file() and path != root / PUBLICATION_MANIFEST_NAME:
            observed.add(path.relative_to(root).as_posix())
    if observed != expected:
        raise ValueError("diagnostic publication file inventory mismatch")
    for artifact in projection.artifacts:
        _verify_publication_ref(root, artifact)


def _verify_publication_ref(
    root: Path, artifact: DiagnosticPublicationArtifact
) -> Path:
    return verify_artifact_file(
        root,
        artifact.path,
        expected_sha256=artifact.sha256,
        expected_size_bytes=artifact.size_bytes,
    )


def _verify_projected_case(
    case: DiagnosticValidationCase,
    corpus_path: Path,
    *,
    solar_verifier: SolarManifestProjectionVerifier,
    blob_resolver: ReferenceResolver | None = None,
) -> None:
    evidence_path = _verified_corpus_reference(
        corpus_path, case.evidence_manifest, blob_resolver=blob_resolver
    )
    solar_path = _verified_corpus_reference(
        corpus_path, case.solar_manifest, blob_resolver=blob_resolver
    )
    evidence = load_and_verify_performance_evidence_manifest(
        evidence_path, require_complete=True
    )
    if evidence.artifacts_of_kind(PerformanceEvidenceArtifactKind.ROCPD):
        raise ValueError("compact publication retains ROCPD evidence")
    if case.pair_id != validation_pair_id(
        workload_sha256=evidence.identity.workload_sha256,
        candidate_sha256=evidence.identity.candidate_sha256,
    ):
        raise ValueError("publication validation pair identity mismatch")
    solar_verifier(solar_path)


__all__ = [
    "PUBLICATION_MANIFEST_NAME",
    "DiagnosticPublicationArtifact",
    "DiagnosticPublicationPolicy",
    "DiagnosticPublicationProjection",
    "SolarManifestProjectionVerifier",
    "SolarManifestProjector",
    "build_diagnostic_publication_projection",
    "verify_diagnostic_publication_projection",
]
