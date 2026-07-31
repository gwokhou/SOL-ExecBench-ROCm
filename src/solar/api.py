# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Public, benchmark-agnostic boundary for paper-defined SOLAR analysis."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from solar.contracts import (
    FORMAL_BOUND_KIND,
    ROOFLINE_BOUND_KIND,
    SOL_BOUND_KINDS,
    AnalysisFailure,
    AnalysisRequest,
    AnalysisResult,
    ArtifactRef,
    ConversionRequest,
    ExtractionKind,
    FormalProducerReadiness,
    IRKind,
    IRPath,
    SolarAnalysisStatus,
    SolarStage,
    SOLBound,
    VerificationPolicy,
    write_request_manifest,
)
from solar.pipeline import (
    PipelineStageError,
    pipeline_reason_code,
    run_pipeline,
)
from solar.pipeline.readiness import (
    ConversionReadinessRequest,
    ConversionReadinessResult,
    ReadinessArtifact,
    ReadinessStage,
    audit_conversion,
)
from solar.rocm.architecture import ArchitectureProfile
from solar.schema_versions import SOLAR_ANALYSIS_SCHEMA_VERSION


def analyze(request: AnalysisRequest) -> AnalysisResult | AnalysisFailure:
    """Run the complete SOLAR responsibility boundary atomically."""
    output = Path(request.output_dir).resolve()
    if output.exists():
        return _failure(
            request,
            SolarStage.PREPARE,
            "output_exists",
            f"exists: {output}",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent),
    )
    stage = SolarStage.ARCHITECTURE
    try:
        profile = ArchitectureProfile.load(request.architecture)
        if isinstance(profile, ArchitectureProfile):
            profile.require_verified_audit_evidence()
        architecture_sha256 = _profile_hash(profile)
        pipeline = run_pipeline(request, profile, staging)
        ir_graph, analysis = pipeline.ir_graph, pipeline.analysis
        _attach_analysis_metadata(analysis, request.analysis_metadata)
        bound = _extract_bound(analysis, request.require_orojenesis)
        artifacts = _finish_artifacts(staging, analysis, ir_graph.path.name)
        write_request_manifest(
            request,
            staging,
            architecture_sha256,
            artifacts,
            bound,
            formal_bound_kind=FORMAL_BOUND_KIND,
        )
        _verify_publication_tree(staging, artifacts)
        staging.replace(output)
        return AnalysisResult(
            status=SolarAnalysisStatus.ANALYZED,
            analysis_id=request.analysis_id,
            output_dir=output,
            architecture_sha256=architecture_sha256,
            ir_path=request.ir_path,
            artifacts=artifacts,
            bound=bound,
        )
    except Exception as exc:  # noqa: BLE001 -- fail-closed public boundary
        if isinstance(exc, PipelineStageError):
            stage, exc = exc.stage, exc.error
        shutil.rmtree(staging, ignore_errors=True)
        return _failure(
            request, stage, pipeline_reason_code(stage, exc), str(exc)
        )


def _attach_analysis_metadata(
    analysis: dict[str, Any],
    requested: Mapping[str, Any],
) -> None:
    """Attach caller-authored semantic metadata before artifact hashing."""
    if not requested:
        return
    metadata = analysis.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("analysis metadata is not a mapping")
    metadata = cast("dict[str, Any]", metadata)
    collisions = sorted(set(metadata) & set(requested))
    if collisions:
        raise ValueError(
            f"analysis metadata keys are reserved: {collisions}",
        )
    metadata.update(requested)


def formal_producer_readiness() -> FormalProducerReadiness:
    """Return whether the reviewed formal mapper is allowlisted."""
    from solar.analysis.orojenesis import OROJENESIS_TRUSTED_MAPPER_SHA256

    if not OROJENESIS_TRUSTED_MAPPER_SHA256:
        return FormalProducerReadiness(False, "formal_mapper_not_allowlisted")
    return FormalProducerReadiness(True, "ready")


def architecture_profile_sha256(
    architecture: str | Path | Mapping[str, Any],
) -> str:
    """Return the canonical identity of a packaged architecture profile."""
    return _profile_hash(ArchitectureProfile.load(architecture))


def _extract_bound(
    analysis: Mapping[str, Any],
    require_orojenesis: bool = True,
) -> SOLBound:
    if analysis.get("schema_version") != SOLAR_ANALYSIS_SCHEMA_VERSION:
        raise ValueError("formal analysis uses an unsupported schema")
    total = analysis.get("total") or {}
    metadata = analysis.get("metadata") or {}
    seconds = total.get("lower_bound_seconds")
    kind = str(metadata.get("bound_kind", ""))
    if (
        seconds is None
        or not math.isfinite(float(seconds))
        or float(seconds) < 0
    ):
        raise ValueError("formal analysis lacks a finite lower bound")
    if require_orojenesis:
        if kind != FORMAL_BOUND_KIND:
            raise ValueError(
                f"strict analysis returned non-tile-aware bound kind {kind!r}",
            )
    elif kind not in SOL_BOUND_KINDS:
        raise ValueError(f"analysis returned unsupported bound kind {kind!r}")
    resource = total.get("compute_resource")
    return SOLBound(float(seconds), kind, str(resource) if resource else None)


def _finish_artifacts(
    staging: Path,
    analysis: dict[str, Any],
    graph_name: str,
) -> tuple[ArtifactRef, ...]:
    metadata = analysis.get("metadata") or {}
    metadata["source_graph"] = graph_name
    analysis["metadata"] = metadata
    old_path = staging / "analysis.yaml"
    analysis_path = staging / "solar-analysis.yaml"
    old_path.unlink(missing_ok=True)
    analysis_path.write_text(yaml.safe_dump(analysis, sort_keys=False))
    required = {
        "operator_graph.yaml",
        graph_name,
        "conversion-attestation.yaml",
        "solar-analysis.yaml",
    }
    paths = _publication_files(staging)
    missing = sorted(
        required - {path.relative_to(staging).as_posix() for path in paths}
    )
    if missing:
        raise ValueError(f"analysis is missing required artifacts: {missing}")
    return tuple(
        ArtifactRef(path.relative_to(staging).as_posix(), _sha256(path))
        for path in paths
    )


def _publication_files(staging: Path) -> tuple[Path, ...]:
    entries = tuple(staging.rglob("*"))
    symlinks = sorted(
        path.relative_to(staging).as_posix()
        for path in entries
        if path.is_symlink()
    )
    if symlinks:
        raise ValueError(
            f"analysis staging tree contains symbolic links: {symlinks}",
        )
    return tuple(
        sorted(
            (
                path
                for path in entries
                if path.is_file() and path.name != "manifest.yaml"
            ),
            key=lambda path: path.relative_to(staging).as_posix(),
        ),
    )


def _verify_publication_tree(
    staging: Path,
    artifacts: tuple[ArtifactRef, ...],
) -> None:
    manifest_path = staging / "manifest.yaml"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("analysis staging tree lacks a regular manifest")
    expected = {artifact.path for artifact in artifacts} | {"manifest.yaml"}
    actual = {
        path.relative_to(staging).as_posix()
        for path in _publication_files(staging)
    } | {"manifest.yaml"}
    if actual != expected:
        raise ValueError(
            "analysis staging tree changed after artifact hashing",
        )


def _profile_hash(profile: ArchitectureProfile) -> str:
    encoded = json.dumps(
        profile.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _failure(
    request: AnalysisRequest,
    stage: SolarStage,
    reason_code: str,
    message: str,
) -> AnalysisFailure:
    return AnalysisFailure(
        status=SolarAnalysisStatus.FAILED,
        analysis_id=request.analysis_id,
        ir_path=request.ir_path,
        stage=stage,
        reason_code=reason_code,
        message=message,
    )


__all__ = [
    "FORMAL_BOUND_KIND",
    "ROOFLINE_BOUND_KIND",
    "SOL_BOUND_KINDS",
    "AnalysisFailure",
    "AnalysisRequest",
    "AnalysisResult",
    "ArtifactRef",
    "ConversionReadinessRequest",
    "ConversionReadinessResult",
    "ConversionRequest",
    "ExtractionKind",
    "FormalProducerReadiness",
    "IRKind",
    "IRPath",
    "ReadinessArtifact",
    "ReadinessStage",
    "SOLBound",
    "SolarAnalysisStatus",
    "SolarStage",
    "VerificationPolicy",
    "analyze",
    "architecture_profile_sha256",
    "audit_conversion",
    "formal_producer_readiness",
]
