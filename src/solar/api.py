# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Public, benchmark-agnostic boundary for paper-defined SOLAR analysis."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from solar.artifacts import stable_json_checksum
from solar.contracts import (
    FORMAL_BOUND_KIND,
    ROOFLINE_BOUND_KIND,
    SOL_BOUND_KINDS,
    AnalysisExecutionPolicy,
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
    ConversionReadinessRequest,
    ConversionReadinessResult,
    PipelineStageError,
    ReadinessArtifact,
    ReadinessStage,
    audit_conversion,
    pipeline_reason_code,
    run_pipeline,
)
from solar.pipeline.publication import (
    extract_bound,
    finish_artifacts,
    verify_publication_tree,
)
from solar.rocm.architecture import ArchitectureProfile


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
        if isinstance(profile, ArchitectureProfile) and (
            request.require_verified_audit
        ):
            profile.require_verified_audit_evidence()
        architecture_sha256 = _profile_hash(profile)
        pipeline = run_pipeline(request, profile, staging)
        ir_graph, analysis = pipeline.ir_graph, pipeline.analysis
        stage = SolarStage.FORMAL_ANALYSIS
        _attach_analysis_metadata(analysis, request.analysis_metadata)
        bound = extract_bound(analysis, request.require_orojenesis)
        artifacts = finish_artifacts(staging, analysis, ir_graph.path.name)
        write_request_manifest(
            request,
            staging,
            architecture_sha256,
            artifacts,
            bound,
            formal_bound_kind=FORMAL_BOUND_KIND,
        )
        verify_publication_tree(staging, artifacts)
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


def _profile_hash(profile: ArchitectureProfile) -> str:
    return stable_json_checksum(profile.to_dict())


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
    "AnalysisExecutionPolicy",
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
