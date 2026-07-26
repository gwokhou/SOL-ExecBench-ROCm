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
from typing import Any

import yaml

from solar.analysis.graph_analyzer import (
    SOLAR_ANALYSIS_SCHEMA_VERSION,
    ArchitectureProfile,
    EinsumGraphAnalyzer,
    OrojenesisError,
    OrojenesisRunner,
)
from solar.contracts import (
    FORMAL_BOUND_KIND,
    AnalysisFailure,
    AnalysisRequest,
    AnalysisResult,
    ArtifactRef,
    SolBound,
    SolarAnalysisStatus,
    SolarStage,
    write_request_manifest,
)
from solar.einsum.conversion import convert_operator_graph
from solar.graph.extraction import extract_operator_graph
from solar.readiness import (
    ConversionReadinessRequest,
    ConversionReadinessResult,
    ReadinessArtifact,
    ReadinessStage,
    audit_conversion,
)
from solar.verification import VerificationError, verify_callable_conversion


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
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    stage = SolarStage.ARCHITECTURE
    try:
        profile = ArchitectureProfile.load(request.architecture)
        if isinstance(profile, ArchitectureProfile):
            profile.require_verified_audit_evidence()
        architecture_sha256 = _profile_hash(profile)
        stage = SolarStage.GRAPH_EXTRACTION
        inputs = tuple(request.input_factory(request.trace_seed))
        operator = extract_operator_graph(
            request.reference,
            inputs,
            device=request.device,
            output_dir=staging,
            name=request.analysis_id,
        )
        stage = SolarStage.EINSUM_CONVERSION
        einsum = convert_operator_graph(operator, output_dir=staging)
        stage = SolarStage.CONVERSION_VERIFICATION
        verify_callable_conversion(
            reference=request.reference,
            input_factory=request.input_factory,
            reference_name=request.reference_name,
            reference_sha256=request.reference_sha256,
            graph_path=einsum.path,
            output_path=staging / "conversion-attestation.yaml",
            atol=request.atol,
            rtol=request.rtol,
            required_matched_ratio=request.required_matched_ratio,
            max_error_cap=request.max_error_cap,
            allow_negative_inf=request.allow_negative_inf,
            seeds=request.verification_seeds,
            device=request.device,
        )
        stage = SolarStage.FORMAL_ANALYSIS
        analysis = _run_analysis(request, profile, staging)
        bound = _extract_bound(analysis, request.require_orojenesis)
        artifacts = _finish_artifacts(staging, analysis)
        write_request_manifest(
            request,
            staging,
            architecture_sha256,
            artifacts,
            bound,
            formal_bound_kind=FORMAL_BOUND_KIND,
        )
        staging.replace(output)
        return AnalysisResult(
            status=SolarAnalysisStatus.ANALYZED,
            analysis_id=request.analysis_id,
            output_dir=output,
            architecture_sha256=architecture_sha256,
            artifacts=artifacts,
            bound=bound,
        )
    except Exception as exc:  # fail closed at the public boundary
        shutil.rmtree(staging, ignore_errors=True)
        return _failure(request, stage, _reason_code(stage, exc), str(exc))


def _run_analysis(
    request: AnalysisRequest, profile: ArchitectureProfile, staging: Path
) -> dict[str, Any]:
    runner = (
        OrojenesisRunner(request.orojenesis_home)
        if request.require_orojenesis or request.orojenesis_home is not None
        else None
    )
    result = EinsumGraphAnalyzer().analyze_graph(
        staging / "einsum_graph.yaml",
        staging,
        precision=request.precision,
        copy_graph=False,
        strict=True,
        architecture=profile,
        orojenesis_runner=runner,
        require_orojenesis=request.require_orojenesis,
    )
    if result is None:
        raise RuntimeError("strict graph analysis produced no artifact")
    return result


def _extract_bound(
    analysis: Mapping[str, Any], require_orojenesis: bool = True
) -> SolBound:
    if analysis.get("schema_version") != SOLAR_ANALYSIS_SCHEMA_VERSION:
        raise ValueError("formal analysis uses an unsupported schema")
    total = analysis.get("total") or {}
    metadata = analysis.get("metadata") or {}
    seconds = total.get("lower_bound_seconds")
    kind = str(metadata.get("bound_kind", ""))
    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) < 0:
        raise ValueError("formal analysis lacks a finite lower bound")
    if require_orojenesis:
        if kind != FORMAL_BOUND_KIND:
            raise ValueError(f"formal analysis returned non-formal bound kind {kind!r}")
    elif kind not in (FORMAL_BOUND_KIND, "diagnostic"):
        raise ValueError(f"analysis returned unsupported bound kind {kind!r}")
    resource = total.get("compute_resource")
    return SolBound(float(seconds), kind, str(resource) if resource else None)


def _finish_artifacts(
    staging: Path, analysis: dict[str, Any]
) -> tuple[ArtifactRef, ...]:
    metadata = analysis.get("metadata") or {}
    metadata["source_graph"] = "einsum_graph.yaml"
    analysis["metadata"] = metadata
    old_path = staging / "analysis.yaml"
    analysis_path = staging / "solar-analysis.yaml"
    old_path.unlink(missing_ok=True)
    analysis_path.write_text(yaml.safe_dump(analysis, sort_keys=False))
    names = (
        "operator_graph.yaml",
        "einsum_graph.yaml",
        "conversion-attestation.yaml",
        "solar-analysis.yaml",
    )
    return tuple(ArtifactRef(name, _sha256(staging / name)) for name in names)


def _profile_hash(profile: ArchitectureProfile) -> str:
    encoded = json.dumps(
        profile.to_dict(), sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reason_code(stage: SolarStage, exc: Exception) -> str:
    if isinstance(exc, OrojenesisError):
        return "toolchain_unavailable"
    if isinstance(exc, VerificationError):
        return "conversion_not_proven"
    return f"{stage}_failed"


def _failure(
    request: AnalysisRequest,
    stage: SolarStage,
    reason_code: str,
    message: str,
) -> AnalysisFailure:
    return AnalysisFailure(
        status=SolarAnalysisStatus.FAILED,
        analysis_id=request.analysis_id,
        stage=stage,
        reason_code=reason_code,
        message=message,
    )


__all__ = [
    "AnalysisFailure",
    "AnalysisRequest",
    "AnalysisResult",
    "ArtifactRef",
    "ConversionReadinessRequest",
    "ConversionReadinessResult",
    "FORMAL_BOUND_KIND",
    "ReadinessArtifact",
    "ReadinessStage",
    "SolarAnalysisStatus",
    "SolarStage",
    "SolBound",
    "analyze",
    "audit_conversion",
]
