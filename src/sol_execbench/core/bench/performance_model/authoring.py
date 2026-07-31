# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Internal authoring workflows for frozen inference and acceptance artifacts."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.bench.performance_model.acceptance import (
    DiagnosticAcceptanceCase,
    DiagnosticAcceptanceManifest,
    DiagnosticAcceptanceResult,
    evaluate_diagnostic_acceptance,
)
from sol_execbench.core.bench.performance_model.attribution import action_scores
from sol_execbench.core.bench.performance_model.builder import (
    PerformanceDiagnosticBuildRequest,
    build_performance_diagnostic,
)
from sol_execbench.core.bench.performance_model.evidence_manifest import (
    load_and_verify_performance_evidence_manifest,
)
from sol_execbench.core.bench.performance_model.inference import (
    DiagnosticInferenceProfile,
    InferenceObservation,
    build_inference_profile,
    point_features,
)
from sol_execbench.core.bench.performance_model.model_identity import (
    build_diagnostic_model_identity,
)
from sol_execbench.core.bench.performance_model.models import (
    PERFORMANCE_MODEL_VERSION,
    DiagnosticCalibrationProfile,
    PerformanceDiagnosticSidecar,
    PerformancePrediction,
)
from sol_execbench.core.bench.performance_model.validation_corpus import (
    DiagnosticValidationCase,
    DiagnosticValidationCorpus,
    require_disjoint_corpora,
    validation_pair_id,
)
from sol_execbench.core.data.json_utils import load_json_file
from sol_execbench.core.integrity import sha256_file, stable_json_checksum


def fit_diagnostic_inference_profile(
    *,
    development_corpus_path: Path,
    calibration_profile_path: Path,
) -> DiagnosticInferenceProfile:
    """Fit only from a declared development corpus and frozen calibration."""
    corpus = load_json_file(
        DiagnosticValidationCorpus,
        development_corpus_path,
    )
    if corpus.role != "development":
        raise ValueError("inference fitting requires development corpus")
    load_json_file(
        DiagnosticCalibrationProfile,
        calibration_profile_path,
    )
    audit_path = calibration_profile_path.with_name(
        f"{calibration_profile_path.stem}.audit.json"
    )
    observations = [
        _development_observation(
            case,
            corpus_path=development_corpus_path,
            calibration_path=calibration_profile_path,
        )
        for case in corpus.cases
    ]
    return build_inference_profile(
        observations,
        model_identity=build_diagnostic_model_identity(
            PERFORMANCE_MODEL_VERSION
        ),
        calibration_profile_sha256=sha256_file(calibration_profile_path),
        calibration_audit_sha256=sha256_file(audit_path),
        development_corpus_sha256=sha256_file(development_corpus_path),
    )


def build_diagnostic_acceptance(
    *,
    development_corpus_path: Path,
    held_out_corpus_path: Path,
    calibration_profile_path: Path,
    inference_profile_path: Path,
) -> tuple[DiagnosticAcceptanceManifest, DiagnosticAcceptanceResult]:
    """Evaluate the frozen profile once against disjoint held-out evidence."""
    development = load_json_file(
        DiagnosticValidationCorpus,
        development_corpus_path,
    )
    held_out = load_json_file(
        DiagnosticValidationCorpus,
        held_out_corpus_path,
    )
    require_disjoint_corpora(development, held_out)
    profile = load_json_file(
        DiagnosticInferenceProfile,
        inference_profile_path,
    )
    calibration = load_json_file(
        DiagnosticCalibrationProfile,
        calibration_profile_path,
    )
    if profile.development_corpus_sha256 != sha256_file(
        development_corpus_path
    ):
        raise ValueError("inference profile development corpus mismatch")
    cases = [
        _acceptance_case(
            case,
            corpus_path=held_out_corpus_path,
            calibration_path=calibration_profile_path,
            inference_path=inference_profile_path,
        )
        for case in held_out.cases
    ]
    manifest = DiagnosticAcceptanceManifest(
        model_identity=profile.model_identity,
        calibration_profile_sha256=sha256_file(calibration_profile_path),
        calibration_identity=calibration.identity,
        inference_profile_sha256=sha256_file(inference_profile_path),
        development_corpus_sha256=sha256_file(development_corpus_path),
        held_out_corpus_sha256=sha256_file(held_out_corpus_path),
        enabled_action_codes=sorted(profile.enabled_action_codes),
        cases=cases,
    )
    return manifest, evaluate_diagnostic_acceptance(manifest)


def _development_observation(
    case: DiagnosticValidationCase,
    *,
    corpus_path: Path,
    calibration_path: Path,
) -> InferenceObservation:
    diagnostic = _build_case_diagnostic(
        case,
        corpus_path=corpus_path,
        calibration_path=calibration_path,
        inference_path=None,
    )
    workload = diagnostic.workloads[0]
    predicted_ms, lower_ms, upper_ms = _prediction_values(workload.t_pred_hw)
    return InferenceObservation(
        case_id=case.case_id,
        workload_kind=case.workload_kind,
        measured_ms=workload.t_measured_ms,
        base_predicted_ms=predicted_ms,
        base_lower_ms=lower_ms,
        base_upper_ms=upper_ms,
        point_features=point_features(workload.semantic),
        action_scores=action_scores(
            semantic=workload.semantic,
            compiled=workload.compiled,
            dispatches=workload.dispatches,
            t_pred_ir=workload.t_pred_ir,
        ),
        gold_action_codes=case.gold_action_codes,
    )


def _acceptance_case(
    case: DiagnosticValidationCase,
    *,
    corpus_path: Path,
    calibration_path: Path,
    inference_path: Path,
) -> DiagnosticAcceptanceCase:
    diagnostic = _build_case_diagnostic(
        case,
        corpus_path=corpus_path,
        calibration_path=calibration_path,
        inference_path=inference_path,
    )
    workload = diagnostic.workloads[0]
    predicted_ms, lower_ms, upper_ms = _prediction_values(workload.t_pred_hw)
    return DiagnosticAcceptanceCase(
        case_id=case.case_id,
        pair_id=case.pair_id,
        workload_kind=case.workload_kind,
        evidence_manifest_sha256=case.evidence_manifest.sha256,
        performance_diagnostic_sha256=stable_json_checksum(
            diagnostic.model_dump(mode="json")
        ),
        predicted_ms=predicted_ms,
        lower_ms=lower_ms,
        upper_ms=upper_ms,
        measured_ms=workload.t_measured_ms,
        predicted_action_codes=[
            attribution.action_code
            for attribution in workload.attributions
            if attribution.action_code is not None
        ],
        gold_action_codes=case.gold_action_codes,
    )


def _build_case_diagnostic(
    case: DiagnosticValidationCase,
    *,
    corpus_path: Path,
    calibration_path: Path,
    inference_path: Path | None,
) -> PerformanceDiagnosticSidecar:
    evidence = _verified_case_path(
        corpus_path,
        case.evidence_manifest.path,
        case.evidence_manifest.sha256,
    )
    solar = _verified_case_path(
        corpus_path,
        case.solar_manifest.path,
        case.solar_manifest.sha256,
    )
    manifest = load_and_verify_performance_evidence_manifest(evidence)
    expected_pair_id = validation_pair_id(
        workload_sha256=manifest.identity.workload_sha256,
        candidate_sha256=manifest.identity.candidate_sha256,
    )
    if case.pair_id != expected_pair_id:
        raise ValueError("validation case pair identity mismatch")
    diagnostic = build_performance_diagnostic(
        PerformanceDiagnosticBuildRequest(
            evidence_manifest_path=evidence,
            solar_manifest_path=solar,
            output_path=corpus_path.parent
            / ".diagnostic-authoring-unused.json",
            calibration_profile_path=calibration_path,
            inference_profile_path=inference_path,
        )
    )
    if len(diagnostic.workloads) != 1:
        raise ValueError("validation case must produce one workload")
    if diagnostic.workloads[0].semantic.workload_kind is not case.workload_kind:
        raise ValueError("validation case workload family mismatch")
    return diagnostic


def _verified_case_path(
    corpus_path: Path,
    relative_path: str,
    expected_sha256: str,
) -> Path:
    path = (corpus_path.parent / relative_path).resolve()
    try:
        path.relative_to(corpus_path.parent.resolve())
    except ValueError as error:
        raise ValueError("validation artifact escapes corpus root") from error
    if sha256_file(path) != expected_sha256:
        raise ValueError("validation artifact hash mismatch")
    return path


def _prediction_values(
    prediction: PerformancePrediction,
) -> tuple[float, float, float]:
    if (
        prediction.predicted_time_ms is None
        or prediction.lower_ms is None
        or prediction.upper_ms is None
    ):
        raise ValueError("validation case lacks an available HW prediction")
    return (
        prediction.predicted_time_ms,
        prediction.lower_ms,
        prediction.upper_ms,
    )


__all__ = [
    "build_diagnostic_acceptance",
    "fit_diagnostic_inference_profile",
]
