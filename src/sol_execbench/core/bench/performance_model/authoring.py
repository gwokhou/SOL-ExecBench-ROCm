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
    SemanticCharacterizationLoader,
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
from sol_execbench.core.bench.performance_model.lifecycle.resolver import (
    ReferenceResolver,
    resolve_corpus_reference,
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
    semantic_loader: SemanticCharacterizationLoader,
    blob_resolver: ReferenceResolver | None = None,
) -> DiagnosticInferenceProfile:
    """Fit only from a declared development corpus and frozen calibration."""
    corpus = load_json_file(
        DiagnosticValidationCorpus,
        development_corpus_path,
    )
    if corpus.role != "development":
        raise ValueError("inference fitting requires development corpus")
    calibration = load_json_file(
        DiagnosticCalibrationProfile,
        calibration_profile_path,
    )
    if calibration.purpose is not corpus.purpose:
        raise ValueError("development corpus/calibration purpose mismatch")
    audit_path = calibration_profile_path.with_name(
        f"{calibration_profile_path.stem}.audit.json"
    )
    observations = [
        _development_observation(
            case,
            corpus_path=development_corpus_path,
            calibration_path=calibration_profile_path,
            semantic_loader=semantic_loader,
            blob_resolver=blob_resolver,
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
        purpose=corpus.purpose,
    )


def build_diagnostic_acceptance(
    *,
    development_corpus_path: Path,
    held_out_corpus_path: Path,
    calibration_profile_path: Path,
    inference_profile_path: Path,
    semantic_loader: SemanticCharacterizationLoader,
    blob_resolver: ReferenceResolver | None = None,
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
    purposes = {
        development.purpose,
        held_out.purpose,
        profile.purpose,
        calibration.purpose,
    }
    if len(purposes) != 1:
        raise ValueError("acceptance inputs cross evidence purpose domains")
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
            semantic_loader=semantic_loader,
            blob_resolver=blob_resolver,
        )
        for case in held_out.cases
    ]
    manifest = DiagnosticAcceptanceManifest(
        purpose=development.purpose,
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


def verify_diagnostic_acceptance(
    *,
    acceptance: DiagnosticAcceptanceResult,
    manifest: DiagnosticAcceptanceManifest,
    development_corpus_path: Path,
    held_out_corpus_path: Path,
    calibration_profile_path: Path,
    inference_profile_path: Path,
    semantic_loader: SemanticCharacterizationLoader,
    blob_resolver: ReferenceResolver | None = None,
) -> None:
    """Rebuild held-out acceptance from source evidence and reject any drift."""
    rebuilt_manifest, rebuilt_result = build_diagnostic_acceptance(
        development_corpus_path=development_corpus_path,
        held_out_corpus_path=held_out_corpus_path,
        calibration_profile_path=calibration_profile_path,
        inference_profile_path=inference_profile_path,
        semantic_loader=semantic_loader,
        blob_resolver=blob_resolver,
    )
    if not _acceptance_manifests_equivalent(rebuilt_manifest, manifest):
        raise ValueError(
            "acceptance manifest disagrees with source corpus evidence"
        )
    if rebuilt_result.model_dump(
        exclude={"manifest_sha256"}
    ) != acceptance.model_dump(exclude={"manifest_sha256"}):
        raise ValueError(
            "acceptance result disagrees with source corpus evidence"
        )


def _acceptance_manifests_equivalent(
    rebuilt: DiagnosticAcceptanceManifest,
    submitted: DiagnosticAcceptanceManifest,
) -> bool:
    """Compare authoritative case values while ignoring path-bound citations.

    A diagnostic sidecar's evidence list retains the caller's artifact path, so
    identical content loaded through an alias has a different whole-sidecar
    checksum. Acceptance authority comes from the re-derived prediction,
    measurement, actions, and cited evidence hashes, not that path-bound audit
    citation.
    """
    if rebuilt.model_dump(exclude={"cases"}) != submitted.model_dump(
        exclude={"cases"}
    ):
        return False
    return all(
        rebuilt_case.model_dump(exclude={"performance_diagnostic_sha256"})
        == submitted_case.model_dump(exclude={"performance_diagnostic_sha256"})
        for rebuilt_case, submitted_case in zip(
            rebuilt.cases,
            submitted.cases,
            strict=True,
        )
    )


def _development_observation(
    case: DiagnosticValidationCase,
    *,
    corpus_path: Path,
    calibration_path: Path,
    semantic_loader: SemanticCharacterizationLoader,
    blob_resolver: ReferenceResolver | None = None,
) -> InferenceObservation:
    diagnostic = _build_case_diagnostic(
        case,
        corpus_path=corpus_path,
        calibration_path=calibration_path,
        inference_path=None,
        semantic_loader=semantic_loader,
        blob_resolver=blob_resolver,
    )
    workload = diagnostic.workloads[0]
    predicted_ms, lower_ms, upper_ms = _prediction_values(
        workload.t_pred_hw,
        case_id=case.case_id,
    )
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
    semantic_loader: SemanticCharacterizationLoader,
    blob_resolver: ReferenceResolver | None = None,
) -> DiagnosticAcceptanceCase:
    diagnostic = _build_case_diagnostic(
        case,
        corpus_path=corpus_path,
        calibration_path=calibration_path,
        inference_path=inference_path,
        semantic_loader=semantic_loader,
        blob_resolver=blob_resolver,
    )
    workload = diagnostic.workloads[0]
    predicted_ms, lower_ms, upper_ms = _prediction_values(
        workload.t_pred_hw,
        case_id=case.case_id,
    )
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
    semantic_loader: SemanticCharacterizationLoader,
    blob_resolver: ReferenceResolver | None = None,
) -> PerformanceDiagnosticSidecar:
    evidence = resolve_corpus_reference(
        case.evidence_manifest,
        resolver=blob_resolver,
        corpus_root=corpus_path.parent,
    )
    solar = resolve_corpus_reference(
        case.solar_manifest,
        resolver=blob_resolver,
        corpus_root=corpus_path.parent,
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
        ),
        semantic_loader=semantic_loader,
    )
    if len(diagnostic.workloads) != 1:
        raise ValueError("validation case must produce one workload")
    if diagnostic.workloads[0].semantic.workload_kind is not case.workload_kind:
        raise ValueError("validation case workload family mismatch")
    return diagnostic


def _prediction_values(
    prediction: PerformancePrediction,
    *,
    case_id: str,
) -> tuple[float, float, float]:
    if (
        prediction.predicted_time_ms is None
        or prediction.lower_ms is None
        or prediction.upper_ms is None
    ):
        raise ValueError(
            f"{case_id} validation case lacks an available HW prediction: "
            f"{prediction.reason_codes}"
        )
    return (
        prediction.predicted_time_ms,
        prediction.lower_ms,
        prediction.upper_ms,
    )


__all__ = [
    "build_diagnostic_acceptance",
    "fit_diagnostic_inference_profile",
    "verify_diagnostic_acceptance",
]
