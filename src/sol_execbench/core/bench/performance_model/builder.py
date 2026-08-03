# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Artifact-governed construction of performance diagnostic sidecars."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Protocol

from sol_execbench.core.bench.diagnostic_sidecar import (
    DiagnosticGovernanceStatus,
    DiagnosticSidecarStatus,
)
from sol_execbench.core.bench.performance_model.access_evidence import (
    AccessPatternSummary,
    PerformanceAccessEvidenceSidecar,
)
from sol_execbench.core.bench.performance_model.attribution import (
    calculate_ratios,
    derive_attributions,
)
from sol_execbench.core.bench.performance_model.calibration_audit import (
    DiagnosticCalibrationAudit,
)
from sol_execbench.core.bench.performance_model.evidence_manifest import (
    PerformanceEvidenceArtifactKind,
    PerformanceEvidenceManifest,
    load_and_verify_performance_evidence_manifest,
)
from sol_execbench.core.bench.performance_model.inference import (
    DiagnosticInferenceProfile,
    apply_conformal_interval,
    point_features,
)
from sol_execbench.core.bench.performance_model.kernel_identity import (
    kernel_symbol_key,
)
from sol_execbench.core.bench.performance_model.model_identity import (
    build_diagnostic_model_identity,
)
from sol_execbench.core.bench.performance_model.models import (
    PERFORMANCE_MODEL_VERSION,
    CompiledCharacterization,
    DiagnosticCalibrationProfile,
    DispatchEvidence,
    EvidenceReference,
    PerformanceDiagnosticSidecar,
    PerformancePrediction,
    PredictionKind,
    ResourceFootprint,
    SemanticCharacterization,
    WorkloadPerformanceDiagnostic,
)
from sol_execbench.core.bench.performance_model.prediction import (
    predict_hw,
    predict_ir,
    validate_calibration_identity,
)
from sol_execbench.core.bench.performance_model.replay_evidence import (
    PerformanceReplayEvidenceSidecar,
)
from sol_execbench.core.bench.performance_model.schedule_evidence import (
    build_schedule_evidence,
)
from sol_execbench.core.bench.performance_model.timing_evidence import (
    PerformanceTimingEvidenceSidecar,
    WorkloadTimingEvidence,
)
from sol_execbench.core.bench.profile_summary import (
    ProfileSummarySidecar,
    evaluate_profile_summary_governance,
    validate_profile_summary_freshness,
)
from sol_execbench.core.bench.rocm_profiler.counter_provenance import (
    Rocprofv3CounterProvenance,
)
from sol_execbench.core.bench.rocm_profiler.counters import (
    CounterPassCSV,
    counter_pass_index,
    parse_and_align_counter_passes,
)
from sol_execbench.core.bench.static_kernel.evidence import (
    StaticISAAnalysis,
    StaticKernelEvidenceKernel,
    StaticKernelEvidenceSidecar,
)
from sol_execbench.core.data.json_utils import (
    load_json_file,
    load_jsonl_file,
)
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.integrity import (
    sha256_file,
    stable_json_checksum,
    verify_artifact_file,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PerformanceDiagnosticBuildRequest:
    """Immutable paths and identity inputs for one diagnostic build."""

    evidence_manifest_path: Path
    solar_manifest_path: Path
    output_path: Path
    calibration_profile_path: Path | None = None
    inference_profile_path: Path | None = None
    frontier_trace_path: Path | None = None


class SemanticCharacterizationLoader(Protocol):
    """Boundary adapter for loading validated SOLAR semantics."""

    def __call__(
        self,
        manifest_path: Path,
        *,
        workload_uuid: str,
        definition: str,
    ) -> SemanticCharacterization:
        """Load one identity-bound semantic characterization."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class _BuildEvidence:
    manifest: PerformanceEvidenceManifest
    trace: Trace
    timing: WorkloadTimingEvidence
    access_patterns: list[AccessPatternSummary]
    compiled: list[CompiledCharacterization]
    calibration: DiagnosticCalibrationProfile | None
    inference: DiagnosticInferenceProfile | None
    dispatches: list[DispatchEvidence]
    semantic: SemanticCharacterization
    identity_reasons: list[str]
    dispatch_reasons: list[str]


@dataclass(frozen=True, slots=True)
class _BuildArtifactPaths:
    trace: Path
    timing: Path
    access: Path
    profile: Path
    static: Path


@dataclass(frozen=True, slots=True, kw_only=True)
class _WorkloadDiagnosticInputs:
    trace: Trace
    semantic: SemanticCharacterization
    timing: WorkloadTimingEvidence
    access_patterns: list[AccessPatternSummary]
    frontier_path: Path | None
    compiled: list[CompiledCharacterization]
    dispatches: list[DispatchEvidence]
    calibration: DiagnosticCalibrationProfile | None
    inference: DiagnosticInferenceProfile | None
    identity_reasons: list[str]
    extra_reasons: list[str]


def build_performance_diagnostic(
    request: PerformanceDiagnosticBuildRequest,
    *,
    semantic_loader: SemanticCharacterizationLoader,
) -> PerformanceDiagnosticSidecar:
    """Validate all input identities and build one diagnostic-only sidecar."""
    evidence = _load_build_evidence(request, semantic_loader=semantic_loader)
    manifest = evidence.manifest
    workload = _workload_diagnostic(
        _WorkloadDiagnosticInputs(
            trace=evidence.trace,
            semantic=evidence.semantic,
            timing=evidence.timing,
            access_patterns=evidence.access_patterns,
            frontier_path=request.frontier_trace_path,
            compiled=evidence.compiled,
            dispatches=evidence.dispatches,
            calibration=evidence.calibration,
            inference=evidence.inference,
            identity_reasons=evidence.identity_reasons,
            extra_reasons=[
                *manifest.reason_codes,
                *evidence.dispatch_reasons,
            ],
        )
    )
    reasons = list(
        dict.fromkeys(
            [
                *manifest.reason_codes,
                *evidence.identity_reasons,
                *evidence.dispatch_reasons,
                *workload.reason_codes,
            ]
        )
    )
    return PerformanceDiagnosticSidecar(
        status=_sidecar_status([workload], reasons),
        model_identity=build_diagnostic_model_identity(
            PERFORMANCE_MODEL_VERSION,
        ),
        inference_profile_sha256=(
            sha256_file(request.inference_profile_path)
            if request.inference_profile_path is not None
            else None
        ),
        run_id=manifest.identity.run_id,
        candidate_sha256=manifest.identity.candidate_sha256,
        gpu_architecture=manifest.identity.gpu_architecture,
        calibration_identity=(
            evidence.calibration.identity if evidence.calibration else None
        ),
        workloads=[workload],
        evidence=[
            *_input_references(request),
            *_manifest_references(
                request.evidence_manifest_path,
                manifest,
            ),
        ],
        reason_codes=reasons,
        limitations=[
            "Performance predictions are diagnostic-only and do not affect SOL Score.",
            "Canonical Trace JSONL remains timing and correctness authority.",
            "Profiler duration and achieved rates are excluded from T_pred(HW).",
            "Counter evidence is a post-canonical, single-workload replay.",
            "Only scope-verified multi-queue overlap topology is admitted.",
        ],
    )


def _load_build_evidence(
    request: PerformanceDiagnosticBuildRequest,
    *,
    semantic_loader: SemanticCharacterizationLoader,
) -> _BuildEvidence:
    manifest = load_and_verify_performance_evidence_manifest(
        request.evidence_manifest_path
    )
    paths = _build_artifact_paths(request.evidence_manifest_path, manifest)
    traces = load_jsonl_file(Trace, paths.trace)
    if len(traces) != 1:
        raise ValueError("performance evidence requires exactly one trace row")
    trace = traces[0]
    _require_manifest_trace_identity(manifest, trace, paths.trace)
    timing = _load_timing_evidence(paths.timing, manifest, trace)
    access_patterns = _load_access_evidence(
        paths.access,
        manifest,
        timing,
    )
    profile = load_json_file(ProfileSummarySidecar, paths.profile)
    _require_current_profile(profile, paths.trace)
    _require_trace_citation(profile, paths.trace)
    static = load_json_file(StaticKernelEvidenceSidecar, paths.static)
    compiled = _manifest_compiled_characterizations(
        manifest,
        static,
        paths.static,
    )
    calibration = _load_calibration(request.calibration_profile_path)
    inference = _load_inference_profile(
        request.inference_profile_path,
        request.calibration_profile_path,
    )
    identity_reasons = _manifest_calibration_reasons(calibration, manifest)
    dispatches, dispatch_reasons = _manifest_dispatch_evidence(
        manifest,
        request.evidence_manifest_path,
        trace,
        timing,
        compiled,
    )
    semantic = semantic_loader(
        request.solar_manifest_path,
        workload_uuid=trace.workload.uuid,
        definition=trace.definition,
    )
    return _BuildEvidence(
        manifest=manifest,
        trace=trace,
        semantic=semantic,
        timing=timing,
        access_patterns=access_patterns,
        compiled=compiled,
        dispatches=dispatches,
        calibration=calibration,
        inference=inference,
        identity_reasons=identity_reasons,
        dispatch_reasons=dispatch_reasons,
    )


def _build_artifact_paths(
    manifest_path: Path,
    manifest: PerformanceEvidenceManifest,
) -> _BuildArtifactPaths:
    def artifact(kind: PerformanceEvidenceArtifactKind) -> Path:
        return _manifest_artifact_path(manifest_path, manifest, kind)

    return _BuildArtifactPaths(
        trace=artifact(PerformanceEvidenceArtifactKind.TRACE),
        timing=artifact(PerformanceEvidenceArtifactKind.TIMING),
        access=artifact(PerformanceEvidenceArtifactKind.ACCESS_PATTERN),
        profile=artifact(PerformanceEvidenceArtifactKind.PROFILE_SUMMARY),
        static=artifact(PerformanceEvidenceArtifactKind.STATIC_EVIDENCE),
    )


def _manifest_artifact_path(
    manifest_path: Path,
    manifest: PerformanceEvidenceManifest,
    kind: PerformanceEvidenceArtifactKind,
) -> Path:
    artifact = manifest.artifact(kind)
    if artifact is None:
        raise ValueError(f"performance evidence lacks {kind}")
    return (manifest_path.parent / artifact.path).resolve()


def _require_manifest_trace_identity(
    manifest: PerformanceEvidenceManifest,
    trace: Trace,
    trace_path: Path,
) -> None:
    identity = manifest.identity
    reasons: list[str] = []
    if identity.run_id != sha256_file(trace_path):
        reasons.append("manifest_run_id_mismatch")
    if identity.definition != trace.definition:
        reasons.append("manifest_definition_mismatch")
    if identity.definition_sha256 != stable_json_checksum(trace.definition):
        reasons.append("manifest_definition_sha256_mismatch")
    if identity.workload_uuid != trace.workload.uuid:
        reasons.append("manifest_workload_uuid_mismatch")
    workload_hash = stable_json_checksum(trace.workload.model_dump(mode="json"))
    if identity.workload_sha256 != workload_hash:
        reasons.append("manifest_workload_sha256_mismatch")
    evaluation = trace.evaluation
    if evaluation is None or evaluation.performance is None:
        reasons.append("canonical_performance_missing")
    elif identity.gpu_architecture != evaluation.environment.hardware:
        reasons.append("manifest_gpu_architecture_mismatch")
    if reasons:
        raise ValueError(",".join(reasons))


def _load_timing_evidence(
    path: Path,
    manifest: PerformanceEvidenceManifest,
    trace: Trace,
) -> WorkloadTimingEvidence:
    timing = load_json_file(PerformanceTimingEvidenceSidecar, path)
    identity = manifest.identity
    if (
        timing.run_id != identity.run_id
        or timing.trace_sha256 != identity.run_id
        or timing.solution_sha256 != identity.solution_sha256
    ):
        raise ValueError("timing_evidence_identity_mismatch")
    matches = [
        item
        for item in timing.workloads
        if item.workload_uuid == trace.workload.uuid
    ]
    if len(matches) != 1:
        raise ValueError("timing_evidence_workload_mismatch")
    evaluation = trace.evaluation
    if (
        evaluation is None
        or evaluation.performance is None
        or matches[0].latency_ms != evaluation.performance.latency_ms
    ):
        raise ValueError("timing_evidence_latency_mismatch")
    return matches[0]


def _load_access_evidence(
    path: Path,
    manifest: PerformanceEvidenceManifest,
    timing: WorkloadTimingEvidence,
) -> list[AccessPatternSummary]:
    access = load_json_file(PerformanceAccessEvidenceSidecar, path)
    identity = manifest.identity
    if (
        access.run_id != identity.run_id
        or access.trace_sha256 != identity.run_id
    ):
        raise ValueError("access_evidence_identity_mismatch")
    matches = [
        item
        for item in access.workloads
        if item.workload_uuid == identity.workload_uuid
    ]
    if (
        len(matches) != 1
        or matches[0].canonical_input_sha256 != timing.input_sha256
    ):
        raise ValueError("access_evidence_workload_mismatch")
    return matches[0].patterns


def _manifest_compiled_characterizations(
    manifest: PerformanceEvidenceManifest,
    static: StaticKernelEvidenceSidecar,
    static_path: Path,
) -> list[CompiledCharacterization]:
    if not manifest.code_object_sha256:
        return []
    compiled, _ = _compiled_characterizations(
        static,
        static_path,
        manifest.identity.gpu_architecture,
    )
    observed = {
        item.code_object_sha256
        for item in compiled
        if item.code_object_sha256 is not None
    }
    if observed != set(manifest.code_object_sha256):
        raise ValueError("manifest_code_object_sha256_mismatch")
    return [
        item.model_copy(
            update={"candidate_sha256": manifest.identity.candidate_sha256}
        )
        for item in compiled
    ]


def _manifest_calibration_reasons(
    calibration: DiagnosticCalibrationProfile | None,
    manifest: PerformanceEvidenceManifest,
) -> list[str]:
    if calibration is None:
        return ["calibration_profile_missing"]
    identity = manifest.identity
    return validate_calibration_identity(
        calibration,
        gpu_architecture=identity.gpu_architecture,
        gpu_id=identity.gpu_id,
        gpu_bdf=identity.gpu_bdf,
        rocm_version=identity.rocm_version,
        compiler_version=identity.compiler_version,
        clock_mode=identity.clock_mode,
        power_profile=identity.power_profile,
    )


def _manifest_dispatch_evidence(
    manifest: PerformanceEvidenceManifest,
    manifest_path: Path,
    trace: Trace,
    timing: WorkloadTimingEvidence,
    compiled: list[CompiledCharacterization],
) -> tuple[list[DispatchEvidence], list[str]]:
    replay, replay_reasons = _verify_replay_evidence(
        manifest,
        manifest_path,
        timing,
    )
    if replay_reasons:
        return [], replay_reasons
    provenance = _verify_counter_provenance(manifest, manifest_path)
    if (
        replay is not None
        and provenance.application_executable_sha256 != "unresolved"
        and any(
            process.process_executable_sha256
            != provenance.application_executable_sha256
            for process in replay.processes
        )
    ):
        raise ValueError("replay_process_executable_sha256_mismatch")
    csv_artifacts = manifest.artifacts_of_kind(
        PerformanceEvidenceArtifactKind.COUNTER_CSV
    )
    if not csv_artifacts:
        return [], ["counter_csv_missing"]
    passes: list[CounterPassCSV] = []
    for artifact in csv_artifacts:
        path = (manifest_path.parent / artifact.path).resolve()
        pass_index = counter_pass_index(path)
        if pass_index is None:
            raise ValueError(f"counter_csv_pass_identity_missing:{path.name}")
        passes.append(CounterPassCSV(pass_index, path))
    dispatches = parse_and_align_counter_passes(
        passes,
        workload_uuid=trace.workload.uuid,
        candidate_sha256=manifest.identity.candidate_sha256,
    )
    collapsed = _collapse_replayed_dispatches(
        _record_static_runtime_conflicts(dispatches, compiled)
    )
    reasons = (
        []
        if any(dispatch.valid for dispatch in collapsed)
        else ["no_valid_dispatch_evidence"]
    )
    return collapsed, reasons


def _verify_replay_evidence(
    manifest: PerformanceEvidenceManifest,
    manifest_path: Path,
    timing: WorkloadTimingEvidence,
) -> tuple[PerformanceReplayEvidenceSidecar | None, list[str]]:
    artifact = manifest.artifact(
        PerformanceEvidenceArtifactKind.REPLAY_EVIDENCE
    )
    if artifact is None:
        return None, ["replay_evidence_missing"]
    replay = load_json_file(
        PerformanceReplayEvidenceSidecar,
        manifest_path.parent / artifact.path,
    )
    identity = manifest.identity
    if (
        replay.run_id != identity.run_id
        or replay.candidate_sha256 != identity.candidate_sha256
        or replay.canonical_input_sha256 != timing.input_sha256
    ):
        raise ValueError("replay_evidence_identity_mismatch")
    if replay.status is not DiagnosticSidecarStatus.AVAILABLE:
        return replay, (
            list(replay.reason_codes) or ["replay_evidence_unavailable"]
        )
    return replay, []


def _verify_counter_provenance(
    manifest: PerformanceEvidenceManifest,
    manifest_path: Path,
) -> Rocprofv3CounterProvenance:
    artifact = manifest.artifact(
        PerformanceEvidenceArtifactKind.COUNTER_PROVENANCE
    )
    if artifact is None:
        raise ValueError("counter_provenance_missing")
    return load_json_file(
        Rocprofv3CounterProvenance,
        manifest_path.parent / artifact.path,
    )


def _require_current_profile(
    profile: ProfileSummarySidecar,
    trace_path: Path,
) -> None:
    freshness = validate_profile_summary_freshness(
        profile,
        trace_path=str(trace_path),
    )
    governance = evaluate_profile_summary_governance(
        sidecar=profile,
        freshness=freshness,
    )
    if governance.status is not DiagnosticGovernanceStatus.USABLE_DIAGNOSTIC:
        raise ValueError(
            "profile summary is not current and usable: "
            + ",".join(governance.reason_codes),
        )


def _require_trace_citation(
    profile: ProfileSummarySidecar,
    trace_path: Path,
) -> None:
    expected = sha256_file(trace_path)
    matches = [
        citation
        for citation in profile.artifact_citations
        if citation.kind == "trace"
    ]
    if not matches or all(citation.sha256 != expected for citation in matches):
        raise ValueError("profile_summary_trace_sha256_mismatch")


def _compiled_characterizations(
    static: StaticKernelEvidenceSidecar,
    static_path: Path,
    gpu_architecture: str,
) -> tuple[list[CompiledCharacterization], str]:
    analyses = [
        analysis
        for analysis in static.isa_analyses
        if analysis.code_object_sha256 is not None
    ]
    code_hashes = sorted(
        {
            code_hash
            for analysis in analyses
            if (code_hash := analysis.code_object_sha256) is not None
        },
    )
    if not code_hashes:
        code_hashes = sorted(
            {
                artifact.sha256
                for artifact in static.artifacts
                if artifact.sha256 is not None
            },
        )
    if not code_hashes:
        raise ValueError("static evidence lacks candidate code-object hash")
    architectures = {analysis.architecture for analysis in analyses}
    if architectures and architectures != {gpu_architecture}:
        raise ValueError("static_runtime_gpu_architecture_mismatch")
    candidate_sha256 = stable_json_checksum({"code_objects": code_hashes})
    source = EvidenceReference(
        kind="static_evidence",
        path=str(static_path),
        sha256=sha256_file(static_path),
    )
    kernels = static.kernels or [
        StaticKernelEvidenceKernel(name=analysis.artifact_id)
        for analysis in analyses
    ]
    compiled: list[CompiledCharacterization] = []
    for kernel in kernels:
        analysis = _kernel_isa_analysis(kernel, kernels, analyses)
        compiled.append(
            _compiled_kernel(
                kernel,
                analysis,
                candidate_sha256,
                _kernel_code_object_sha256(kernel, analyses, code_hashes),
                gpu_architecture,
                source,
                isa_mapping_ambiguous=bool(analyses) and analysis is None,
            ),
        )
    return compiled, candidate_sha256


def _kernel_isa_analysis(
    kernel: StaticKernelEvidenceKernel,
    kernels: list[StaticKernelEvidenceKernel],
    analyses: list[StaticISAAnalysis],
) -> StaticISAAnalysis | None:
    if len(kernels) == 1 and len(analyses) == 1:
        return analyses[0]
    artifact_id = _kernel_mapping_artifact_id(kernel, analyses)
    matches = [
        analysis for analysis in analyses if analysis.artifact_id == artifact_id
    ]
    kernels_for_artifact = sum(
        _kernel_mapping_artifact_id(item, analyses) == artifact_id
        for item in kernels
    )
    if len(matches) == 1 and kernels_for_artifact == 1:
        return matches[0]
    return None


def _kernel_mapping_artifact_id(
    kernel: StaticKernelEvidenceKernel,
    analyses: list[StaticISAAnalysis],
) -> str | None:
    artifact_id = _kernel_artifact_id(kernel)
    if artifact_id is not None:
        return artifact_id
    if any(analysis.artifact_id == kernel.name for analysis in analyses):
        return kernel.name
    return None


def _kernel_artifact_id(
    kernel: StaticKernelEvidenceKernel,
) -> str | None:
    footprint = kernel.footprint
    if footprint is None or footprint.identity is None:
        return None
    return footprint.identity.artifact_id


def _kernel_code_object_sha256(
    kernel: StaticKernelEvidenceKernel,
    analyses: list[StaticISAAnalysis],
    code_hashes: list[str],
) -> str | None:
    artifact_id = _kernel_mapping_artifact_id(kernel, analyses)
    matches = [
        analysis.code_object_sha256
        for analysis in analyses
        if analysis.artifact_id == artifact_id
        and analysis.code_object_sha256 is not None
    ]
    if len(matches) == 1:
        return matches[0]
    if len(code_hashes) == 1:
        return code_hashes[0]
    return None


def _compiled_kernel(
    kernel: StaticKernelEvidenceKernel,
    analysis: object,
    candidate_sha256: str,
    code_object_sha256: str | None,
    gpu_architecture: str,
    source: EvidenceReference,
    *,
    isa_mapping_ambiguous: bool,
) -> CompiledCharacterization:
    isa = analysis if isinstance(analysis, StaticISAAnalysis) else None
    footprint = kernel.footprint
    return CompiledCharacterization(
        candidate_sha256=candidate_sha256,
        code_object_sha256=code_object_sha256,
        gpu_architecture=gpu_architecture,
        kernel_symbol=kernel.name,
        functional_group_counts=(
            isa.functional_group_counts if isa is not None else {}
        ),
        functional_subgroup_counts=(
            isa.functional_subgroup_counts if isa is not None else {}
        ),
        observed_matrix_units=(
            isa.observed_matrix_units if isa is not None else []
        ),
        footprint=ResourceFootprint(
            vgpr_count=footprint.vgpr_used if footprint else None,
            sgpr_count=footprint.sgpr_used if footprint else None,
            lds_bytes=footprint.lds_bytes if footprint else None,
            scratch_bytes=footprint.scratch_bytes if footprint else None,
        ),
        source=source,
        reason_codes=(
            ["static_isa_kernel_mapping_ambiguous"]
            if isa_mapping_ambiguous
            else []
        ),
    )


def _load_calibration(
    path: Path | None,
) -> DiagnosticCalibrationProfile | None:
    if path is None:
        return None
    profile = load_json_file(DiagnosticCalibrationProfile, path)
    audit_path = path.with_name(f"{path.stem}.audit.json")
    if not audit_path.is_file():
        raise ValueError("calibration_audit_missing")
    audit = load_json_file(DiagnosticCalibrationAudit, audit_path)
    _verify_calibration_audit(profile, audit, audit_path)
    return profile


def _load_inference_profile(
    path: Path | None,
    calibration_path: Path | None,
) -> DiagnosticInferenceProfile | None:
    if path is None:
        return None
    if calibration_path is None:
        raise ValueError("inference_profile_requires_calibration")
    profile = load_json_file(DiagnosticInferenceProfile, path)
    expected_identity = build_diagnostic_model_identity(
        PERFORMANCE_MODEL_VERSION,
    )
    if profile.model_identity != expected_identity:
        raise ValueError("inference_profile_model_identity_mismatch")
    audit_path = calibration_path.with_name(
        f"{calibration_path.stem}.audit.json"
    )
    if profile.calibration_profile_sha256 != sha256_file(calibration_path):
        raise ValueError("inference_profile_calibration_sha256_mismatch")
    if profile.calibration_audit_sha256 != sha256_file(audit_path):
        raise ValueError("inference_profile_calibration_audit_sha256_mismatch")
    return profile


def _verify_calibration_audit(
    profile: DiagnosticCalibrationProfile,
    audit: DiagnosticCalibrationAudit,
    audit_path: Path,
) -> None:
    probe = audit.probe_identity
    protocol = audit.protocol
    tuning = [item.model_dump(mode="json") for item in audit.tuning_evidence]
    estimation = [
        item.model_dump(mode="json")
        for item in audit.parameter_estimation_evidence
    ]
    estimation_hashes = {
        stable_json_checksum(estimation),
        sha256_file(audit_path),
    }
    if not estimation_hashes <= set(
        profile.parameter_estimation_evidence_sha256
    ):
        raise ValueError("calibration_parameter_estimation_sha256_mismatch")
    if stable_json_checksum(tuning) not in profile.tuning_evidence_sha256:
        raise ValueError("calibration_tuning_evidence_sha256_mismatch")
    if (
        stable_json_checksum(probe.model_dump(mode="json"))
        not in profile.probe_evidence_sha256
    ):
        raise ValueError("calibration_probe_evidence_sha256_mismatch")
    if (
        probe.architecture != profile.identity.gpu_architecture
        or probe.rocm_version != profile.identity.rocm_version
        or probe.gpu_id != profile.identity.gpu_id
        or probe.gpu_bdf != profile.identity.gpu_bdf
        or probe.compiler_version != profile.identity.compiler_version
    ):
        raise ValueError("calibration_audit_identity_mismatch")
    if not protocol.configuration_frozen_before_parameter_estimation or any(
        batch.get("phase") != "parameter_estimation_after_configuration_freeze"
        or batch.get("clocks_locked") is not True
        for batch in estimation
    ):
        raise ValueError("calibration_parameter_estimation_protocol_invalid")
    if (
        protocol.bootstrap_seed != profile.bootstrap_seed
        or protocol.bootstrap_replicates != profile.bootstrap_replicates
    ):
        raise ValueError("calibration_bootstrap_protocol_mismatch")
    if protocol.parameter_estimation_process_batches < 5:
        raise ValueError(
            "calibration_parameter_estimation_processes_insufficient"
        )


def _rocm_version(trace: Trace) -> str | None:
    if trace.evaluation is None:
        return None
    libraries = trace.evaluation.environment.libs
    return libraries.get("rocm") or libraries.get("ROCm")


def _record_static_runtime_conflicts(
    dispatches: list[DispatchEvidence],
    compiled: list[CompiledCharacterization],
) -> list[DispatchEvidence]:
    static_by_symbol = {
        key: item.footprint
        for item in compiled
        if (key := kernel_symbol_key(item.kernel_symbol)) is not None
    }
    result: list[DispatchEvidence] = []
    for dispatch in dispatches:
        runtime = dispatch.runtime_footprint
        static = static_by_symbol.get(kernel_symbol_key(dispatch.kernel_symbol))
        if static is None:
            result.append(
                dispatch.model_copy(
                    update={
                        "valid": False,
                        "reason_codes": [
                            *dispatch.reason_codes,
                            "dispatch_static_kernel_identity_mismatch",
                        ],
                    },
                ),
            )
            continue
        if runtime is None:
            result.append(dispatch)
            continue
        conflicts = [
            f"static_runtime_{field}_conflict"
            for field in (
                "vgpr_count",
                "sgpr_count",
                "lds_bytes",
                "scratch_bytes",
            )
            if getattr(runtime, field) is not None
            and getattr(static, field) is not None
            and getattr(runtime, field) != getattr(static, field)
        ]
        result.append(
            dispatch.model_copy(
                update={
                    "evidence_conflicts": [
                        *dispatch.evidence_conflicts,
                        *conflicts,
                    ],
                },
            ),
        )
    return result


def _collapse_replayed_dispatches(
    dispatches: list[DispatchEvidence],
) -> list[DispatchEvidence]:
    """Reduce profiler replay samples to one representative invocation."""
    valid = [dispatch for dispatch in dispatches if dispatch.valid]
    invalid = [dispatch for dispatch in dispatches if not dispatch.valid]
    if not valid:
        return invalid
    grouped: dict[
        tuple[
            str,
            tuple[int, int, int],
            tuple[int, int, int],
            str | None,
            str | None,
        ],
        list[DispatchEvidence],
    ] = {}
    for dispatch in valid:
        key = (
            dispatch.kernel_symbol,
            dispatch.grid,
            dispatch.workgroup,
            dispatch.queue_id,
            dispatch.stream_id,
        )
        grouped.setdefault(key, []).append(dispatch)
    repetitions = reduce(gcd, (len(group) for group in grouped.values()))
    if repetitions <= 1:
        return dispatches
    representatives = [
        _representative_dispatch(group[slot::multiplicity], slot)
        for group in grouped.values()
        for multiplicity in [len(group) // repetitions]
        for slot in range(multiplicity)
    ]
    return [*representatives, *invalid]


def _representative_dispatch(
    samples: list[DispatchEvidence],
    slot: int,
) -> DispatchEvidence:
    """Collapse replay counters while retaining duration-free topology.

    The first aligned invocation supplies only ordering and overlap relations.
    Prediction code never consumes its timestamp distances as durations; the
    median counters below remain the quantitative replay aggregate.
    """
    first = samples[0]
    footprints = {sample.runtime_footprint for sample in samples}
    if len(footprints) > 1:
        return first.model_copy(
            update={
                "valid": False,
                "reason_codes": [
                    *first.reason_codes,
                    "replayed_dispatch_footprint_mismatch",
                ],
            },
        )
    counter_names = set().union(*(sample.counters.keys() for sample in samples))
    counters = {
        name: statistics.median(
            sample.counters[name]
            for sample in samples
            if name in sample.counters
        )
        for name in sorted(counter_names)
    }
    sources = {
        (source.kind, source.path, source.sha256): source
        for sample in samples
        for source in sample.sources
    }
    return first.model_copy(
        update={
            "dispatch_id": f"representative:{first.kernel_symbol}:{slot}",
            "correlation_id": None,
            "iteration_ordinal": slot,
            "counter_passes": sorted(
                {
                    index
                    for sample in samples
                    for index in sample.counter_passes
                },
            ),
            "counters": counters,
            "sources": list(sources.values()),
        },
    )


def _counter_artifact_paths(
    profile: ProfileSummarySidecar,
    profile_path: Path,
) -> list[Path]:
    citations = [
        citation
        for citation in profile.artifact_citations
        if citation.label == "counter_csv" and citation.path is not None
    ]
    if not citations:
        return []
    trace_name = profile.identity.trace_path
    if not trace_name:
        raise ValueError("profile_summary_trace_path_missing")
    root = profile_path.parent / f"{trace_name}.rocprofv3"
    verified: list[Path] = []
    for citation in citations:
        if (
            citation.path is None
            or citation.sha256 is None
            or citation.size_bytes is None
        ):
            raise ValueError("counter_csv_citation_integrity_missing")
        verified.append(
            verify_artifact_file(
                root,
                citation.path,
                expected_sha256=citation.sha256,
                expected_size_bytes=citation.size_bytes,
            ),
        )
    if len(verified) != len(set(verified)):
        raise ValueError("counter_csv_citation_duplicate")
    return sorted(verified)


def _workload_diagnostic(
    inputs: _WorkloadDiagnosticInputs,
) -> WorkloadPerformanceDiagnostic:
    ir, hw = _workload_predictions(
        semantic=inputs.semantic,
        access_patterns=inputs.access_patterns,
        compiled=inputs.compiled,
        dispatches=inputs.dispatches,
        calibration=inputs.calibration,
        identity_reasons=inputs.identity_reasons,
        extra_reasons=inputs.extra_reasons,
    )
    hw = apply_conformal_interval(
        hw,
        inputs.semantic.workload_kind,
        point_features(inputs.semantic),
        inputs.inference,
    )
    evaluation = inputs.trace.evaluation
    if evaluation is None or evaluation.performance is None:
        raise ValueError(
            f"workload {inputs.trace.workload.uuid} lacks canonical performance timing",
        )
    measured = evaluation.performance.latency_ms
    frontier, frontier_reasons = _frontier_time(
        inputs.frontier_path,
        inputs.trace.workload.uuid,
        inputs.trace,
    )
    compiled_reasons = _compiled_reason_codes(inputs.compiled)
    ratios = calculate_ratios(
        t_pred_ir=ir,
        t_pred_hw=hw,
        t_measured_ms=measured,
        t_measured_lower_ms=inputs.timing.lower_ms,
        t_measured_upper_ms=inputs.timing.upper_ms,
        t_sol_ms=inputs.semantic.t_sol_ms,
        t_frontier_ms=frontier,
        frontier_reason_codes=frontier_reasons,
    )
    return WorkloadPerformanceDiagnostic(
        workload_uuid=inputs.trace.workload.uuid,
        semantic=inputs.semantic,
        compiled=inputs.compiled,
        dispatches=inputs.dispatches,
        schedule=build_schedule_evidence(
            inputs.dispatches,
            scope_verified=_schedule_scope_verified(inputs.extra_reasons),
        ),
        t_pred_ir=ir,
        t_pred_hw=hw,
        t_measured_ms=measured,
        t_measured_lower_ms=inputs.timing.lower_ms,
        t_measured_upper_ms=inputs.timing.upper_ms,
        t_frontier_ms=frontier,
        ratios=ratios,
        attributions=derive_attributions(
            semantic=inputs.semantic,
            compiled=inputs.compiled,
            dispatches=inputs.dispatches,
            t_pred_ir=ir,
            t_pred_hw=hw,
            ratios=ratios,
            inference_profile=inputs.inference,
        ),
        reason_codes=[
            *inputs.identity_reasons,
            *inputs.extra_reasons,
            *compiled_reasons,
            *frontier_reasons,
        ],
    )


def _schedule_scope_verified(reason_codes: list[str]) -> bool:
    identity_reasons = {
        "gpu_identity_snapshot_incomplete",
        "gpu_id_snapshot_invalid",
        "gpu_bdf_snapshot_invalid",
    }
    return not any(
        reason.startswith("replay_") or reason in identity_reasons
        for reason in reason_codes
    )


def _workload_predictions(
    *,
    semantic: SemanticCharacterization,
    access_patterns: list[AccessPatternSummary],
    compiled: list[CompiledCharacterization],
    dispatches: list[DispatchEvidence],
    calibration: DiagnosticCalibrationProfile | None,
    identity_reasons: list[str],
    extra_reasons: list[str],
) -> tuple[PerformancePrediction, PerformancePrediction]:
    if calibration is None:
        reasons = ["calibration_profile_missing"]
        return (
            _unavailable_prediction(PredictionKind.IR, reasons),
            _unavailable_prediction(PredictionKind.HW, reasons),
        )
    return (
        predict_ir(
            semantic,
            calibration,
            identity_reason_codes=identity_reasons,
            access_patterns=access_patterns,
        ),
        predict_hw(
            semantic,
            compiled,
            dispatches,
            calibration,
            identity_reason_codes=identity_reasons,
            evidence_reason_codes=extra_reasons,
            access_patterns=access_patterns,
        ),
    )


def _compiled_reason_codes(
    compiled: list[CompiledCharacterization],
) -> list[str]:
    non_blocking = {"static_isa_kernel_mapping_ambiguous"}
    return list(
        dict.fromkeys(
            reason
            for characterization in compiled
            for reason in characterization.reason_codes
            if reason not in non_blocking
        ),
    )


def _frontier_time(
    path: Path | None,
    workload_uuid: str,
    canonical: Trace,
) -> tuple[float | None, list[str]]:
    if path is None:
        return None, []
    traces = load_jsonl_file(Trace, path)
    matches = [
        trace
        for trace in traces
        if trace.workload.uuid == workload_uuid
        and trace.evaluation is not None
        and trace.evaluation.performance is not None
    ]
    if len(matches) != 1:
        raise ValueError(
            f"frontier trace does not uniquely contain {workload_uuid}",
        )
    evaluation = matches[0].evaluation
    if evaluation is None or evaluation.performance is None:
        raise ValueError(f"frontier trace lacks timing for {workload_uuid}")
    reasons = _frontier_identity_reasons(canonical, matches[0])
    if reasons:
        return None, reasons
    return evaluation.performance.latency_ms, []


def _frontier_identity_reasons(
    canonical: Trace,
    frontier: Trace,
) -> list[str]:
    if canonical.evaluation is None or frontier.evaluation is None:
        return ["frontier_evaluation_identity_missing"]
    expected = canonical.evaluation.environment
    observed = frontier.evaluation.environment
    reasons: list[str] = []
    if expected.hardware != observed.hardware:
        reasons.append("frontier_gpu_architecture_mismatch")
    reasons.extend(
        _identity_field_reasons(
            "rocm_version",
            _rocm_version(canonical),
            _rocm_version(frontier),
        ),
    )
    reasons.extend(
        _identity_field_reasons(
            "clock_state",
            expected.clocks_locked,
            observed.clocks_locked,
        ),
    )
    reasons.extend(
        _identity_field_reasons(
            "timing_protocol",
            expected.timing_protocol,
            observed.timing_protocol,
        ),
    )
    return reasons


def _identity_field_reasons(
    name: str,
    expected: object,
    observed: object,
) -> list[str]:
    if expected is None or observed is None:
        return [f"frontier_{name}_unverified"]
    if expected != observed:
        return [f"frontier_{name}_mismatch"]
    return []


def _unavailable_prediction(
    kind: PredictionKind,
    reasons: list[str],
) -> PerformancePrediction:
    return PerformancePrediction(
        kind=kind,
        status=DiagnosticSidecarStatus.UNAVAILABLE,
        reason_codes=reasons,
        limitations=["Required governed evidence is unavailable."],
    )


def _sidecar_status(
    workloads: list[WorkloadPerformanceDiagnostic],
    reasons: list[str],
) -> DiagnosticSidecarStatus:
    if not workloads:
        return DiagnosticSidecarStatus.UNAVAILABLE
    if reasons or any(
        workload.reason_codes
        or workload.t_pred_ir.status is not DiagnosticSidecarStatus.AVAILABLE
        or workload.t_pred_hw.status is not DiagnosticSidecarStatus.AVAILABLE
        for workload in workloads
    ):
        return DiagnosticSidecarStatus.PARTIAL
    return DiagnosticSidecarStatus.AVAILABLE


def _input_references(
    request: PerformanceDiagnosticBuildRequest,
) -> list[EvidenceReference]:
    paths = [
        ("performance_evidence_manifest", request.evidence_manifest_path),
        ("solar_manifest", request.solar_manifest_path),
    ]
    if request.calibration_profile_path is not None:
        paths.append(("calibration_profile", request.calibration_profile_path))
    if request.inference_profile_path is not None:
        paths.append(("inference_profile", request.inference_profile_path))
    if request.frontier_trace_path is not None:
        paths.append(("frontier_trace", request.frontier_trace_path))
    return [
        EvidenceReference(kind=kind, path=str(path), sha256=sha256_file(path))
        for kind, path in paths
    ]


def _manifest_references(
    manifest_path: Path,
    manifest: PerformanceEvidenceManifest,
) -> list[EvidenceReference]:
    return [
        EvidenceReference(
            kind=artifact.kind,
            path=str((manifest_path.parent / artifact.path).resolve()),
            sha256=artifact.sha256,
        )
        for artifact in manifest.artifacts
    ]


__all__ = [
    "PerformanceDiagnosticBuildRequest",
    "SemanticCharacterizationLoader",
    "build_performance_diagnostic",
]
