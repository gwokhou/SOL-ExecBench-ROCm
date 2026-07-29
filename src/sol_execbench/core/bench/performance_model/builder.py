# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Artifact-governed construction of performance diagnostic sidecars."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from functools import reduce
from math import gcd
from pathlib import Path

from sol_execbench.core.bench.diagnostic_sidecar import (
    DiagnosticGovernanceStatus,
    DiagnosticSidecarStatus,
)
from sol_execbench.core.bench.performance_model.attribution import (
    calculate_ratios,
    derive_attributions,
)
from sol_execbench.core.bench.performance_model.models import (
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
from sol_execbench.core.bench.profile_summary import (
    ProfileSummarySidecar,
    evaluate_profile_summary_governance,
    validate_profile_summary_freshness,
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
    load_json_value,
    load_jsonl_file,
)
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.integrity import (
    sha256_file,
    stable_json_checksum,
    verify_artifact_file,
)
from sol_execbench.core.solar_bridge.performance import (
    load_semantic_characterization,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PerformanceDiagnosticBuildRequest:
    """Immutable paths and identity inputs for one diagnostic build."""

    trace_path: Path
    solar_analysis_paths: dict[str, Path]
    profile_summary_path: Path
    static_evidence_path: Path
    output_path: Path
    frontier_trace_paths: dict[str, Path]
    calibration_profile_path: Path | None = None
    gpu_id: str | None = None
    compiler_version: str | None = None
    power_profile: str | None = None


def build_performance_diagnostic(
    request: PerformanceDiagnosticBuildRequest,
) -> PerformanceDiagnosticSidecar:
    """Validate all input identities and build one diagnostic-only sidecar."""
    traces = load_jsonl_file(Trace, request.trace_path)
    if not traces:
        raise ValueError("canonical trace contains no rows")
    by_workload = _traces_by_workload(traces)
    _require_exact_workload_mapping(by_workload, request.solar_analysis_paths)
    profile = load_json_file(
        ProfileSummarySidecar, request.profile_summary_path
    )
    _require_current_profile(profile, request.trace_path)
    _require_trace_citation(profile, request.trace_path)
    static = load_json_file(
        StaticKernelEvidenceSidecar,
        request.static_evidence_path,
    )
    gpu_architecture = _gpu_architecture(traces)
    compiled, candidate_sha256 = _compiled_characterizations(
        static,
        request.static_evidence_path,
        gpu_architecture,
    )
    calibration = _load_calibration(request.calibration_profile_path)
    identity_reasons = _calibration_reasons(
        calibration,
        request=request,
        traces=traces,
        gpu_architecture=gpu_architecture,
    )
    dispatches, dispatch_reasons = _dispatch_evidence(
        profile,
        request.profile_summary_path,
        by_workload,
        candidate_sha256,
        compiled,
    )
    workloads = [
        _workload_diagnostic(
            trace=by_workload[workload_uuid],
            solar_path=request.solar_analysis_paths[workload_uuid],
            frontier_path=request.frontier_trace_paths.get(workload_uuid),
            compiled=compiled,
            dispatches=dispatches.get(workload_uuid, []),
            calibration=calibration,
            identity_reasons=identity_reasons,
            extra_reasons=dispatch_reasons,
        )
        for workload_uuid in sorted(by_workload)
    ]
    reasons = [
        *identity_reasons,
        *dispatch_reasons,
        *(reason for workload in workloads for reason in workload.reason_codes),
    ]
    reasons = list(dict.fromkeys(reasons))
    status = _sidecar_status(workloads, reasons)
    return PerformanceDiagnosticSidecar(
        status=status,
        run_id=profile.identity.run_id or sha256_file(request.trace_path),
        candidate_sha256=candidate_sha256,
        gpu_architecture=gpu_architecture,
        calibration_identity=calibration.identity if calibration else None,
        workloads=workloads,
        evidence=_input_references(request),
        reason_codes=reasons,
        limitations=[
            "Performance predictions are diagnostic-only and do not affect SOL Score.",
            "Canonical Trace JSONL remains timing and correctness authority.",
            "Profiler duration and achieved rates are excluded from T_pred(HW).",
            "Repeated profiler dispatches are reduced by median counters to one invocation.",
        ],
    )


def _traces_by_workload(traces: list[Trace]) -> dict[str, Trace]:
    result: dict[str, Trace] = {}
    for trace in traces:
        workload_uuid = trace.workload.uuid
        if workload_uuid in result:
            raise ValueError(
                f"duplicate workload UUID in trace: {workload_uuid}"
            )
        if trace.evaluation is None or trace.evaluation.performance is None:
            raise ValueError(
                f"workload {workload_uuid} lacks canonical performance timing",
            )
        result[workload_uuid] = trace
    return result


def _require_exact_workload_mapping(
    traces: dict[str, Trace],
    solar_paths: dict[str, Path],
) -> None:
    missing = sorted(traces.keys() - solar_paths.keys())
    extra = sorted(solar_paths.keys() - traces.keys())
    if missing or extra:
        raise ValueError(
            f"solar workload mapping mismatch: missing={missing}, extra={extra}",
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


def _gpu_architecture(traces: list[Trace]) -> str:
    architectures = {
        trace.evaluation.environment.hardware
        for trace in traces
        if trace.evaluation is not None
    }
    if len(architectures) != 1:
        raise ValueError("canonical trace GPU architecture mismatch")
    return next(iter(architectures))


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
    audit = load_json_value(audit_path)
    if not isinstance(audit, dict):
        raise ValueError("calibration_audit_invalid")
    _verify_calibration_audit(profile, audit, audit_path)
    return profile


def _verify_calibration_audit(
    profile: DiagnosticCalibrationProfile,
    audit: dict[str, object],
    audit_path: Path,
) -> None:
    probe = audit.get("probe_identity")
    protocol = audit.get("protocol")
    held_out = audit.get("held_out_evidence")
    if not isinstance(probe, dict) or not isinstance(protocol, dict):
        raise ValueError("calibration_audit_invalid")
    if not isinstance(held_out, list) or not held_out:
        raise ValueError("calibration_held_out_evidence_missing")
    expected_hashes = {
        stable_json_checksum(held_out),
        sha256_file(audit_path),
    }
    if not expected_hashes <= set(profile.held_out_evidence_sha256):
        raise ValueError("calibration_held_out_evidence_sha256_mismatch")
    if stable_json_checksum(probe) not in profile.probe_evidence_sha256:
        raise ValueError("calibration_probe_evidence_sha256_mismatch")
    if (
        probe.get("architecture") != profile.identity.gpu_architecture
        or probe.get("rocm_version") != profile.identity.rocm_version
        or probe.get("gpu_id") != profile.identity.gpu_id
        or probe.get("compiler_version") != profile.identity.compiler_version
    ):
        raise ValueError("calibration_audit_identity_mismatch")
    if protocol.get(
        "configuration_frozen_before_held_out_measurement"
    ) is not True or any(
        not isinstance(batch, dict)
        or batch.get("phase") != "held_out_after_configuration_freeze"
        or batch.get("clocks_locked") is not True
        for batch in held_out
    ):
        raise ValueError("calibration_held_out_protocol_invalid")


def _calibration_reasons(
    calibration: DiagnosticCalibrationProfile | None,
    *,
    request: PerformanceDiagnosticBuildRequest,
    traces: list[Trace],
    gpu_architecture: str,
) -> list[str]:
    if calibration is None:
        return ["calibration_profile_missing"]
    rocm_versions = {
        _rocm_version(trace)
        for trace in traces
        if _rocm_version(trace) is not None
    }
    if len(rocm_versions) > 1:
        return ["trace_rocm_version_mismatch"]
    rocm_version = next(iter(rocm_versions), None)
    clocks_locked = {
        trace.evaluation.environment.clocks_locked
        for trace in traces
        if trace.evaluation is not None
    }
    clock_mode = "locked" if clocks_locked == {True} else "unlocked"
    return validate_calibration_identity(
        calibration,
        gpu_architecture=gpu_architecture,
        gpu_id=request.gpu_id,
        rocm_version=rocm_version,
        compiler_version=request.compiler_version,
        clock_mode=clock_mode,
        power_profile=request.power_profile,
    )


def _rocm_version(trace: Trace) -> str | None:
    if trace.evaluation is None:
        return None
    libraries = trace.evaluation.environment.libs
    return libraries.get("rocm") or libraries.get("ROCm")


def _dispatch_evidence(
    profile: ProfileSummarySidecar,
    profile_path: Path,
    traces: dict[str, Trace],
    candidate_sha256: str,
    compiled: list[CompiledCharacterization],
) -> tuple[dict[str, list[DispatchEvidence]], list[str]]:
    if len(traces) != 1:
        return {}, ["counter_workload_alignment_unavailable"]
    workload_uuid = next(iter(traces))
    counter_paths = _counter_artifact_paths(profile, profile_path)
    if not counter_paths:
        return {}, ["counter_csv_missing"]
    passes = []
    for path in counter_paths:
        pass_index = counter_pass_index(path)
        if pass_index is None:
            raise ValueError(f"counter_csv_pass_identity_missing:{path.name}")
        passes.append(CounterPassCSV(pass_index, path))
    dispatches = parse_and_align_counter_passes(
        passes,
        workload_uuid=workload_uuid,
        candidate_sha256=candidate_sha256,
    )
    return {
        workload_uuid: _collapse_replayed_dispatches(
            _record_static_runtime_conflicts(
                dispatches,
                compiled,
            ),
        ),
    }, []


def _record_static_runtime_conflicts(
    dispatches: list[DispatchEvidence],
    compiled: list[CompiledCharacterization],
) -> list[DispatchEvidence]:
    static_by_symbol = {item.kernel_symbol: item.footprint for item in compiled}
    result: list[DispatchEvidence] = []
    for dispatch in dispatches:
        runtime = dispatch.runtime_footprint
        static = static_by_symbol.get(dispatch.kernel_symbol)
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
        tuple[str, tuple[int, int, int], tuple[int, int, int]],
        list[DispatchEvidence],
    ] = {}
    for dispatch in valid:
        key = (dispatch.kernel_symbol, dispatch.grid, dispatch.workgroup)
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
            "start_timestamp_ns": None,
            "end_timestamp_ns": None,
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
    *,
    trace: Trace,
    solar_path: Path,
    frontier_path: Path | None,
    compiled: list[CompiledCharacterization],
    dispatches: list[DispatchEvidence],
    calibration: DiagnosticCalibrationProfile | None,
    identity_reasons: list[str],
    extra_reasons: list[str],
) -> WorkloadPerformanceDiagnostic:
    semantic = load_semantic_characterization(
        solar_path,
        workload_uuid=trace.workload.uuid,
    )
    ir, hw = _workload_predictions(
        semantic=semantic,
        compiled=compiled,
        dispatches=dispatches,
        calibration=calibration,
        identity_reasons=identity_reasons,
        extra_reasons=extra_reasons,
    )
    evaluation = trace.evaluation
    if evaluation is None or evaluation.performance is None:
        raise ValueError(
            f"workload {trace.workload.uuid} lacks canonical performance timing",
        )
    measured = evaluation.performance.latency_ms
    frontier, frontier_reasons = _frontier_time(
        frontier_path,
        trace.workload.uuid,
        trace,
    )
    compiled_reasons = _compiled_reason_codes(compiled)
    ratios = calculate_ratios(
        t_pred_ir=ir,
        t_pred_hw=hw,
        t_measured_ms=measured,
        timing_noise_ms=max(measured * 0.02, 0.0001),
        t_sol_ms=semantic.t_sol_ms,
        t_frontier_ms=frontier,
        frontier_reason_codes=frontier_reasons,
    )
    return WorkloadPerformanceDiagnostic(
        workload_uuid=trace.workload.uuid,
        semantic=semantic,
        compiled=compiled,
        dispatches=dispatches,
        t_pred_ir=ir,
        t_pred_hw=hw,
        t_measured_ms=measured,
        t_frontier_ms=frontier,
        ratios=ratios,
        attributions=derive_attributions(
            semantic=semantic,
            compiled=compiled,
            dispatches=dispatches,
            t_pred_ir=ir,
            t_pred_hw=hw,
            ratios=ratios,
        ),
        reason_codes=[
            *identity_reasons,
            *extra_reasons,
            *compiled_reasons,
            *frontier_reasons,
        ],
    )


def _workload_predictions(
    *,
    semantic: SemanticCharacterization,
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
        ),
        predict_hw(
            compiled,
            dispatches,
            calibration,
            identity_reason_codes=[*identity_reasons, *extra_reasons],
        ),
    )


def _compiled_reason_codes(
    compiled: list[CompiledCharacterization],
) -> list[str]:
    return list(
        dict.fromkeys(
            reason
            for characterization in compiled
            for reason in characterization.reason_codes
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
        ("trace", request.trace_path),
        ("profile_summary", request.profile_summary_path),
        ("static_evidence", request.static_evidence_path),
        *(
            ("solar_analysis", path)
            for path in request.solar_analysis_paths.values()
        ),
        *(
            ("frontier_trace", path)
            for path in request.frontier_trace_paths.values()
        ),
    ]
    if request.calibration_profile_path is not None:
        paths.append(("calibration_profile", request.calibration_profile_path))
    return [
        EvidenceReference(kind=kind, path=str(path), sha256=sha256_file(path))
        for kind, path in paths
    ]


__all__ = [
    "PerformanceDiagnosticBuildRequest",
    "build_performance_diagnostic",
]
