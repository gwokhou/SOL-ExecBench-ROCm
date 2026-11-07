# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Performance diagnostic evidence sidecar publication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from sol_execbench.cli.evaluation.compilation import CompilePhaseResult
from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.access_evidence import (
    WorkloadAccessEvidence,
    build_performance_access_evidence,
)
from sol_execbench.core.bench.performance_model.evidence_manifest import (
    PerformanceEvidenceArtifact,
    PerformanceEvidenceArtifactKind,
    PerformanceEvidenceManifest,
    PerformanceRunIdentity,
    artifact_reference,
    candidate_sha256,
)
from sol_execbench.core.bench.performance_model.replay_evidence import (
    build_performance_replay_evidence,
)
from sol_execbench.core.bench.performance_model.timing_evidence import (
    RAW_TIMING_FILENAME,
    PerformanceTimingEvidenceSidecar,
    RawPerformanceTimingRecord,
    build_performance_timing_evidence,
)
from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3ArtifactKind,
    Rocprofv3ProfileResult,
)
from sol_execbench.core.bench.static_kernel.evidence import (
    StaticKernelEvidenceSidecar,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
    load_jsonl_file,
)
from sol_execbench.core.data.solution import Solution
from sol_execbench.core.data.trace import Environment, Trace
from sol_execbench.core.evidence.runtime_evidence.models import (
    RuntimeGPUTelemetry,
)
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.platform.hardware import PCIeTopologyIdentity

console = Console(stderr=True)


@dataclass(frozen=True, slots=True, kw_only=True)
class PerformanceEvidenceManifestWriteRequest:
    """Inputs required to publish one performance evidence root."""

    output_file: Path | None
    traces: list[Trace]
    solution: Solution
    timing_path: Path | None
    access_path: Path | None
    profile_summary_path: Path | None
    static_evidence_path: Path | None
    profile_result: Rocprofv3ProfileResult | None
    static_evidence: StaticKernelEvidenceSidecar | None
    compile_result: CompilePhaseResult | None
    replay_path: Path | None = None


def performance_timing_sidecar_path(
    output_file: Path | None,
) -> Path | None:
    """Return the timing sidecar path for a persisted canonical trace."""
    if output_file is None:
        return None
    return output_file.with_name(f"{output_file.name}.performance-timing.json")


def write_performance_timing_sidecar(
    *,
    output_file: Path | None,
    staging_dir: Path,
    traces: list[Trace],
    solution: Solution,
) -> Path | None:
    """Publish raw canonical samples without changing Trace."""
    destination = performance_timing_sidecar_path(output_file)
    raw_path = staging_dir / RAW_TIMING_FILENAME
    if destination is None or output_file is None or not raw_path.is_file():
        return None
    try:
        solution_sha256 = stable_json_checksum(
            solution.model_dump(mode="json"),
        )
        sidecar = build_performance_timing_evidence(
            raw_path=raw_path,
            trace_path=output_file,
            traces=traces,
            solution_sha256=solution_sha256,
        )
        atomic_write_json_value(destination, sidecar.to_dict())
    except (OSError, ValueError) as error:
        console.print(
            f"[yellow]Performance timing evidence skipped: {error}[/yellow]",
        )
        return None
    console.print(
        f"[green]Saved performance timing evidence to {destination}[/green]",
    )
    return destination


def performance_access_sidecar_path(
    output_file: Path | None,
) -> Path | None:
    """Return the access-evidence path for a persisted canonical trace."""
    if output_file is None:
        return None
    return output_file.with_name(f"{output_file.name}.performance-access.json")


def write_performance_access_sidecar(
    *,
    output_file: Path | None,
    staging_dir: Path,
    traces: list[Trace],
) -> Path | None:
    """Publish de-identified trusted-input access summaries."""
    destination = performance_access_sidecar_path(output_file)
    raw_path = staging_dir / RAW_TIMING_FILENAME
    if destination is None or output_file is None or not raw_path.is_file():
        return None
    try:
        raw_records = load_jsonl_file(RawPerformanceTimingRecord, raw_path)
        expected = {
            trace.workload.uuid
            for trace in traces
            if trace.evaluation is not None
            and trace.evaluation.performance is not None
        }
        if {record.workload_uuid for record in raw_records} != expected:
            raise ValueError("raw access workload identity mismatch")
        sidecar = build_performance_access_evidence(
            trace_path=output_file,
            workloads=[
                WorkloadAccessEvidence(
                    workload_uuid=record.workload_uuid,
                    canonical_input_sha256=record.input_sha256,
                    patterns=record.access_patterns,
                )
                for record in raw_records
            ],
        )
        atomic_write_json_value(destination, sidecar.to_dict())
    except (OSError, ValueError) as error:
        console.print(
            f"[yellow]Performance access evidence skipped: {error}[/yellow]",
        )
        return None
    console.print(
        f"[green]Saved performance access evidence to {destination}[/green]",
    )
    return destination


def performance_evidence_manifest_path(
    output_file: Path | None,
) -> Path | None:
    """Return the root-manifest path for a persisted canonical trace."""
    if output_file is None:
        return None
    return output_file.with_name(
        f"{output_file.name}.performance-evidence.json"
    )


def performance_replay_sidecar_path(
    output_file: Path | None,
) -> Path | None:
    """Return the replay evidence path for one persisted canonical trace."""
    if output_file is None:
        return None
    return output_file.with_name(f"{output_file.name}.performance-replay.json")


def write_performance_replay_sidecar(
    *,
    output_file: Path | None,
    staging_dir: Path,
    traces: list[Trace],
    solution: Solution,
    timing_path: Path | None,
    profile_result: Rocprofv3ProfileResult | None,
    static_evidence: StaticKernelEvidenceSidecar | None,
    compile_result: CompilePhaseResult | None,
) -> Path | None:
    """Publish trusted-input and cross-pass replay alignment evidence."""
    destination = performance_replay_sidecar_path(output_file)
    if (
        destination is None
        or output_file is None
        or timing_path is None
        or profile_result is None
        or len(traces) != 1
    ):
        return None
    identity, _code_hashes, _reasons = _manifest_identity(
        trace_path=output_file,
        trace=traces[0],
        solution=solution,
        timing_path=timing_path,
        profile_result=profile_result,
        static_evidence=static_evidence,
        compile_result=compile_result,
    )
    try:
        sidecar = build_performance_replay_evidence(
            staging_dir=staging_dir,
            run_id=identity.run_id,
            candidate_sha256=identity.candidate_sha256,
            canonical_timing_path=timing_path,
            artifact_paths=[
                artifact.path for artifact in profile_result.artifacts
            ],
            counter_paths=[
                artifact.path
                for artifact in profile_result.artifacts
                if artifact.kind is Rocprofv3ArtifactKind.COUNTER_CSV
            ],
            expected_gpu_id=identity.gpu_id,
            expected_gpu_bdf=identity.gpu_bdf,
            expected_pcie_topology=identity.pcie_topology,
            environment=list(profile_result.environment_snapshots),
        )
        atomic_write_json_value(destination, sidecar.to_dict())
    except (OSError, ValueError) as error:
        console.print(
            f"[yellow]Performance replay evidence skipped: {error}[/yellow]"
        )
        return None
    return destination


def write_performance_evidence_manifest(
    request: PerformanceEvidenceManifestWriteRequest,
) -> Path | None:
    """Publish a single-workload counter-evidence root manifest."""
    destination = performance_evidence_manifest_path(request.output_file)
    if (
        destination is None
        or request.output_file is None
        or request.profile_result is None
        or "counter_definition_sha256" not in request.profile_result.provenance
    ):
        return None
    try:
        manifest = _build_performance_evidence_manifest(request)
        atomic_write_json_value(destination, manifest.to_dict())
    except (OSError, ValueError) as error:
        console.print(
            f"[yellow]Performance evidence manifest skipped: {error}[/yellow]",
        )
        return None
    console.print(
        f"[green]Saved performance evidence manifest to {destination}[/green]",
    )
    return destination


def _build_performance_evidence_manifest(
    request: PerformanceEvidenceManifestWriteRequest,
) -> PerformanceEvidenceManifest:
    if request.output_file is None or request.profile_result is None:
        raise ValueError("performance manifest request is incomplete")
    if len(request.traces) != 1:
        raise ValueError("counter evidence requires exactly one workload")
    trace = request.traces[0]
    artifacts, reasons = _manifest_artifacts(
        root=request.output_file.parent,
        trace_path=request.output_file,
        timing_path=request.timing_path,
        access_path=request.access_path,
        profile_summary_path=request.profile_summary_path,
        static_evidence_path=request.static_evidence_path,
        profile_result=request.profile_result,
        replay_path=request.replay_path,
    )
    identity, code_hashes, identity_reasons = _manifest_identity(
        trace_path=request.output_file,
        trace=trace,
        solution=request.solution,
        timing_path=request.timing_path,
        profile_result=request.profile_result,
        static_evidence=request.static_evidence,
        compile_result=request.compile_result,
    )
    reasons.extend(identity_reasons)
    reasons = list(dict.fromkeys(reasons))
    return PerformanceEvidenceManifest(
        status=(
            DiagnosticSidecarStatus.PARTIAL
            if reasons
            else DiagnosticSidecarStatus.AVAILABLE
        ),
        identity=identity,
        artifacts=artifacts,
        code_object_sha256=code_hashes,
        reason_codes=reasons,
    )


def _manifest_identity(
    *,
    trace_path: Path,
    trace: Trace,
    solution: Solution,
    timing_path: Path | None,
    profile_result: Rocprofv3ProfileResult,
    static_evidence: StaticKernelEvidenceSidecar | None,
    compile_result: CompilePhaseResult | None,
) -> tuple[PerformanceRunIdentity, list[str], list[str]]:
    reasons: list[str] = []
    code_hashes = _code_object_hashes(static_evidence)
    if not code_hashes:
        reasons.append("inspectable_code_object_missing")
    solution_hash = stable_json_checksum(solution.model_dump(mode="json"))
    command_hash, compiler_hash, compiler_version = _compiler_identity(
        compile_result
    )
    if command_hash is None or compiler_hash is None:
        reasons.append("candidate_compile_identity_missing")
    candidate_hash = candidate_sha256(
        solution_sha256=solution_hash,
        compile_command_sha256=command_hash,
        compiler_sha256=compiler_hash,
        code_object_sha256=code_hashes,
    )
    gpu_id, gpu_bdf, pcie_topology, gpu_reasons = _gpu_identity(
        profile_result.environment_snapshots,
    )
    reasons.extend(gpu_reasons)
    timing = _timing_identity(timing_path, trace_path)
    reasons.extend(timing[1])
    environment = trace.evaluation.environment if trace.evaluation else None
    (
        gpu_architecture,
        rocm_version,
        clock_mode,
        power_profile,
        timing_protocol,
    ) = _trace_environment_identity(environment)
    if rocm_version is None:
        reasons.append("rocm_version_missing")
    if compiler_version is None:
        reasons.append("compiler_version_missing")
    reasons.extend(_environment_reasons(environment, profile_result))
    identity = PerformanceRunIdentity(
        run_id=sha256_file(trace_path),
        definition=trace.definition,
        definition_sha256=stable_json_checksum(trace.definition),
        workload_uuid=trace.workload.uuid,
        workload_sha256=stable_json_checksum(
            trace.workload.model_dump(mode="json")
        ),
        solution_sha256=solution_hash,
        candidate_sha256=candidate_hash,
        gpu_architecture=gpu_architecture,
        gpu_id=gpu_id,
        gpu_bdf=gpu_bdf,
        pcie_topology=pcie_topology,
        rocm_version=rocm_version,
        compiler_version=compiler_version,
        clock_mode=clock_mode,
        power_profile=power_profile,
        timing_protocol=timing_protocol,
    )
    return identity, code_hashes, reasons


def _trace_environment_identity(
    environment: Environment | None,
) -> tuple[str, str | None, str, str | None, str]:
    if environment is None:
        return "unavailable", None, "unlocked", None, "unverified"
    locked = environment.clocks_locked is True
    return (
        environment.hardware,
        environment.libs.get("rocm") or environment.libs.get("ROCm"),
        "locked" if locked else "unlocked",
        "stable_peak" if locked else None,
        environment.timing_protocol or "unverified",
    )


def _environment_reasons(
    environment: Environment | None,
    profile_result: Rocprofv3ProfileResult,
) -> list[str]:
    reasons: list[str] = []
    hardware = environment.hardware if environment else None
    clocks_locked = environment.clocks_locked if environment else None
    timing_protocol = environment.timing_protocol if environment else None
    if hardware != "gfx1200":
        reasons.append("unsupported_gpu_architecture")
    if clocks_locked is not True:
        reasons.append("clock_identity_unverified")
    if not timing_protocol:
        reasons.append("timing_protocol_unverified")
    if not profile_result.succeeded:
        reasons.append("counter_collection_incomplete")
    if any(
        value == "unresolved" for value in profile_result.provenance.values()
    ):
        reasons.append("counter_provenance_incomplete")
    return reasons


def _manifest_artifacts(
    *,
    root: Path,
    trace_path: Path,
    timing_path: Path | None,
    access_path: Path | None,
    profile_summary_path: Path | None,
    static_evidence_path: Path | None,
    profile_result: Rocprofv3ProfileResult,
    replay_path: Path | None = None,
) -> tuple[list[PerformanceEvidenceArtifact], list[str]]:
    paths = (
        (PerformanceEvidenceArtifactKind.TRACE, trace_path),
        (PerformanceEvidenceArtifactKind.TIMING, timing_path),
        (PerformanceEvidenceArtifactKind.ACCESS_PATTERN, access_path),
        (
            PerformanceEvidenceArtifactKind.PROFILE_SUMMARY,
            profile_summary_path,
        ),
        (
            PerformanceEvidenceArtifactKind.STATIC_EVIDENCE,
            static_evidence_path,
        ),
        (PerformanceEvidenceArtifactKind.REPLAY_EVIDENCE, replay_path),
    )
    artifacts = [
        artifact_reference(kind=kind, path=path, root=root)
        for kind, path in paths
        if path is not None and path.is_file()
    ]
    reasons = [
        f"{kind}_missing"
        for kind, path in paths
        if path is None or not path.is_file()
    ]
    for artifact in profile_result.artifacts:
        kind = _profile_artifact_kind(artifact.kind, artifact.path)
        if kind is not None:
            artifacts.append(
                artifact_reference(kind=kind, path=artifact.path, root=root)
            )
    kinds = {artifact.kind for artifact in artifacts}
    for required in (
        PerformanceEvidenceArtifactKind.COUNTER_PROVENANCE,
        PerformanceEvidenceArtifactKind.COUNTER_CSV,
        PerformanceEvidenceArtifactKind.ROCPD,
    ):
        if required not in kinds:
            reasons.append(f"{required}_missing")
    return artifacts, reasons


def _profile_artifact_kind(
    kind: Rocprofv3ArtifactKind,
    path: Path,
) -> PerformanceEvidenceArtifactKind | None:
    if kind is Rocprofv3ArtifactKind.COUNTER_CSV:
        return PerformanceEvidenceArtifactKind.COUNTER_CSV
    if kind is Rocprofv3ArtifactKind.ROCPD:
        return PerformanceEvidenceArtifactKind.ROCPD
    if path.name.endswith(".counter-metadata.json"):
        return PerformanceEvidenceArtifactKind.COUNTER_PROVENANCE
    return None


def _code_object_hashes(
    static: StaticKernelEvidenceSidecar | None,
) -> list[str]:
    if static is None:
        return []
    inspectable = {
        artifact.sha256
        for artifact in static.artifacts
        if artifact.inspectable
        and artifact.sha256 is not None
        and artifact.target_architecture == "gfx1200"
    }
    hashes = {
        analysis.code_object_sha256
        for analysis in static.isa_analyses
        if analysis.code_object_sha256 is not None
        and analysis.code_object_sha256 in inspectable
    }
    return sorted(hashes)


def _compiler_identity(
    result: CompilePhaseResult | None,
) -> tuple[str | None, str | None, str | None]:
    if result is None or not result.succeeded or not result.command:
        return None, None, None
    command_hash = stable_json_checksum(list(result.command))
    return (
        command_hash,
        result.compiler_sha256,
        result.compiler_version,
    )


def _gpu_identity(
    snapshots: tuple[RuntimeGPUTelemetry, ...],
) -> tuple[
    str | None,
    str | None,
    PCIeTopologyIdentity | None,
    list[str],
]:
    if {snapshot.phase for snapshot in snapshots} != {"pre", "post"}:
        return None, None, None, ["gpu_identity_snapshot_incomplete"]
    gpu_ids = {snapshot.gpu_id for snapshot in snapshots}
    gpu_bdfs = {snapshot.gpu_bdf for snapshot in snapshots}
    if None in gpu_ids or len(gpu_ids) != 1:
        return None, None, None, ["gpu_id_snapshot_invalid"]
    if None in gpu_bdfs or len(gpu_bdfs) != 1:
        return None, None, None, ["gpu_bdf_snapshot_invalid"]
    topologies = [snapshot.pcie_topology for snapshot in snapshots]
    if any(topology is None for topology in topologies):
        return (
            next(iter(gpu_ids)),
            next(iter(gpu_bdfs)),
            None,
            ["pcie_topology_snapshot_incomplete"],
        )
    topology = topologies[0]
    if any(candidate != topology for candidate in topologies[1:]):
        return (
            next(iter(gpu_ids)),
            next(iter(gpu_bdfs)),
            None,
            ["pcie_topology_snapshot_changed"],
        )
    return next(iter(gpu_ids)), next(iter(gpu_bdfs)), topology, []


def _timing_identity(
    timing_path: Path | None,
    trace_path: Path,
) -> tuple[PerformanceTimingEvidenceSidecar | None, list[str]]:
    if timing_path is None or not timing_path.is_file():
        return None, ["timing_evidence_missing"]
    timing = load_json_file(PerformanceTimingEvidenceSidecar, timing_path)
    if timing.trace_sha256 != sha256_file(trace_path):
        return None, ["timing_trace_sha256_mismatch"]
    return timing, []


__all__ = [
    "performance_access_sidecar_path",
    "performance_evidence_manifest_path",
    "performance_replay_sidecar_path",
    "performance_timing_sidecar_path",
    "write_performance_access_sidecar",
    "write_performance_evidence_manifest",
    "write_performance_replay_sidecar",
    "write_performance_timing_sidecar",
]
