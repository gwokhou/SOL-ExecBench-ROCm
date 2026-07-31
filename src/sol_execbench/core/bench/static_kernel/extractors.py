# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Static kernel evidence extractor routing and execution."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sol_execbench.core.bench.static_kernel.amdgpu_metadata import (
    extract_amdgpu_footprints,
    extract_amdgpu_kernels,
)
from sol_execbench.core.bench.static_kernel.evidence_builders import (
    build_static_kernel_evidence_sidecar,
)
from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceKernel,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceSidecar,
    StaticKernelEvidenceSourceReference,
    StaticKernelEvidenceStatus,
    StaticKernelEvidenceToolRun,
    StaticKernelEvidenceWarning,
    StaticResourceFootprint,
)
from sol_execbench.core.bench.static_kernel.extractor_execution import (
    ExtractorRunner,
    run_static_extractor,
)
from sol_execbench.core.bench.static_kernel.extractor_routing import (
    aggregate_extractor_reason,
    aggregate_extractor_status,
    artifact_persisted_path,
    classification_from_tool_runs,
    extractor_command,
    reason_for_route_status,
    route_static_tool,
    static_extractor_tool_ids,
    tool_run_from_route_decision,
    toolchain_artifact_type_for_static_artifact,
)
from sol_execbench.core.bench.static_kernel.footprint_parsers import (
    parse_roc_objdump_resource_usage,
)
from sol_execbench.core.bench.static_kernel.isa_analysis import (
    collect_static_isa_analyses,
)
from sol_execbench.core.platform.toolchain import (
    ProbeRunner,
    ToolchainArtifactType,
    ToolchainCapability,
    ToolchainStatus,
    Which,
)
from sol_execbench.core.process.subprocesses import (
    ProbeCompletedProcess,
    run_bounded_probe,
)

_FOOTPRINT_EXTRACTOR_TOOL_IDS = ("roc-objdump",)


@dataclass(frozen=True, slots=True)
class _ExtractorContext:
    """Shared paths, tool routing, and execution settings for one collection."""

    evidence_root: Path
    sidecar_base: Path
    timeout_seconds: float
    runner: ExtractorRunner | None
    probe_runner: ProbeRunner
    which: Which
    registry: Sequence[ToolchainCapability] | None


def _memoize_which(which: Which) -> Which:
    """Cache ``which(binary)`` lookups; tool paths are invariant across a run."""
    cache: dict[str, str | None] = {}

    def resolved(binary: str) -> str | None:
        if binary not in cache:
            cache[binary] = which(binary)
        return cache[binary]

    return resolved


def _memoize_probe_runner(runner: ProbeRunner | None) -> ProbeRunner:
    """Cache probe results per command.

    A tool's version probe (``<binary> --version``) is invariant across
    artifacts in one run, so memoizing it avoids N redundant subprocess spawns
    when the extractor loop routes the same tool per artifact.
    """
    cache: dict[tuple[str, ...], ProbeCompletedProcess] = {}

    def resolved(
        command: list[str],
        timeout_seconds: float,
    ) -> ProbeCompletedProcess:
        key = tuple(command)
        if key not in cache:
            cache[key] = (runner or run_bounded_probe)(command, timeout_seconds)
        return cache[key]

    return resolved


def run_static_kernel_extractors(
    *,
    artifacts: Sequence[StaticKernelEvidenceArtifact],
    evidence_directory: Path,
    sidecar_base_directory: Path | None = None,
    timeout_seconds: float = 10.0,
    runner: ExtractorRunner | None = None,
    probe_runner: ProbeRunner | None = None,
    which: Which = shutil.which,
    registry: Sequence[ToolchainCapability] | None = None,
    analyze_isa: bool = False,
) -> StaticKernelEvidenceSidecar:
    """Run routed bounded static extractors for persisted artifacts."""
    evidence_root = evidence_directory.resolve()
    sidecar_base = (
        sidecar_base_directory.resolve()
        if sidecar_base_directory is not None
        else evidence_root
    )
    context = _ExtractorContext(
        evidence_root=evidence_root,
        sidecar_base=sidecar_base,
        timeout_seconds=timeout_seconds,
        runner=runner,
        probe_runner=_memoize_probe_runner(probe_runner),
        which=_memoize_which(which),
        registry=list(registry) if registry is not None else None,
    )
    tool_runs, warnings, output_artifacts = _run_routed_extractors(
        artifacts,
        context,
    )

    footprints, kernels, amdgpu_runs = _collect_resource_footprints(
        artifacts=artifacts,
        context=context,
        output_artifacts=output_artifacts,
    )
    tool_runs.extend(amdgpu_runs)
    isa_analyses = []
    if analyze_isa:
        isa_analyses, isa_runs, isa_artifacts = collect_static_isa_analyses(
            artifacts=artifacts,
            evidence_root=evidence_root,
            sidecar_base=sidecar_base,
            timeout_seconds=max(timeout_seconds, 30.0),
        )
        tool_runs.extend(isa_runs)
        output_artifacts.extend(isa_artifacts)
    all_artifacts = list(artifacts) + output_artifacts
    return build_static_kernel_evidence_sidecar(
        status=aggregate_extractor_status(tool_runs),
        reason_code=aggregate_extractor_reason(tool_runs),
        artifacts=all_artifacts,
        tool_runs=tool_runs,
        kernels=kernels,
        footprints=footprints,
        isa_analyses=isa_analyses,
        warnings=warnings,
        classification=classification_from_tool_runs(tool_runs, artifacts),
    )


def _run_routed_extractors(
    artifacts: Sequence[StaticKernelEvidenceArtifact],
    context: _ExtractorContext,
) -> tuple[
    list[StaticKernelEvidenceToolRun],
    list[StaticKernelEvidenceWarning],
    list[StaticKernelEvidenceArtifact],
]:
    """Run standard disassembly extractors for every valid persisted artifact."""
    tool_runs: list[StaticKernelEvidenceToolRun] = []
    warnings: list[StaticKernelEvidenceWarning] = []
    output_artifacts: list[StaticKernelEvidenceArtifact] = []
    for artifact in artifacts:
        runs, artifact_warnings, produced = _run_extractors_for_artifact(
            artifact,
            context,
        )
        tool_runs.extend(runs)
        warnings.extend(artifact_warnings)
        output_artifacts.extend(produced)
    return tool_runs, warnings, output_artifacts


def _run_extractors_for_artifact(
    artifact: StaticKernelEvidenceArtifact,
    context: _ExtractorContext,
) -> tuple[
    list[StaticKernelEvidenceToolRun],
    list[StaticKernelEvidenceWarning],
    list[StaticKernelEvidenceArtifact],
]:
    artifact_type = toolchain_artifact_type_for_static_artifact(artifact)
    if artifact_type is None:
        return [_unsupported_artifact_run(artifact, context)], [], []
    artifact_path = artifact_persisted_path(artifact, context.sidecar_base)
    if artifact_path is None or not artifact_path.is_file():
        return [_missing_artifact_run(artifact, context)], [], []

    tool_runs: list[StaticKernelEvidenceToolRun] = []
    warnings: list[StaticKernelEvidenceWarning] = []
    output_artifacts: list[StaticKernelEvidenceArtifact] = []
    for tool_id in static_extractor_tool_ids():
        tool_run, warning, raw_artifact = _run_routed_tool(
            tool_id,
            artifact,
            artifact_type,
            artifact_path,
            context,
        )
        tool_runs.append(tool_run)
        if warning is not None:
            warnings.append(warning)
        if raw_artifact is not None:
            output_artifacts.append(raw_artifact)
    return tool_runs, warnings, output_artifacts


def _unsupported_artifact_run(
    artifact: StaticKernelEvidenceArtifact,
    context: _ExtractorContext,
) -> StaticKernelEvidenceToolRun:
    return StaticKernelEvidenceToolRun(
        tool_id="static-extractor",
        command=[],
        status=StaticKernelEvidenceStatus.UNSUPPORTED,
        reason_code=StaticKernelEvidenceReasonCode.UNSUPPORTED_ARTIFACT_TYPE,
        stderr_tail=(
            f"{artifact.artifact_type} is not supported by llvm-objdump/readelf extraction."
        ),
        timeout_seconds=context.timeout_seconds,
    )


def _missing_artifact_run(
    artifact: StaticKernelEvidenceArtifact,
    context: _ExtractorContext,
) -> StaticKernelEvidenceToolRun:
    return StaticKernelEvidenceToolRun(
        tool_id="static-extractor",
        command=[],
        status=StaticKernelEvidenceStatus.UNAVAILABLE,
        reason_code=StaticKernelEvidenceReasonCode.ARTIFACT_UNAVAILABLE,
        stderr_tail=f"{artifact.artifact_id} persisted artifact is missing.",
        timeout_seconds=context.timeout_seconds,
    )


def _run_routed_tool(
    tool_id: str,
    artifact: StaticKernelEvidenceArtifact,
    artifact_type: ToolchainArtifactType,
    artifact_path: Path,
    context: _ExtractorContext,
) -> tuple[
    StaticKernelEvidenceToolRun,
    StaticKernelEvidenceWarning | None,
    StaticKernelEvidenceArtifact | None,
]:
    route_decision = route_static_tool(
        tool_id=tool_id,
        artifact_type=artifact_type,
        registry=context.registry,
        runner=context.probe_runner,
        which=context.which,
        timeout_seconds=context.timeout_seconds,
    )
    if route_decision is None:
        return (
            tool_run_from_route_decision(
                tool_id=tool_id,
                command=[],
                decision_status=ToolchainStatus.UNAVAILABLE,
                reason_code=StaticKernelEvidenceReasonCode.TOOLCHAIN_UNAVAILABLE,
                reason="No route decision was produced.",
                timeout_seconds=context.timeout_seconds,
            ),
            None,
            None,
        )
    command = extractor_command(tool_id, artifact_path)
    if route_decision.status == ToolchainStatus.AVAILABLE:
        tool_run, raw_artifact = run_static_extractor(
            tool_id=tool_id,
            command=command,
            artifact=artifact,
            evidence_root=context.evidence_root,
            sidecar_base=context.sidecar_base,
            timeout_seconds=context.timeout_seconds,
            runner=context.runner,
        )
        return tool_run, None, raw_artifact
    warning = StaticKernelEvidenceWarning(
        code=f"{tool_id}_not_executed",
        message=route_decision.reason,
        source_reference=StaticKernelEvidenceSourceReference(
            kind="toolchain_route",
            value=tool_id,
            description=route_decision.reason_code,
        ),
    )
    return (
        tool_run_from_route_decision(
            tool_id=tool_id,
            command=command,
            decision_status=route_decision.status,
            reason_code=reason_for_route_status(route_decision.status),
            reason=route_decision.reason,
            timeout_seconds=context.timeout_seconds,
        ),
        warning,
        None,
    )


def _collect_resource_footprints(
    *,
    artifacts: Sequence[StaticKernelEvidenceArtifact],
    context: _ExtractorContext,
    output_artifacts: list[StaticKernelEvidenceArtifact],
) -> tuple[
    list[StaticResourceFootprint],
    list[StaticKernelEvidenceKernel],
    list[StaticKernelEvidenceToolRun],
]:
    """Run footprint extractors (``roc-objdump`` then native AMDGPU metadata).

    ``roc-objdump`` is the routed extractor where available; on ROCm 7.x
    (roc-objdump removed) the native AMDGPU metadata parser runs as a fallback.
    Returns the parsed footprints plus an ``amdgpu-metadata`` tool-run per
    artifact that went through the native fallback, so the sidecar records the
    real footprint source.
    """
    footprints: list[StaticResourceFootprint] = []
    kernels: dict[str, StaticKernelEvidenceKernel] = {}
    amdgpu_runs: list[StaticKernelEvidenceToolRun] = []
    for artifact in artifacts:
        artifact_type = toolchain_artifact_type_for_static_artifact(artifact)
        if artifact_type is None:
            continue
        artifact_path = artifact_persisted_path(artifact, context.sidecar_base)
        if artifact_path is None or not artifact_path.is_file():
            continue
        routed_footprints, raw_artifacts = _collect_routed_footprints(
            artifact,
            artifact_type,
            artifact_path,
            context,
        )
        footprints.extend(routed_footprints)
        output_artifacts.extend(raw_artifacts)
        native_footprints, native_kernels, native_run = (
            _collect_native_amdgpu_footprints(
                artifact,
                artifact_path,
                context,
            )
        )
        if not routed_footprints:
            footprints.extend(native_footprints)
        for kernel in native_kernels:
            kernels.setdefault(kernel.name, kernel)
        if native_run is not None and not routed_footprints:
            amdgpu_runs.append(native_run)
    return footprints, list(kernels.values()), amdgpu_runs


def _collect_routed_footprints(
    artifact: StaticKernelEvidenceArtifact,
    artifact_type: ToolchainArtifactType,
    artifact_path: Path,
    context: _ExtractorContext,
) -> tuple[list[StaticResourceFootprint], list[StaticKernelEvidenceArtifact]]:
    """Collect resource footprints from routed text extractors when available."""
    footprints: list[StaticResourceFootprint] = []
    raw_artifacts: list[StaticKernelEvidenceArtifact] = []
    for tool_id in _FOOTPRINT_EXTRACTOR_TOOL_IDS:
        route_decision = route_static_tool(
            tool_id=tool_id,
            artifact_type=artifact_type,
            registry=context.registry,
            runner=context.probe_runner,
            which=context.which,
            timeout_seconds=context.timeout_seconds,
        )
        if (
            route_decision is None
            or route_decision.status != ToolchainStatus.AVAILABLE
        ):
            continue
        tool_run, raw_artifact = run_static_extractor(
            tool_id=tool_id,
            command=extractor_command(tool_id, artifact_path),
            artifact=artifact,
            evidence_root=context.evidence_root,
            sidecar_base=context.sidecar_base,
            timeout_seconds=context.timeout_seconds,
            runner=context.runner,
        )
        if raw_artifact is not None:
            raw_artifacts.append(raw_artifact)
        footprint = _footprint_from_collected_output(
            artifact,
            tool_run,
            raw_artifact,
            context.sidecar_base,
        )
        if footprint is not None:
            footprints.append(footprint)
    return footprints, raw_artifacts


def _footprint_from_collected_output(
    artifact: StaticKernelEvidenceArtifact,
    tool_run: StaticKernelEvidenceToolRun,
    raw_artifact: StaticKernelEvidenceArtifact | None,
    sidecar_base: Path,
) -> StaticResourceFootprint | None:
    if (
        tool_run.status != StaticKernelEvidenceStatus.COLLECTED
        or raw_artifact is None
    ):
        return None
    raw_path = artifact_persisted_path(raw_artifact, sidecar_base)
    if raw_path is None or not raw_path.is_file():
        return None
    return parse_roc_objdump_resource_usage(
        raw_path.read_text(encoding="utf-8", errors="replace"),
        artifact_id=artifact.artifact_id,
        source_sha256=raw_artifact.sha256,
    )


def _collect_native_amdgpu_footprints(
    artifact: StaticKernelEvidenceArtifact,
    artifact_path: Path,
    context: _ExtractorContext,
) -> tuple[
    list[StaticResourceFootprint],
    list[StaticKernelEvidenceKernel],
    StaticKernelEvidenceToolRun | None,
]:
    """Read AMDGPU code-object metadata when no routed footprint is available."""
    try:
        data = artifact_path.read_bytes()
        footprints = extract_amdgpu_footprints(
            data,
            artifact_id=artifact.artifact_id,
            source_sha256=artifact.sha256,
            target_architecture=artifact.target_architecture,
        )
        kernels = extract_amdgpu_kernels(
            data,
            artifact_id=artifact.artifact_id,
            source_sha256=artifact.sha256,
            target_architecture=artifact.target_architecture,
        )
    except OSError:
        return [], [], None
    if not footprints:
        return [], kernels, None
    tool_run = StaticKernelEvidenceToolRun(
        tool_id="amdgpu-metadata",
        command=["amdgpu-metadata", str(artifact_path)],
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        stdout_tail=f"native AMDGPU metadata: {len(footprints)} footprint(s)",
        timeout_seconds=context.timeout_seconds,
    )
    return footprints, kernels, tool_run
