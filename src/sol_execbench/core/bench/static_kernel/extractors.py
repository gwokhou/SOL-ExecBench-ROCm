# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Static kernel evidence extractor routing and execution."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

from sol_execbench.core.bench.static_kernel.evidence_builders import (
    build_static_kernel_evidence_sidecar,
)
from sol_execbench.core.bench.static_kernel.amdgpu_metadata import (
    extract_amdgpu_footprints,
)
from sol_execbench.core.bench.static_kernel.footprint_parsers import (
    parse_roc_objdump_resource_usage,
)
from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticKernelEvidenceArtifact,
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
    run_static_extractor as _run_static_extractor,
)
from sol_execbench.core.bench.static_kernel.extractor_routing import (
    aggregate_extractor_reason as _aggregate_extractor_reason,
    aggregate_extractor_status as _aggregate_extractor_status,
    artifact_persisted_path as _artifact_persisted_path,
    classification_from_tool_runs as _classification_from_tool_runs,
    extractor_command as _extractor_command,
    reason_for_route_status as _reason_for_route_status,
    route_static_tool as _route_static_tool,
    tool_run_from_route_decision as _tool_run_from_route_decision,
    toolchain_artifact_type_for_static_artifact,
)
from sol_execbench.core.platform.toolchain import (
    ProbeRunner,
    ToolchainCapability,
    ToolchainStatus,
    Which,
)


_STATIC_EXTRACTOR_TOOL_IDS = ("llvm-objdump", "readelf")
_FOOTPRINT_EXTRACTOR_TOOL_IDS = ("roc-objdump",)


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
) -> StaticKernelEvidenceSidecar:
    """Run routed bounded static extractors for persisted artifacts."""

    evidence_root = evidence_directory.resolve()
    sidecar_base = (
        sidecar_base_directory.resolve()
        if sidecar_base_directory is not None
        else evidence_root
    )
    effective_registry = list(registry) if registry is not None else None
    tool_runs: list[StaticKernelEvidenceToolRun] = []
    warnings: list[StaticKernelEvidenceWarning] = []
    output_artifacts: list[StaticKernelEvidenceArtifact] = []

    for artifact in artifacts:
        artifact_type = toolchain_artifact_type_for_static_artifact(artifact)
        if artifact_type is None:
            tool_runs.append(
                StaticKernelEvidenceToolRun(
                    tool_id="static-extractor",
                    command=[],
                    status=StaticKernelEvidenceStatus.UNSUPPORTED,
                    reason_code=(
                        StaticKernelEvidenceReasonCode.UNSUPPORTED_ARTIFACT_TYPE
                    ),
                    stderr_tail=(
                        f"{artifact.artifact_type} is not supported by "
                        "llvm-objdump/readelf extraction."
                    ),
                    timeout_seconds=timeout_seconds,
                )
            )
            continue

        artifact_path = _artifact_persisted_path(artifact, sidecar_base)
        if artifact_path is None or not artifact_path.is_file():
            tool_runs.append(
                StaticKernelEvidenceToolRun(
                    tool_id="static-extractor",
                    command=[],
                    status=StaticKernelEvidenceStatus.UNAVAILABLE,
                    reason_code=StaticKernelEvidenceReasonCode.ARTIFACT_UNAVAILABLE,
                    stderr_tail=f"{artifact.artifact_id} persisted artifact is missing.",
                    timeout_seconds=timeout_seconds,
                )
            )
            continue

        for tool_id in _STATIC_EXTRACTOR_TOOL_IDS:
            route_decision = _route_static_tool(
                tool_id=tool_id,
                artifact_type=artifact_type,
                registry=effective_registry,
                runner=probe_runner,
                which=which,
                timeout_seconds=timeout_seconds,
            )
            if route_decision is None:
                tool_runs.append(
                    _tool_run_from_route_decision(
                        tool_id=tool_id,
                        command=[],
                        decision_status=ToolchainStatus.UNAVAILABLE,
                        reason_code=StaticKernelEvidenceReasonCode.TOOLCHAIN_UNAVAILABLE,
                        reason="No route decision was produced.",
                        timeout_seconds=timeout_seconds,
                    )
                )
                continue
            command = _extractor_command(tool_id, artifact_path)
            if route_decision.status != ToolchainStatus.AVAILABLE:
                tool_runs.append(
                    _tool_run_from_route_decision(
                        tool_id=tool_id,
                        command=command,
                        decision_status=route_decision.status,
                        reason_code=_reason_for_route_status(route_decision.status),
                        reason=route_decision.reason,
                        timeout_seconds=timeout_seconds,
                    )
                )
                warnings.append(
                    StaticKernelEvidenceWarning(
                        code=f"{tool_id}_not_executed",
                        message=route_decision.reason,
                        source_reference=StaticKernelEvidenceSourceReference(
                            kind="toolchain_route",
                            value=tool_id,
                            description=route_decision.reason_code,
                        ),
                    )
                )
                continue

            tool_run, raw_artifact = _run_static_extractor(
                tool_id=tool_id,
                command=command,
                artifact=artifact,
                evidence_root=evidence_root,
                sidecar_base=sidecar_base,
                timeout_seconds=timeout_seconds,
                runner=runner,
            )
            tool_runs.append(tool_run)
            if raw_artifact is not None:
                output_artifacts.append(raw_artifact)

    footprints = _collect_resource_footprints(
        artifacts=artifacts,
        sidecar_base=sidecar_base,
        evidence_root=evidence_root,
        probe_runner=probe_runner,
        which=which,
        registry=effective_registry,
        runner=runner,
        timeout_seconds=timeout_seconds,
        output_artifacts=output_artifacts,
    )
    all_artifacts = list(artifacts) + output_artifacts
    return build_static_kernel_evidence_sidecar(
        status=_aggregate_extractor_status(tool_runs),
        reason_code=_aggregate_extractor_reason(tool_runs),
        artifacts=all_artifacts,
        tool_runs=tool_runs,
        footprints=footprints,
        warnings=warnings,
        classification=_classification_from_tool_runs(tool_runs, artifacts),
    )


def _collect_resource_footprints(
    *,
    artifacts: Sequence[StaticKernelEvidenceArtifact],
    sidecar_base: Path,
    evidence_root: Path,
    probe_runner: ProbeRunner | None,
    which: Which,
    registry: Sequence[ToolchainCapability] | None,
    runner: ExtractorRunner | None,
    timeout_seconds: float,
    output_artifacts: list[StaticKernelEvidenceArtifact],
) -> list[StaticResourceFootprint]:
    """Run routed footprint extractors (``roc-objdump``) as an enhancement.

    Footprint extractors are optional enhancements over the base static
    evidence: when a tool is unavailable it is skipped silently rather than
    downgrading the aggregate static evidence status. Successful runs append
    their raw output artifact for diagnostics and contribute a parsed
    ``StaticResourceFootprint``.
    """

    footprints: list[StaticResourceFootprint] = []
    for artifact in artifacts:
        artifact_type = toolchain_artifact_type_for_static_artifact(artifact)
        if artifact_type is None:
            continue
        artifact_path = _artifact_persisted_path(artifact, sidecar_base)
        if artifact_path is None or not artifact_path.is_file():
            continue
        covered_footprint = False
        for tool_id in _FOOTPRINT_EXTRACTOR_TOOL_IDS:
            route_decision = _route_static_tool(
                tool_id=tool_id,
                artifact_type=artifact_type,
                registry=registry,
                runner=probe_runner,
                which=which,
                timeout_seconds=timeout_seconds,
            )
            if (
                route_decision is None
                or route_decision.status != ToolchainStatus.AVAILABLE
            ):
                continue
            command = _extractor_command(tool_id, artifact_path)
            tool_run, raw_artifact = _run_static_extractor(
                tool_id=tool_id,
                command=command,
                artifact=artifact,
                evidence_root=evidence_root,
                sidecar_base=sidecar_base,
                timeout_seconds=timeout_seconds,
                runner=runner,
            )
            if raw_artifact is not None:
                output_artifacts.append(raw_artifact)
            if (
                tool_run.status == StaticKernelEvidenceStatus.COLLECTED
                and raw_artifact is not None
            ):
                raw_path = _artifact_persisted_path(raw_artifact, sidecar_base)
                if raw_path is not None and raw_path.is_file():
                    text = raw_path.read_text(encoding="utf-8", errors="replace")
                    footprint = parse_roc_objdump_resource_usage(
                        text,
                        artifact_id=artifact.artifact_id,
                        source_sha256=raw_artifact.sha256,
                    )
                    if footprint is not None:
                        footprints.append(footprint)
                        covered_footprint = True
        if not covered_footprint:
            # ROCm 7.x fallback: read AMDGPU code-object metadata directly
            # (roc-objdump is absent). Covers CDNA + RDNA (amdgcn ABI).
            try:
                footprints.extend(
                    extract_amdgpu_footprints(
                        artifact_path.read_bytes(),
                        artifact_id=artifact.artifact_id,
                    )
                )
            except OSError:
                pass
    return footprints
