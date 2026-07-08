# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Routing and status helpers for static kernel extractors."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from sol_execbench.core.bench.static_kernel_evidence_models import (
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceClassification,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceStatus,
    StaticKernelEvidenceToolRun,
)
from sol_execbench.core.bench.static_kernel_status import (
    aggregate_extractor_reason_value,
    aggregate_extractor_status_value,
)
from sol_execbench.core.toolchain import (
    ProbeRunner,
    ToolchainArtifactType,
    ToolchainCapability,
    ToolchainEvidenceLevel,
    ToolchainRoutingDecision,
    ToolchainRoutingRequest,
    ToolchainStatus,
    Which,
    build_toolchain_routing_report,
    default_toolchain_registry,
)


def route_static_tool(
    *,
    tool_id: str,
    artifact_type: ToolchainArtifactType,
    registry: Sequence[ToolchainCapability] | None,
    runner: ProbeRunner | None,
    which: Which,
    timeout_seconds: float,
) -> ToolchainRoutingDecision | None:
    effective_registry = (
        list(registry) if registry is not None else default_toolchain_registry()
    )
    tool_registry = [entry for entry in effective_registry if entry.tool_id == tool_id]
    report = build_toolchain_routing_report(
        ToolchainRoutingRequest(
            evidence_level=ToolchainEvidenceLevel.STATIC,
            artifact_type=artifact_type,
        ),
        registry=tool_registry,
        runner=runner,
        which=which,
        timeout_seconds=timeout_seconds,
    )
    for decision in report.decisions:
        if decision.tool_id == tool_id:
            return decision
    return None


def tool_run_from_route_decision(
    *,
    tool_id: str,
    command: list[str],
    decision_status: ToolchainStatus,
    reason_code: StaticKernelEvidenceReasonCode,
    reason: str,
    timeout_seconds: float,
) -> StaticKernelEvidenceToolRun:
    return StaticKernelEvidenceToolRun(
        tool_id=tool_id,
        command=command,
        status=status_for_route_status(decision_status),
        reason_code=reason_code,
        stderr_tail=reason,
        timeout_seconds=timeout_seconds,
    )


def extractor_command(tool_id: str, artifact_path: Path) -> list[str]:
    if tool_id == "llvm-objdump":
        return [tool_id, "--disassemble", str(artifact_path)]
    if tool_id == "readelf":
        return [tool_id, "--headers", "--wide", str(artifact_path)]
    raise ValueError(f"unsupported static extractor: {tool_id}")


def toolchain_artifact_type_for_static_artifact(
    artifact: StaticKernelEvidenceArtifact,
) -> ToolchainArtifactType | None:
    if artifact.artifact_type in {"shared_library", "hsaco", "code_object"}:
        return ToolchainArtifactType.ROCM_BINARY
    if artifact.artifact_type == "object_file":
        return ToolchainArtifactType.ELF_OBJECT
    return None


def artifact_persisted_path(
    artifact: StaticKernelEvidenceArtifact,
    sidecar_base: Path,
) -> Path | None:
    if artifact.persisted_path is None:
        return None
    path = Path(artifact.persisted_path)
    if path.is_absolute():
        return path
    return sidecar_base / path


def status_for_route_status(status: ToolchainStatus) -> StaticKernelEvidenceStatus:
    if status == ToolchainStatus.UNSUPPORTED_ARTIFACT:
        return StaticKernelEvidenceStatus.UNSUPPORTED
    if status == ToolchainStatus.UNSUPPORTED_ARCH:
        return StaticKernelEvidenceStatus.UNSUPPORTED
    if status == ToolchainStatus.FAILED:
        return StaticKernelEvidenceStatus.FAILED
    return StaticKernelEvidenceStatus.UNAVAILABLE


def reason_for_route_status(
    status: ToolchainStatus,
) -> StaticKernelEvidenceReasonCode:
    if status == ToolchainStatus.UNSUPPORTED_ARTIFACT:
        return StaticKernelEvidenceReasonCode.UNSUPPORTED_ARTIFACT_TYPE
    if status == ToolchainStatus.UNSUPPORTED_ARCH:
        return StaticKernelEvidenceReasonCode.UNSUPPORTED_ARCHITECTURE
    if status == ToolchainStatus.FAILED:
        return StaticKernelEvidenceReasonCode.EXTRACTOR_FAILED
    return StaticKernelEvidenceReasonCode.TOOLCHAIN_UNAVAILABLE


def aggregate_extractor_status(
    tool_runs: Sequence[StaticKernelEvidenceToolRun],
) -> StaticKernelEvidenceStatus:
    return aggregate_extractor_status_value(
        tuple(tool_runs),
        collected=StaticKernelEvidenceStatus.COLLECTED,
        partial=StaticKernelEvidenceStatus.PARTIAL,
        failed=StaticKernelEvidenceStatus.FAILED,
        unavailable=StaticKernelEvidenceStatus.UNAVAILABLE,
    )


def aggregate_extractor_reason(
    tool_runs: Sequence[StaticKernelEvidenceToolRun],
) -> StaticKernelEvidenceReasonCode:
    status = aggregate_extractor_status(tool_runs)
    return aggregate_extractor_reason_value(
        tuple(tool_runs),
        status=status,
        collected_status=StaticKernelEvidenceStatus.COLLECTED,
        partial_status=StaticKernelEvidenceStatus.PARTIAL,
        failed_status=StaticKernelEvidenceStatus.FAILED,
        collected_reason=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        partial_reason=StaticKernelEvidenceReasonCode.PARTIAL_ARTIFACT_METADATA,
        partial_disassembly_reason=StaticKernelEvidenceReasonCode.PARTIAL_DISASSEMBLY_ONLY,
        failed_reason=StaticKernelEvidenceReasonCode.EXTRACTOR_FAILED,
        timeout_reason=StaticKernelEvidenceReasonCode.EXTRACTOR_TIMEOUT,
        unavailable_reason=StaticKernelEvidenceReasonCode.TOOLCHAIN_UNAVAILABLE,
    )


def classification_from_tool_runs(
    tool_runs: Sequence[StaticKernelEvidenceToolRun],
    artifacts: Sequence[StaticKernelEvidenceArtifact],
) -> StaticKernelEvidenceClassification:
    metadata_present = any(
        run.tool_id == "readelf" and run.status == StaticKernelEvidenceStatus.COLLECTED
        for run in tool_runs
    )
    disassembly_present = any(
        run.tool_id == "llvm-objdump"
        and run.status == StaticKernelEvidenceStatus.COLLECTED
        and bool(run.stdout_tail.strip())
        for run in tool_runs
    )
    detected_architectures = sorted(
        {
            architecture
            for artifact in artifacts
            for architecture in artifact.classification.detected_architectures
        }
    )
    return StaticKernelEvidenceClassification(
        metadata_present=metadata_present,
        disassembly_present=disassembly_present,
        detected_architectures=detected_architectures,
    )
