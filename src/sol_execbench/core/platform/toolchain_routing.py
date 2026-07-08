# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Toolchain routing decisions."""

from __future__ import annotations

import fnmatch
import shutil
from collections.abc import Callable
from datetime import UTC, datetime

from .toolchain_models import (
    DEFAULT_TOOLCHAIN_PROBE_TIMEOUT_SECONDS,
    ProbeRunner,
    ToolLifecycle,
    ToolchainCapability,
    ToolchainRoutingDecision,
    ToolchainRoutingReport,
    ToolchainRoutingRequest,
    ToolchainStatus,
    Which,
)
from .toolchain_probes import probe_toolchain_tool
from .toolchain_registry import default_toolchain_registry


def build_toolchain_routing_report(
    request: ToolchainRoutingRequest,
    *,
    registry: list[ToolchainCapability] | None = None,
    runner: ProbeRunner | None = None,
    which: Which = shutil.which,
    timeout_seconds: float = DEFAULT_TOOLCHAIN_PROBE_TIMEOUT_SECONDS,
    now: Callable[[], datetime] | None = None,
) -> ToolchainRoutingReport:
    """Build a diagnostic routing report for a requested evidence path."""

    generated_at = (now or (lambda: datetime.now(UTC)))().isoformat()
    decisions: list[ToolchainRoutingDecision] = []
    effective_registry = registry or default_toolchain_registry()
    for capability in effective_registry:
        decision = decision_for_capability(
            request,
            capability,
            runner=runner,
            which=which,
            timeout_seconds=timeout_seconds,
        )
        if decision is not None:
            decisions.append(decision)

    selected_tool_id = None
    for index, decision in enumerate(decisions):
        if decision.status == ToolchainStatus.AVAILABLE:
            selected_tool_id = decision.tool_id
            decisions[index] = decision.model_copy(update={"selected": True})
            break

    if selected_tool_id is None and not decisions:
        decisions.append(
            ToolchainRoutingDecision(
                tool_id="none",
                lifecycle=ToolLifecycle.REJECTED,
                status=ToolchainStatus.UNSUPPORTED_ARTIFACT,
                reason_code="no_registry_entry",
                reason=(
                    "No registry entry supports the requested evidence level "
                    f"{request.evidence_level.value} and artifact "
                    f"{request.artifact_type.value}."
                ),
            )
        )

    return ToolchainRoutingReport(
        generated_at=generated_at,
        request=request,
        selected_tool_id=selected_tool_id,
        decisions=decisions,
    )


def decision_for_capability(
    request: ToolchainRoutingRequest,
    capability: ToolchainCapability,
    *,
    runner: ProbeRunner | None,
    which: Which,
    timeout_seconds: float,
) -> ToolchainRoutingDecision | None:
    if request.evidence_level not in capability.evidence_levels:
        return None
    if request.artifact_type not in capability.artifact_types:
        return ToolchainRoutingDecision(
            tool_id=capability.tool_id,
            lifecycle=capability.lifecycle,
            status=ToolchainStatus.UNSUPPORTED_ARTIFACT,
            reason_code="unsupported_artifact",
            reason=(
                f"{capability.tool_id} does not support artifact type "
                f"{request.artifact_type.value} for {request.evidence_level.value}."
            ),
            source_refs=capability.source_refs,
        )
    if request.gpu_architecture and not supports_arch(
        capability, request.gpu_architecture
    ):
        return ToolchainRoutingDecision(
            tool_id=capability.tool_id,
            lifecycle=capability.lifecycle,
            status=ToolchainStatus.UNSUPPORTED_ARCH,
            reason_code="unsupported_arch",
            reason=(
                f"{capability.tool_id} does not declare support for "
                f"{request.gpu_architecture}."
            ),
            source_refs=capability.source_refs,
        )
    if capability.lifecycle == ToolLifecycle.DEPRECATED:
        return lifecycle_decision(capability, ToolchainStatus.DEPRECATED)
    if capability.lifecycle == ToolLifecycle.MIGRATED:
        return lifecycle_decision(capability, ToolchainStatus.MIGRATED)
    if capability.lifecycle == ToolLifecycle.PLANNED:
        return lifecycle_decision(capability, ToolchainStatus.PLANNED)
    if capability.lifecycle == ToolLifecycle.REJECTED:
        return lifecycle_decision(capability, ToolchainStatus.REJECTED)
    if not capability.expected_binaries:
        return ToolchainRoutingDecision(
            tool_id=capability.tool_id,
            lifecycle=capability.lifecycle,
            status=ToolchainStatus.UNAVAILABLE,
            reason_code="repository_reference_only",
            reason=f"{capability.tool_id} is a source reference, not a runnable tool.",
            source_refs=capability.source_refs,
        )

    binary = capability.expected_binaries[0]
    command = capability.probe_command or [binary, "--version"]
    probe = probe_toolchain_tool(
        capability.tool_id,
        binary,
        command,
        runner=runner,
        which=which,
        timeout_seconds=timeout_seconds,
    )
    return ToolchainRoutingDecision(
        tool_id=capability.tool_id,
        lifecycle=capability.lifecycle,
        status=probe.status,
        reason_code=f"probe_{probe.status.value}",
        reason=(
            f"{capability.tool_id} probe {probe.status.value}."
            if probe.status != ToolchainStatus.UNAVAILABLE
            else f"{binary} is not available on PATH."
        ),
        source_refs=capability.source_refs,
        probe=probe,
    )


def supports_arch(capability: ToolchainCapability, gpu_architecture: str) -> bool:
    if not capability.gpu_arch_patterns:
        return True
    return any(
        fnmatch.fnmatchcase(gpu_architecture, pattern)
        for pattern in capability.gpu_arch_patterns
    )


def lifecycle_decision(
    capability: ToolchainCapability,
    status: ToolchainStatus,
) -> ToolchainRoutingDecision:
    reason_code = f"tool_{status.value}"
    replacement = (
        f" Use {capability.replacement_tool_id} instead."
        if capability.replacement_tool_id
        else ""
    )
    return ToolchainRoutingDecision(
        tool_id=capability.tool_id,
        lifecycle=capability.lifecycle,
        status=status,
        reason_code=reason_code,
        reason=f"{capability.tool_id} lifecycle is {status.value}.{replacement}",
        fallback_tool_id=capability.replacement_tool_id,
        source_refs=capability.source_refs,
    )
