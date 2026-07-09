# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ROCm toolchain capability registry and routing helpers facade."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from datetime import datetime

from .models import (
    DEFAULT_TOOLCHAIN_PROBE_TIMEOUT_SECONDS,
    TOOLCHAIN_ROUTING_SCHEMA_VERSION,
    ProbeRunner,
    ToolLifecycle,
    ToolchainArtifactType,
    ToolchainCapability,
    ToolchainEvidenceLevel,
    ToolchainProbeResult,
    ToolchainRoutingDecision,
    ToolchainRoutingReport,
    ToolchainRoutingRequest,
    ToolchainStatus,
    Which,
)
from .probes import probe_toolchain_tool, run_probe as _run_probe
from .registry import default_toolchain_registry
from .routing import (
    build_toolchain_routing_report as _build_toolchain_routing_report,
)


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

    return _build_toolchain_routing_report(
        request,
        registry=registry,
        runner=runner or _run_probe,
        which=which,
        timeout_seconds=timeout_seconds,
        now=now,
    )


__all__ = [
    "DEFAULT_TOOLCHAIN_PROBE_TIMEOUT_SECONDS",
    "TOOLCHAIN_ROUTING_SCHEMA_VERSION",
    "ProbeRunner",
    "ToolLifecycle",
    "ToolchainArtifactType",
    "ToolchainCapability",
    "ToolchainEvidenceLevel",
    "ToolchainProbeResult",
    "ToolchainRoutingDecision",
    "ToolchainRoutingReport",
    "ToolchainRoutingRequest",
    "ToolchainStatus",
    "Which",
    "build_toolchain_routing_report",
    "default_toolchain_registry",
    "probe_toolchain_tool",
]
