# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ROCm toolchain capability registry and routing helpers facade."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from datetime import datetime

from sol_execbench.core.platform.runtime import Which
from sol_execbench.core.platform.toolchain.models import (
    DEFAULT_TOOLCHAIN_PROBE_TIMEOUT_SECONDS,
    TOOLCHAIN_ROUTING_SCHEMA_VERSION,
    ToolchainArtifactType,
    ToolchainCapability,
    ToolchainEvidenceLevel,
    ToolchainProbeResult,
    ToolchainRoutingDecision,
    ToolchainRoutingReport,
    ToolchainRoutingRequest,
    ToolchainStatus,
    ToolLifecycle,
)
from sol_execbench.core.platform.toolchain.probes import probe_toolchain_tool
from sol_execbench.core.platform.toolchain.registry import (
    default_toolchain_registry,
)
from sol_execbench.core.platform.toolchain.routing import (
    build_toolchain_routing_report as _build_toolchain_routing_report,
)
from sol_execbench.core.process.subprocesses import (
    ProbeRunner,
    run_bounded_probe as _run_probe,
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
    "ToolLifecycle",
    "ToolchainArtifactType",
    "ToolchainCapability",
    "ToolchainEvidenceLevel",
    "ToolchainProbeResult",
    "ToolchainRoutingDecision",
    "ToolchainRoutingReport",
    "ToolchainRoutingRequest",
    "ToolchainStatus",
    "build_toolchain_routing_report",
    "default_toolchain_registry",
    "probe_toolchain_tool",
]
