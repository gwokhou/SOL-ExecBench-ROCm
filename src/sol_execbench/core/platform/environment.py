# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Optional ROCm runtime environment snapshot evidence facade."""

from __future__ import annotations

from .environment_diagnostics import (
    build_environment_diagnostics,
    run_pytorch_smoke_checks,
)
from .environment_models import (
    DEFAULT_PROBE_TIMEOUT_SECONDS,
    ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION,
    EnvironmentCheckResult,
    EnvironmentDiagnostics,
    EnvironmentEvidenceStatus,
    EnvironmentSnapshot,
    GpuEnvironmentSummary,
    ProbeCompletedProcess,
    ProbeRunner,
    PytorchRocmSummary,
    RocmEnvironmentSummary,
    ToolProbeResult,
    Which,
)
from .environment_probes import (
    collect_pytorch_rocm_summary,
    probe_tool,
)
from .environment_snapshot import collect_environment_snapshot

__all__ = [
    "DEFAULT_PROBE_TIMEOUT_SECONDS",
    "ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION",
    "EnvironmentCheckResult",
    "EnvironmentDiagnostics",
    "EnvironmentEvidenceStatus",
    "EnvironmentSnapshot",
    "GpuEnvironmentSummary",
    "ProbeCompletedProcess",
    "ProbeRunner",
    "PytorchRocmSummary",
    "RocmEnvironmentSummary",
    "ToolProbeResult",
    "Which",
    "build_environment_diagnostics",
    "collect_environment_snapshot",
    "collect_pytorch_rocm_summary",
    "probe_tool",
    "run_pytorch_smoke_checks",
]
