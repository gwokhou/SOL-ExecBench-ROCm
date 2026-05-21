# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal ROCm diagnostics and stage-aware failure helpers.

These helpers intentionally do not define a new CLI or trace schema. They are a
small internal surface for making ROCm readiness and failure messages more
consistent while preserving existing public contracts.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class DiagnosticStage(str, Enum):
    """High-level evaluation stages used in actionable failure messages."""

    PARSE = "parse"
    PACKAGE = "package"
    COMPILE = "compile"
    RUNTIME = "runtime"
    VERIFY = "verify"
    TIMING = "timing"
    ENVIRONMENT = "environment"


@dataclass(frozen=True)
class StageDiagnostic:
    """Actionable diagnostic information for one stage or tool."""

    stage: DiagnosticStage
    status: str
    message: str
    hint: str | None = None

    def format(self) -> str:
        """Return a compact human-readable diagnostic message."""
        prefix = f"{self.stage.value}: {self.status}: {self.message}"
        if self.hint:
            return f"{prefix}\n  Fix: {self.hint}"
        return prefix


class SolExecBenchError(RuntimeError):
    """Stage-aware internal error with an optional remediation hint."""

    def __init__(
        self,
        stage: DiagnosticStage,
        message: str,
        *,
        hint: str | None = None,
    ) -> None:
        super().__init__(StageDiagnostic(stage, "failed", message, hint).format())
        self.stage = stage
        self.hint = hint


class ProfilerBackend(str, Enum):
    """Internal profiling-readiness backend classification."""

    ROCPROFV3 = "rocprofv3"
    ROCPROFILER_COMPUTE = "rocprofiler-compute"
    OMNIPERF = "omniperf"
    SKIP = "skip"


@dataclass(frozen=True)
class ProfilerReadiness:
    """Selected profiling route and explanation for the current ROCm environment."""

    backend: ProfilerBackend
    reason: str
    fallback_applied: bool
    effective_level: str


def detect_tool(path: str, which: Callable[[str], str | None] = shutil.which) -> bool:
    """Return whether *path* is available on ``PATH``."""
    return which(path) is not None


def classify_gfx(gfx: str | None) -> str:
    """Classify an AMD gfx target into a broad architecture family."""
    if not gfx:
        return "unknown"
    if gfx.startswith("gfx94"):
        return "cdna3"
    if gfx.startswith("gfx12"):
        return "rdna4"
    if gfx.startswith("gfx11"):
        return "rdna3"
    return "unknown"


def select_profiler_backend(
    requested_level: str,
    gfx: str | None,
    *,
    rocprofiler_compute: bool = False,
    omniperf: bool = False,
    rocprofv3: bool = True,
) -> ProfilerReadiness:
    """Select an internal profiling readiness route for ROCm diagnostics.

    The return value is descriptive only. It does not change benchmark timing,
    output formats, or CLI behavior.
    """
    level = requested_level.strip().lower()
    if level not in {"basic", "full"}:
        raise SolExecBenchError(
            DiagnosticStage.ENVIRONMENT,
            "requested profiling level must be 'basic' or 'full'",
            hint="Use 'basic' for portable readiness checks or 'full' for architecture-specific profiling.",
        )

    if level == "basic":
        backend = ProfilerBackend.ROCPROFV3 if rocprofv3 else ProfilerBackend.SKIP
        return ProfilerReadiness(
            backend=backend,
            reason="basic profile requested"
            if rocprofv3
            else "rocprofv3 not available for basic profiling",
            fallback_applied=not rocprofv3,
            effective_level="basic" if rocprofv3 else "skip",
        )

    family = classify_gfx(gfx)
    if family == "cdna3":
        if rocprofiler_compute:
            return ProfilerReadiness(
                backend=ProfilerBackend.ROCPROFILER_COMPUTE,
                reason="CDNA 3 detected and rocprofiler-compute is available",
                fallback_applied=False,
                effective_level="full",
            )
        if omniperf:
            return ProfilerReadiness(
                backend=ProfilerBackend.OMNIPERF,
                reason="CDNA 3 detected; Omniperf fallback is available",
                fallback_applied=False,
                effective_level="full",
            )
        return ProfilerReadiness(
            backend=ProfilerBackend.ROCPROFV3 if rocprofv3 else ProfilerBackend.SKIP,
            reason="CDNA 3 full profiling requested but no CDNA-specific profiler is available",
            fallback_applied=True,
            effective_level="basic" if rocprofv3 else "skip",
        )

    if family.startswith("rdna"):
        return ProfilerReadiness(
            backend=ProfilerBackend.ROCPROFV3 if rocprofv3 else ProfilerBackend.SKIP,
            reason=f"{family.upper()} detected; rocprofv3 readiness route selected"
            if rocprofv3
            else f"{family.upper()} detected but rocprofv3 is unavailable",
            fallback_applied=not rocprofv3,
            effective_level="full" if rocprofv3 else "skip",
        )

    return ProfilerReadiness(
        backend=ProfilerBackend.SKIP,
        reason="GPU architecture unknown; profiling readiness skipped",
        fallback_applied=True,
        effective_level="skip",
    )


def rocm_tool_diagnostics(
    *,
    which: Callable[[str], str | None] = shutil.which,
) -> list[StageDiagnostic]:
    """Return diagnostics for ROCm tools used by the benchmark environment."""
    tools = {
        "hipcc": "Install ROCm HIP compiler tooling and ensure hipcc is on PATH.",
        "rocminfo": "Install ROCm runtime tools or source the ROCm environment setup.",
        "rocm-smi": "Install ROCm SMI tooling if hardware state capture is required.",
        "rocprofv3": "Install ROCm profiling tools for profiler readiness checks.",
    }
    diagnostics: list[StageDiagnostic] = []
    for tool, hint in tools.items():
        available = detect_tool(tool, which)
        diagnostics.append(
            StageDiagnostic(
                stage=DiagnosticStage.ENVIRONMENT,
                status="available" if available else "missing",
                message=f"{tool} {'found' if available else 'not found'}",
                hint=None if available else hint,
            )
        )
    return diagnostics


def local_gfx_target(
    *,
    check_output: Callable[..., str] = subprocess.check_output,
) -> str | None:
    """Best-effort local gfx target detection through ROCm command output."""
    for cmd in (["rocm_agent_enumerator", "-name"], ["rocminfo"]):
        try:
            output = check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        except Exception:
            continue
        for token in output.replace("\n", " ").split():
            if token.startswith("gfx") and token != "gfx000":
                return token
    return None
