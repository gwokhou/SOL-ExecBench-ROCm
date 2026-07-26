# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Internal ROCm diagnostics and stage-aware failure helpers.

These helpers intentionally do not define a new CLI or trace schema. They are a
small internal surface for making ROCm readiness and failure messages more
consistent while preserving existing public contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DiagnosticStage(StrEnum):
    """High-level evaluation stages used in actionable failure messages."""

    PARSE = "parse"
    PACKAGE = "package"
    COMPILE = "compile"
    RUNTIME = "runtime"
    VERIFY = "verify"
    TIMING = "timing"
    ENVIRONMENT = "environment"


class DiagnosticStatus(StrEnum):
    """Closed readiness and reporting states for stage diagnostics."""

    FAILED = "failed"
    AVAILABLE = "available"
    MISSING = "missing"
    WARNING = "warning"


@dataclass(frozen=True)
class StageDiagnostic:
    """Actionable diagnostic information for one stage or tool."""

    stage: DiagnosticStage
    status: DiagnosticStatus
    message: str
    hint: str | None = None

    def __post_init__(self) -> None:
        """Normalize public constructor values and reject unknown states."""
        object.__setattr__(self, "stage", DiagnosticStage(self.stage))
        object.__setattr__(self, "status", DiagnosticStatus(self.status))

    def format(self) -> str:
        """Return a compact human-readable diagnostic message."""
        prefix = f"{self.stage}: {self.status}: {self.message}"
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
        super().__init__(
            StageDiagnostic(stage, DiagnosticStatus.FAILED, message, hint).format()
        )
        self.stage = stage
        self.hint = hint


class ProfilerBackend(StrEnum):
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


@dataclass(frozen=True)
class RocmLibrarySpec:
    """Filesystem-level requirements for one ROCm library category."""

    name: str
    headers: tuple[str, ...]
    libraries: tuple[str, ...]
    packages: tuple[str, ...]
    hint: str


@dataclass(frozen=True)
class RocmLibraryReadiness:
    """Readiness result for a ROCm library dependency group."""

    spec: RocmLibrarySpec
    header_paths: tuple[str, ...]
    library_paths: tuple[str, ...]
    missing_headers: tuple[str, ...]
    missing_libraries: tuple[str, ...]

    @property
    def ready(self) -> bool:
        """Whether all required headers and libraries were found."""
        return not self.missing_headers and not self.missing_libraries

    @property
    def status(self) -> DiagnosticStatus:
        """Short readiness label."""
        return DiagnosticStatus.AVAILABLE if self.ready else DiagnosticStatus.MISSING

    def to_diagnostic(self) -> StageDiagnostic:
        """Return a stage diagnostic with actionable missing dependency detail."""
        if self.ready:
            return StageDiagnostic(
                stage=DiagnosticStage.ENVIRONMENT,
                status=DiagnosticStatus.AVAILABLE,
                message=f"{self.spec.name} library dependencies found",
            )

        missing: list[str] = []
        if self.missing_headers:
            missing.append("headers: " + ", ".join(self.missing_headers))
        if self.missing_libraries:
            missing.append("libraries: " + ", ".join(self.missing_libraries))
        return StageDiagnostic(
            stage=DiagnosticStage.ENVIRONMENT,
            status=DiagnosticStatus.MISSING,
            message=f"{self.spec.name} missing " + "; ".join(missing),
            hint=self.spec.hint,
        )


ROCM_LIBRARY_SPECS: dict[str, RocmLibrarySpec] = {
    "hipblas": RocmLibrarySpec(
        name="hipBLAS",
        headers=("hipblas/hipblas.h",),
        libraries=("hipblas",),
        packages=("hipblas", "hipblas-dev"),
        hint="Install hipBLAS development files from the ROCm distribution.",
    ),
    "miopen": RocmLibrarySpec(
        name="MIOpen",
        headers=("miopen/miopen.h",),
        libraries=("MIOpen",),
        packages=("miopen-hip", "miopen-hip-dev"),
        hint="Install MIOpen HIP development files from the ROCm distribution.",
    ),
    "ck": RocmLibrarySpec(
        name="Composable Kernel",
        headers=("ck/ck.hpp",),
        libraries=(),
        packages=("composablekernel-dev", "composable-kernel"),
        hint=(
            "Install Composable Kernel headers, usually from the ROCm "
            "rocm-libraries or composable-kernel package."
        ),
    ),
    "rocwmma": RocmLibrarySpec(
        name="rocWMMA",
        headers=("rocwmma/rocwmma.hpp",),
        libraries=(),
        packages=("rocwmma-dev", "rocwmma"),
        hint="Install rocWMMA development headers from the ROCm distribution.",
    ),
}
