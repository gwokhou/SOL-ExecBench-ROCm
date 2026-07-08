# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Internal ROCm diagnostics and stage-aware failure helpers.

These helpers intentionally do not define a new CLI or trace schema. They are a
small internal surface for making ROCm readiness and failure messages more
consistent while preserving existing public contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


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


@dataclass(frozen=True)
class ValidationReadiness:
    """Internal validation-readiness metadata for future hardware runs."""

    target_family: str
    ready: bool
    claim: str
    commands: tuple[str, ...]
    evidence_required: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable readiness payload."""
        return {
            "target_family": self.target_family,
            "ready": self.ready,
            "claim": self.claim,
            "commands": list(self.commands),
            "evidence_required": list(self.evidence_required),
            "acceptance_criteria": list(self.acceptance_criteria),
            "blockers": list(self.blockers),
        }


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
    def status(self) -> str:
        """Short readiness label."""
        return "available" if self.ready else "missing"

    def to_diagnostic(self) -> StageDiagnostic:
        """Return a stage diagnostic with actionable missing dependency detail."""
        if self.ready:
            return StageDiagnostic(
                stage=DiagnosticStage.ENVIRONMENT,
                status="available",
                message=f"{self.spec.name} library dependencies found",
            )

        missing: list[str] = []
        if self.missing_headers:
            missing.append("headers: " + ", ".join(self.missing_headers))
        if self.missing_libraries:
            missing.append("libraries: " + ", ".join(self.missing_libraries))
        return StageDiagnostic(
            stage=DiagnosticStage.ENVIRONMENT,
            status="missing",
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


CDNA3_VALIDATION_COMMANDS: tuple[str, ...] = (
    "uv run --no-sync pytest tests/",
    "uv run python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())'",
    "rocm-smi --showproductname --showdriverversion --showhw || true",
    'rocminfo | grep -E "Name: *gfx94|Marketing Name" || true',
)


CDNA3_EVIDENCE_REQUIRED: tuple[str, ...] = (
    "Exact GPU name and gfx94* architecture",
    "ROCm/HIP/PyTorch versions",
    "Full pytest command and final pass/skip/fail counts",
    "Expected skips and CDNA 3-specific deviations",
)


CDNA3_ACCEPTANCE_CRITERIA: tuple[str, ...] = (
    "Full adapted pytest suite completes successfully on a real CDNA 3 GPU",
    "Recorded environment reports gfx94*",
    "Support matrix claim is updated only after recorded evidence exists",
)


RDNA4_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "environment_sidecar",
    "execution_closure",
    "per_problem_traces",
    "clock_lock_evidence",
    "timing_sidecars",
    "profiler_backed_timing",
    "amd_native_score_report",
    "amd_sol_sidecars",
    "solar_derivation_sidecars",
    "derived_exclusion_retry",
    "missing_trace_triage",
    "failure_triage",
    "claim_boundary_report",
)


RDNA4_VALIDATION_RESULT_CATEGORIES: tuple[str, ...] = (
    "attempted_passed",
    "reference_gpu_oom",
    "input_generation_gpu_oom",
    "execution_timeout",
    "gpu_oom_no_trace",
    "user_function_gpu_oom",
    "timing_gpu_oom",
    "incorrect_numerical",
    "readiness_blocked",
    "derived_sidecar_memory_blockers",
)


MI300X_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "pytest_full_suite_log",
    "run_dataset_summary",
    "environment_report",
    "clock_lock_evidence",
    "per_problem_traces",
    "rocm_timing_evidence",
    "amd_native_score_report",
    "fp8_validation_result",
    "nvfp4_mxfp4_deferred_status",
)


MI300X_VALIDATION_RESULT_CATEGORIES: tuple[str, ...] = (
    "expected_skips",
    "missing_tools",
    "functional_failures",
    "timing_instability",
    "missing_evidence",
    "fp8_validation",
    "deferred_quantization_formats",
)


MI300X_FP8_READINESS: tuple[str, ...] = (
    "MI300X, as a CDNA 3 GPU, can validate FP8 behavior once hardware access exists",
    "NVFP4/MXFP4 validation is deferred until a suitable AMD hardware and methodology path exists",
)
