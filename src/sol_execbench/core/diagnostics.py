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
from collections.abc import Mapping
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

    def to_dict(self) -> dict[str, object]:
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

MI300X_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "pytest_full_suite_log",
    "run_dataset_summary",
    "environment_report",
    "clock_lock_evidence",
)

MI300X_FP8_READINESS: tuple[str, ...] = (
    "MI300X/CDNA 3 can validate FP8 behavior once hardware access exists",
    "NVFP4/MXFP4 validation is deferred until a suitable AMD hardware and methodology path exists",
)


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


def cdna3_validation_readiness(
    gfx: str | None,
    *,
    tool_diagnostics: list[StageDiagnostic] | tuple[StageDiagnostic, ...] = (),
) -> ValidationReadiness:
    """Return readiness metadata for a future real CDNA 3 validation run.

    This helper never claims a hardware-validation pass. It only describes
    whether the supplied target and tooling are ready to attempt one.
    """
    family = classify_gfx(gfx)
    blockers: list[str] = []
    if family != "cdna3":
        if family == "rdna4":
            blockers.append(
                "Detected RDNA 4 target; CDNA 3 validation requires gfx94* hardware."
            )
        else:
            blockers.append(
                "CDNA 3 validation requires an AMD gfx94* target such as gfx942."
            )

    missing_tools = [
        diagnostic.message.removesuffix(" not found")
        for diagnostic in tool_diagnostics
        if diagnostic.status == "missing"
    ]
    if missing_tools:
        blockers.append(
            "Missing ROCm validation tools: " + ", ".join(sorted(missing_tools))
        )

    ready = family == "cdna3" and not blockers
    claim = (
        "cdna3_readiness_implemented" if ready else "cdna3_hardware_validation_deferred"
    )
    return ValidationReadiness(
        target_family=family,
        ready=ready,
        claim=claim,
        commands=CDNA3_VALIDATION_COMMANDS,
        evidence_required=CDNA3_EVIDENCE_REQUIRED,
        acceptance_criteria=CDNA3_ACCEPTANCE_CRITERIA,
        blockers=tuple(blockers),
    )


def mi300x_validation_claim_blockers(evidence: Mapping[str, object]) -> tuple[str, ...]:
    """Return blockers that prevent marking MI300X/CDNA3 reports as validated.

    This pure guard is intentionally strict. It does not run hardware checks; it
    validates that a report has already recorded the minimum evidence required
    to upgrade from readiness/deferred status to hardware-validated status.
    """
    blockers: list[str] = []
    gpu_name = str(evidence.get("gpu_name", ""))
    gfx = str(evidence.get("gfx", ""))
    if "MI300X" not in gpu_name:
        blockers.append("gpu_name must identify AMD Instinct MI300X")
    if classify_gfx(gfx) != "cdna3":
        blockers.append("gfx must be a CDNA 3 gfx94* target")
    if not evidence.get("rocm_version"):
        blockers.append("rocm_version must be recorded")
    if evidence.get("clocks_locked") is not True:
        blockers.append("clock-lock evidence must record clocks_locked=True")
    if evidence.get("full_suite_passed") is not True:
        blockers.append("full adapted suite must pass on MI300X")

    artifacts = evidence.get("artifacts", ())
    if not isinstance(artifacts, (list, tuple, set)):
        artifacts = ()
    missing_artifacts = [
        artifact for artifact in MI300X_REQUIRED_ARTIFACTS if artifact not in artifacts
    ]
    if missing_artifacts:
        blockers.append("missing validation artifacts: " + ", ".join(missing_artifacts))

    fp8_status = evidence.get("fp8_validation")
    if fp8_status not in {"passed", "deferred_no_case"}:
        blockers.append("fp8_validation must be 'passed' or 'deferred_no_case'")

    nvfp4_status = evidence.get("nvfp4_mxfp4_validation")
    if nvfp4_status != "deferred_no_amd_path":
        blockers.append("NVFP4/MXFP4 validation must remain deferred_no_amd_path")
    return tuple(blockers)


def can_mark_mi300x_hardware_validated(evidence: Mapping[str, object]) -> bool:
    """Return whether evidence is sufficient for an MI300X validation claim."""
    return not mi300x_validation_claim_blockers(evidence)


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
