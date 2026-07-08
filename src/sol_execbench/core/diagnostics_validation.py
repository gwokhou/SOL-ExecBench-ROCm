# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Internal ROCm diagnostics and stage-aware failure helpers.

These helpers intentionally do not define a new CLI or trace schema. They are a
small internal surface for making ROCm readiness and failure messages more
consistent while preserving existing public contracts.
"""

from __future__ import annotations

from collections.abc import Mapping

from sol_execbench.core.diagnostics_models import (
    CDNA3_ACCEPTANCE_CRITERIA,
    CDNA3_EVIDENCE_REQUIRED,
    CDNA3_VALIDATION_COMMANDS,
    MI300X_REQUIRED_ARTIFACTS,
    MI300X_VALIDATION_RESULT_CATEGORIES,
    RDNA4_REQUIRED_ARTIFACTS,
    RDNA4_VALIDATION_RESULT_CATEGORIES,
    DiagnosticStage,
    ProfilerBackend,
    ProfilerReadiness,
    SolExecBenchError,
    StageDiagnostic,
    ValidationReadiness,
)


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
    """Return blockers that prevent marking MI300X-as-CDNA3 reports as validated.

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

    result_categories = evidence.get("result_categories", ())
    if not isinstance(result_categories, (list, tuple, set)):
        result_categories = ()
    missing_categories = [
        category
        for category in MI300X_VALIDATION_RESULT_CATEGORIES
        if category not in result_categories
    ]
    if missing_categories:
        blockers.append(
            "missing validation result categories: " + ", ".join(missing_categories)
        )

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


def rdna4_validation_claim_blockers(evidence: Mapping[str, object]) -> tuple[str, ...]:
    """Return blockers that prevent a stronger RDNA4 validation claim.

    Current project evidence supports bounded RDNA4 `gfx1200` ready-subset
    wording. This guard is for future claim upgrades and keeps full paper-scale,
    authoritative timing, score, or broader hardware wording blocked until the
    evidence record explicitly carries the missing pieces.
    """
    blockers: list[str] = []
    gpu_name = str(evidence.get("gpu_name", ""))
    gfx = str(evidence.get("gfx", ""))
    if not gpu_name:
        blockers.append("gpu_name must be recorded")
    if classify_gfx(gfx) != "rdna4":
        blockers.append("gfx must be an RDNA 4 gfx12* target")
    if not evidence.get("rocm_version"):
        blockers.append("rocm_version must be recorded")
    if evidence.get("clocks_locked") is not True:
        blockers.append("clock-lock evidence must record clocks_locked=True")
    if evidence.get("profiler_backed_timing") is not True:
        blockers.append(
            "profiler-backed rocprofv3 kernel activity timing must be recorded"
        )
    if evidence.get("full_paper_problem_count") != 235:
        blockers.append("full paper denominator must account for 235 problems")

    failed_workload_count = _int_evidence(evidence.get("failed_workload_count"))
    if failed_workload_count is None:
        blockers.append("failed_workload_count must be recorded")
    elif failed_workload_count != 0:
        blockers.append("failed workload count must be zero for claim upgrade")

    derived_blocker_count = _int_evidence(evidence.get("derived_sidecar_blocker_count"))
    if derived_blocker_count is None:
        blockers.append("derived_sidecar_blocker_count must be recorded")
    elif derived_blocker_count != 0:
        blockers.append("derived sidecar blocker count must be zero for claim upgrade")

    artifacts = evidence.get("artifacts", ())
    if not isinstance(artifacts, (list, tuple, set)):
        artifacts = ()
    missing_artifacts = [
        artifact for artifact in RDNA4_REQUIRED_ARTIFACTS if artifact not in artifacts
    ]
    if missing_artifacts:
        blockers.append("missing validation artifacts: " + ", ".join(missing_artifacts))

    result_categories = evidence.get("result_categories", ())
    if not isinstance(result_categories, (list, tuple, set)):
        result_categories = ()
    missing_categories = [
        category
        for category in RDNA4_VALIDATION_RESULT_CATEGORIES
        if category not in result_categories
    ]
    if missing_categories:
        blockers.append(
            "missing validation result categories: " + ", ".join(missing_categories)
        )
    return tuple(blockers)


def can_mark_rdna4_validation_upgraded(evidence: Mapping[str, object]) -> bool:
    """Return whether RDNA4 evidence is sufficient for a stronger claim."""
    return not rdna4_validation_claim_blockers(evidence)


def _int_evidence(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
