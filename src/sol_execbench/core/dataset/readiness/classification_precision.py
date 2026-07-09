# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Quant and low-precision readiness classification handlers."""

from __future__ import annotations

from ..low_precision import (
    CDNA4_VALIDATION_DEFERRED_CODE,
    LOW_PRECISION_COMPATIBILITY_EVIDENCE_CODE,
)
from .classification_state import ReadinessClassificationState
from .hints import _blackwell_low_precision, _low_precision_or_quant
from .models import ReadinessClass


def classify_quant(state: ReadinessClassificationState) -> bool:
    if state.problem.category != "Quant":
        return False
    # Refined Quant routing based on actual dtypes and formats.
    # Must come before custom_inputs so NVFP4/low-precision Quant problems
    # are blocked even when they also have custom input entrypoints.
    if _blackwell_low_precision(state.problem, state.workload):
        _add_blackwell_low_precision_blocker(
            state,
            message=(
                "NVFP4/Blackwell Quant workload still needs CDNA4 hardware evidence "
                "before validation claims."
            ),
        )
    elif _low_precision_or_quant(state.problem, state.workload):
        state.status = "needs_hardware_evidence"
        state.readiness_class = ReadinessClass.BLOCKED_MISSING_EVIDENCE
        state.layers.hardware_validation = "needed"
        message = (
            "Quant workload with FP8 dtypes needs hardware validation evidence before "
            "validation claims."
        )
        next_action = "Collect hardware evidence during execution closure."
        state.add_reason(
            "low_precision_requires_hardware_evidence",
            message,
            next_action,
            state.problem.problem_path,
        )
        state.add_blocker(
            code="low_precision_requires_hardware_evidence",
            blocker_type="low_precision_format_dependency",
            message=message,
            next_action=next_action,
            evidence_path=state.problem.problem_path,
        )
    else:
        state.add_reason(
            "quant_rocm_compatible_reference",
            "Quant semantic reference is PyTorch ROCm-compatible; readiness does not imply low-precision hardware authority.",
            "Run bounded execution closure in Phase 55.",
            state.problem.problem_path,
        )
        if (
            state.problem.definition
            and state.problem.definition.reference_runtime_false_positive_evidence
        ):
            for (
                evidence
            ) in state.problem.definition.reference_runtime_false_positive_evidence:
                state.add_reason(
                    "quant_cuda_false_positive_cleared",
                    f"Quant lexical CUDA hint '{evidence['token']}' was a {evidence['match_kind']} and ignored as a false positive.",
                    "Run bounded execution closure in Phase 55.",
                    state.problem.problem_path,
                )
    return True


def _add_blackwell_low_precision_blocker(
    state: ReadinessClassificationState,
    *,
    message: str,
) -> None:
    state.status = "needs_hardware_evidence"
    state.readiness_class = ReadinessClass.NVFP4_BLACKWELL_SPECIFIC
    state.layers.hardware_validation = "needed"
    state.add_reason(
        LOW_PRECISION_COMPATIBILITY_EVIDENCE_CODE,
        "Phase 134 CPU semantic compatibility path is available for NVFP4/Blackwell low-precision metadata, packing, and fallback behavior.",
        "Use the compatibility path for migrated definitions, then collect CDNA4 hardware evidence before validation claims.",
        state.problem.problem_path,
    )
    state.add_reason(
        CDNA4_VALIDATION_DEFERRED_CODE,
        message,
        "Collect CDNA4 hardware evidence before validation or performance claims.",
        state.problem.problem_path,
    )
    state.add_blocker(
        code=CDNA4_VALIDATION_DEFERRED_CODE,
        blocker_type="low_precision_format_dependency",
        message=message,
        next_action="Collect CDNA4 hardware evidence before validation or performance claims.",
        evidence_path=state.problem.problem_path,
    )
