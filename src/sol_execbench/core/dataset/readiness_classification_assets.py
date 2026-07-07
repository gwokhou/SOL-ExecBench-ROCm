# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Asset and non-Quant low-precision readiness classification handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .readiness_classification_precision import _add_blackwell_low_precision_blocker
from .readiness_classification_state import ReadinessClassificationState
from .readiness_hints import _blackwell_low_precision, _low_precision_or_quant
from .readiness_models import ReadinessClass

def classify_safetensors_and_low_precision(
    state: ReadinessClassificationState,
    *,
    dataset_root: Path,
    solution_hints: set[str],
) -> bool:
    missing_safetensors: list[tuple[dict[str, Any], str]] = []
    dataset_root_resolved = Path(dataset_root).resolve()
    for ref in state.workload.safetensors_refs:
        raw_path = Path(ref["path"])
        reason_code = "safetensors_asset_missing"
        if raw_path.is_absolute():
            reason_code = "safetensors_path_outside_dataset_root"
            missing_safetensors.append((ref, reason_code))
            continue
        candidate = (dataset_root_resolved / raw_path).resolve()
        if not candidate.is_relative_to(dataset_root_resolved):
            reason_code = "safetensors_path_outside_dataset_root"
            missing_safetensors.append((ref, reason_code))
            continue
        if not candidate.exists():
            missing_safetensors.append((ref, reason_code))
    if missing_safetensors:
        state.status = "runtime_blocked"
        state.readiness_class = ReadinessClass.BLOCKED_MISSING_EVIDENCE
        state.layers.input_generation = "needs_asset"
        for ref, reason_code in missing_safetensors:
            message = (
                f"Safetensors asset {ref['path']} for key {ref['tensor_key']} "
                "is unavailable inside the dataset root."
            )
            next_action = (
                "Acquire asset inside the dataset root or configure a safe blob root "
                "before execution."
            )
            state.add_reason(reason_code, message, next_action, ref["path"])
            state.add_blocker(
                code=reason_code,
                blocker_type="missing_blob",
                message=message,
                next_action=next_action,
                evidence_path=ref["path"],
            )
    elif (
        _blackwell_low_precision(state.problem, state.workload)
        or "blackwell_low_precision" in solution_hints
    ):
        _add_blackwell_low_precision_blocker(
            state,
            message=(
                "NVFP4/Blackwell low-precision workload still needs CDNA4 hardware "
                "evidence before validation claims."
            ),
        )
    elif _low_precision_or_quant(state.problem, state.workload):
        # Non-Quant low-precision workload
        state.status = "needs_hardware_evidence"
        state.readiness_class = ReadinessClass.BLOCKED_MISSING_EVIDENCE
        state.layers.hardware_validation = "needed"
        message = (
            "Low-precision workload needs hardware validation evidence before "
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
    return True
