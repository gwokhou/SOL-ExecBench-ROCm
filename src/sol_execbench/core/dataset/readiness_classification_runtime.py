# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Runtime dependency readiness classification handlers."""

from __future__ import annotations

from pathlib import Path

from .readiness_classification_state import ReadinessClassificationState
from .readiness_hints import (
    FLASHINFER_RUNTIME_BUCKET_TO_REASON,
    _flashinfer_reference_is_runtime_dependent,
    _flashinfer_semantic_bucket,
    _reference_has_nvidia_blocker,
)
from .readiness_models import ReadinessClass

def classify_flashinfer(
    state: ReadinessClassificationState,
    *,
    dataset_root: Path,
    solution_hints: set[str],
) -> bool:
    is_flashinfer = (
        state.problem.category == "FlashInfer-Bench"
        or "flashinfer" in state.problem.problem_path.lower()
        or "flashinfer_runtime" in solution_hints
    )
    if not is_flashinfer:
        return False
    flashinfer_bucket = _flashinfer_semantic_bucket(state.problem)
    runtime_dependent = _flashinfer_reference_is_runtime_dependent(
        state.problem, dataset_root
    )
    if not runtime_dependent:
        reason_code = (
            "flashinfer_pytorch_compatible_reference"
            if flashinfer_bucket == "pytorch_compatible"
            else "flashinfer_pytorch_semantic_reference_migrated"
        )
        bucket_name = flashinfer_bucket.replace("_", " ")
        state.add_reason(
            reason_code,
            (
                "FlashInfer workload has a PyTorch-level reference migrated for "
                f"{bucket_name} semantics and no direct FlashInfer runtime dependency."
            ),
            "Run bounded execution closure in Phase 55.",
            state.problem.problem_path,
        )
        return True

    state.status = "runtime_blocked"
    state.readiness_class = ReadinessClass.FLASHINFER_SPECIFIC
    state.layers.reference_execution = "blocked"
    reason_code = FLASHINFER_RUNTIME_BUCKET_TO_REASON.get(
        flashinfer_bucket,
        "flashinfer_runtime_unknown",
    )
    bucket_name = flashinfer_bucket.replace("_", " ")
    state.add_reason(
        reason_code,
        (
            f"FlashInfer-Bench workload is classified as {bucket_name} runtime-dependent "
            "and needs runtime semantics not guaranteed by PyTorch alone."
        ),
        "Route through a dedicated ROCm FlashInfer compatibility/port path before execution.",
        state.problem.problem_path,
    )
    state.add_blocker(
        code=reason_code,
        blocker_type="flashinfer_runtime_dependency",
        message=(
            f"FlashInfer-Bench workload is classified as {bucket_name} "
            "runtime-dependent and cannot be attempted without a FlashInfer compatibility path."
        ),
        next_action=(
            "Route through a dedicated ROCm FlashInfer compatibility/port "
            "path before execution."
        ),
        evidence_path=state.problem.problem_path,
    )
    return True


def classify_nvidia_reference(state: ReadinessClassificationState) -> bool:
    if not _reference_has_nvidia_blocker(state.problem):
        return False
    # Lexical false positives are already filtered out by inventory.py
    # (Phase 172 Quant triage), so any remaining hint is a true blocker.
    state.status = "unsupported_nvidia_only_path"
    state.readiness_class = ReadinessClass.ROCM_PORT_NEEDED
    state.layers.reference_execution = "blocked"
    message = "Static NVIDIA/CUDA runtime hint detected."
    next_action = "Port or exclude NVIDIA-only reference path."
    state.add_reason(
        "nvidia_cuda_runtime_hint",
        message,
        next_action,
        state.problem.problem_path,
    )
    state.add_blocker(
        code="nvidia_cuda_runtime_hint",
        blocker_type="cuda_kernel_dependency",
        message=message,
        next_action=next_action,
        evidence_path=state.problem.problem_path,
    )
    return True


def classify_cuda_solution(
    state: ReadinessClassificationState,
    *,
    solution_hints: set[str],
) -> bool:
    if "cuda_kernel" not in solution_hints:
        return False
    state.status = "unsupported_nvidia_only_path"
    state.readiness_class = ReadinessClass.ROCM_PORT_NEEDED
    state.layers.candidate_execution = "blocked"
    message = "Migrated solution still contains CUDA/NVIDIA kernel dependencies."
    next_action = (
        "Port candidate solution to HIP, Triton ROCm, or a ROCm library before execution."
    )
    state.add_reason(
        "cuda_solution_dependency",
        message,
        next_action,
        state.problem.problem_path,
    )
    state.add_blocker(
        code="cuda_solution_dependency",
        blocker_type="cuda_kernel_dependency",
        message=message,
        next_action=next_action,
        evidence_path=state.problem.problem_path,
    )
    return True


def classify_nvidia_dsl(
    state: ReadinessClassificationState,
    *,
    solution_hints: set[str],
) -> bool:
    if "nvidia_dsl" not in solution_hints:
        return False
    state.status = "unsupported_nvidia_only_path"
    state.readiness_class = ReadinessClass.UNSUPPORTED
    state.layers.candidate_execution = "blocked"
    message = "Migrated solution depends on NVIDIA-specific DSL/runtime code."
    next_action = "Replace with a ROCm-native solution path before execution."
    state.add_reason(
        "unsupported_nvidia_dsl",
        message,
        next_action,
        state.problem.problem_path,
    )
    state.add_blocker(
        code="unsupported_nvidia_dsl",
        blocker_type="cuda_kernel_dependency",
        message=message,
        next_action=next_action,
        evidence_path=state.problem.problem_path,
    )
    return True
