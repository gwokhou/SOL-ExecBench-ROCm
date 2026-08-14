# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared fail-closed contract for large batch GPU qualification."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.control_plane_schema_versions import (
    BatchGPUQualificationArtifactKind,
    ExecutionControlSchema,
)
from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    NonEmptyString,
    StrictArtifactModel,
)
from sol_execbench.core.integrity import SHA256Digest, sha256_file

_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class BatchGPUQualificationStage(StrEnum):
    """Ordered public stage names shared by every large batch GPU task."""

    STATIC = "static"
    CANARY = "canary"
    FULL = "full"

    @property
    def command(self) -> str:
        """Return the canonical CLI command for this stage."""
        return f"qualify-{self.value}"


class LargeBatchGPUTask(StrEnum):
    """Current closed inventory of repository-owned large batch GPU tasks."""

    AKA_TOLERANCE_CALIBRATION = "aka_tolerance_calibration"
    DIAGNOSTIC_COUNTER_COLLECTION = "diagnostic_counter_collection"
    RDNA4_DIAGNOSTIC_CALIBRATION = "rdna4_diagnostic_calibration"
    RDNA4_RESOURCE_PEAK_CALIBRATION = "rdna4_resource_peak_calibration"
    RELEASE_EVALUATION = "release_evaluation"
    SOLAR_CROSS_PATH_FOCUS = "solar_cross_path_focus"
    SOLAR_RELEASE_BUILD = "solar_release_build"


class QualificationArtifact(StrictArtifactModel):
    """One content-bound artifact beneath an isolated qualification root."""

    model_config = _CONFIG
    path: NonEmptyString
    sha256: SHA256Digest
    size_bytes: int = Field(ge=0)


class BatchGPUQualificationReceipt(CurrentSchemaModel):
    """Verified result for one partition within a qualification stage."""

    model_config = _CONFIG
    current_schema_version = ExecutionControlSchema.BATCH_GPU_QUALIFICATION
    current_artifact_kind = BatchGPUQualificationArtifactKind.RECEIPT

    schema_version: Literal[ExecutionControlSchema.BATCH_GPU_QUALIFICATION] = (
        ExecutionControlSchema.BATCH_GPU_QUALIFICATION
    )
    artifact_kind: Literal[BatchGPUQualificationArtifactKind.RECEIPT] = (
        BatchGPUQualificationArtifactKind.RECEIPT
    )
    stage: BatchGPUQualificationStage
    partition: NonEmptyString
    item_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    input_sha256: SHA256Digest
    artifacts: tuple[QualificationArtifact, ...] = Field(min_length=1)
    all_passed: Literal[True] = True
    performance_authority: Literal[False] = False

    @model_validator(mode="after")
    def item_ids_are_unique(self) -> BatchGPUQualificationReceipt:
        """Reject ambiguous item coverage."""
        if len(self.item_ids) != len(set(self.item_ids)):
            raise ValueError("qualification receipt repeats item IDs")
        return self


class BatchGPUQualificationGate(CurrentSchemaModel):
    """Content-bound completion gate required before a large batch GPU run."""

    model_config = _CONFIG
    current_schema_version = ExecutionControlSchema.BATCH_GPU_QUALIFICATION
    current_artifact_kind = BatchGPUQualificationArtifactKind.GATE

    schema_version: Literal[ExecutionControlSchema.BATCH_GPU_QUALIFICATION] = (
        ExecutionControlSchema.BATCH_GPU_QUALIFICATION
    )
    artifact_kind: Literal[BatchGPUQualificationArtifactKind.GATE] = (
        BatchGPUQualificationArtifactKind.GATE
    )
    task: LargeBatchGPUTask
    stage: BatchGPUQualificationStage
    scope_id: NonEmptyString
    subject_sha256: SHA256Digest
    runner_sha256: SHA256Digest
    configuration_sha256: SHA256Digest
    source_revision: NonEmptyString
    parent_gate_sha256: SHA256Digest | None = None
    item_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    receipts: tuple[BatchGPUQualificationReceipt, ...] = Field(min_length=1)
    created_at: NonEmptyString
    purpose: Literal["correctness_qualification"] = "correctness_qualification"
    performance_authority: Literal[False] = False

    @model_validator(mode="after")
    def chain_and_coverage_are_consistent(self) -> BatchGPUQualificationGate:
        """Require exact parent shape and non-overlapping receipt coverage."""
        if len(self.item_ids) != len(set(self.item_ids)):
            raise ValueError("qualification gate repeats item IDs")
        if self.stage is BatchGPUQualificationStage.STATIC:
            if self.parent_gate_sha256 is not None:
                raise ValueError("static qualification cannot have a parent")
        elif self.parent_gate_sha256 is None:
            raise ValueError("GPU qualification requires a parent gate")
        receipt_items = tuple(
            item_id for receipt in self.receipts for item_id in receipt.item_ids
        )
        if set(receipt_items) != set(self.item_ids):
            raise ValueError("qualification receipts do not cover gate items")
        if len(receipt_items) != len(set(receipt_items)):
            raise ValueError("qualification receipts overlap item IDs")
        if any(receipt.stage is not self.stage for receipt in self.receipts):
            raise ValueError("qualification receipt stage mismatch")
        return self


def qualification_gate_path(
    root: Path,
    stage: BatchGPUQualificationStage,
) -> Path:
    """Return the canonical gate path below one isolated task root."""
    return root / stage.value / "gate.json"


def qualification_parent_stage(
    stage: BatchGPUQualificationStage,
) -> BatchGPUQualificationStage | None:
    """Return the mandatory immediate predecessor for *stage*."""
    if stage is BatchGPUQualificationStage.STATIC:
        return None
    if stage is BatchGPUQualificationStage.CANARY:
        return BatchGPUQualificationStage.STATIC
    return BatchGPUQualificationStage.CANARY


def require_isolated_qualification_root(
    qualification_root: Path,
    protected_root: Path,
) -> Path:
    """Reject qualification output nested with formal batch output."""
    qualification = qualification_root.resolve()
    protected = protected_root.resolve()
    if qualification == protected:
        raise ValueError(
            "qualification root must be isolated from batch output"
        )
    if qualification.is_relative_to(protected):
        raise ValueError("qualification root cannot be inside batch output")
    if protected.is_relative_to(qualification):
        raise ValueError("qualification root cannot contain batch output")
    return qualification


def qualification_artifact(root: Path, path: Path) -> QualificationArtifact:
    """Build a verified root-relative artifact reference."""
    resolved_root = root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise ValueError(f"qualification artifact escapes root: {path}")
    return QualificationArtifact(
        path=resolved.relative_to(resolved_root).as_posix(),
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def verify_qualification_artifact(
    root: Path,
    artifact: QualificationArtifact,
) -> Path:
    """Verify one recorded artifact without following a root escape."""
    resolved_root = root.resolve()
    candidate = resolved_root / artifact.path
    if candidate.is_symlink():
        raise ValueError(
            f"qualification artifact cannot be a symlink: {candidate}"
        )
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise ValueError(f"qualification artifact is missing: {candidate}")
    if resolved.stat().st_size != artifact.size_bytes:
        raise ValueError(f"qualification artifact size drift: {candidate}")
    if sha256_file(resolved) != artifact.sha256:
        raise ValueError(f"qualification artifact hash drift: {candidate}")
    return resolved


def select_risk_first_axis_extrema[ItemT](
    items: Sequence[ItemT],
    *,
    item_id: Callable[[ItemT], str],
    axes: Callable[[ItemT], Mapping[str, int]],
) -> tuple[ItemT, ...]:
    """Select every per-axis extreme and order larger shapes first."""
    if not items:
        return ()
    resolved = {item_id(item): dict(axes(item)) for item in items}
    selected: set[str] = set()
    axis_names = sorted({name for value in resolved.values() for name in value})
    for name in axis_names:
        ordered = sorted(
            items,
            key=lambda item: (
                resolved[item_id(item)].get(name, 0),
                item_id(item),
            ),
        )
        selected.update((item_id(ordered[0]), item_id(ordered[-1])))
    if not selected:
        selected.add(item_id(items[0]))
    return tuple(
        sorted(
            (item for item in items if item_id(item) in selected),
            key=lambda item: (
                -_axis_volume(resolved[item_id(item)]),
                item_id(item),
            ),
        )
    )


def _axis_volume(axes: Mapping[str, int]) -> int:
    volume = 1
    for value in axes.values():
        volume *= max(1, value)
    return volume


__all__ = [
    "BatchGPUQualificationGate",
    "BatchGPUQualificationReceipt",
    "BatchGPUQualificationStage",
    "LargeBatchGPUTask",
    "QualificationArtifact",
    "qualification_artifact",
    "qualification_gate_path",
    "qualification_parent_stage",
    "require_isolated_qualification_root",
    "select_risk_first_axis_extrema",
    "verify_qualification_artifact",
]
