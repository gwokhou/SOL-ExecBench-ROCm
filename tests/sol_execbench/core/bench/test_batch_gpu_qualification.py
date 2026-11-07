"""Contracts shared by every large batch GPU qualification workflow."""

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.batch_gpu_qualification import (
    BatchGPUQualificationGate,
    BatchGPUQualificationReceipt,
    BatchGPUQualificationStage,
    LargeBatchGPUTask,
    QualificationArtifact,
    require_isolated_qualification_root,
    select_risk_first_axis_extrema,
)
from sol_execbench.core.control_plane_schema_versions import (
    BatchGPUQualificationArtifactKind,
    ExecutionControlSchema,
)

_DIGEST = "a" * 64


def _receipt(
    stage: BatchGPUQualificationStage,
    item_ids: tuple[str, ...],
) -> BatchGPUQualificationReceipt:
    return BatchGPUQualificationReceipt(
        stage=stage,
        partition="partition",
        item_ids=item_ids,
        input_sha256=_DIGEST,
        artifacts=(
            QualificationArtifact(
                path="evidence.json",
                sha256=_DIGEST,
                size_bytes=1,
            ),
        ),
    )


def test_gate_requires_the_uniform_parent_chain() -> None:
    with pytest.raises(ValidationError, match="requires a parent"):
        BatchGPUQualificationGate(
            task=LargeBatchGPUTask.RELEASE_EVALUATION,
            stage=BatchGPUQualificationStage.CANARY,
            scope_id="release",
            subject_sha256=_DIGEST,
            runner_sha256=_DIGEST,
            configuration_sha256=_DIGEST,
            source_revision="revision",
            item_ids=("item",),
            receipts=(_receipt(BatchGPUQualificationStage.CANARY, ("item",)),),
            created_at="2026-08-09T00:00:00Z",
        )


def test_gate_rejects_incomplete_receipt_coverage() -> None:
    with pytest.raises(ValidationError, match="do not cover"):
        BatchGPUQualificationGate(
            task=LargeBatchGPUTask.SOLAR_RELEASE_BUILD,
            stage=BatchGPUQualificationStage.FULL,
            scope_id="solar",
            subject_sha256=_DIGEST,
            runner_sha256=_DIGEST,
            configuration_sha256=_DIGEST,
            source_revision="revision",
            parent_gate_sha256=_DIGEST,
            item_ids=("one", "two"),
            receipts=(_receipt(BatchGPUQualificationStage.FULL, ("one",)),),
            created_at="2026-08-09T00:00:00Z",
        )


def test_gate_and_receipt_share_one_discriminated_contract() -> None:
    receipt = _receipt(BatchGPUQualificationStage.STATIC, ("item",))
    gate = BatchGPUQualificationGate(
        task=LargeBatchGPUTask.RELEASE_EVALUATION,
        stage=BatchGPUQualificationStage.STATIC,
        scope_id="release",
        subject_sha256=_DIGEST,
        runner_sha256=_DIGEST,
        configuration_sha256=_DIGEST,
        source_revision="revision",
        item_ids=("item",),
        receipts=(receipt,),
        created_at="2026-08-09T00:00:00Z",
    )

    assert gate.schema_version == ExecutionControlSchema.BATCH_GPU_QUALIFICATION
    assert gate.artifact_kind == BatchGPUQualificationArtifactKind.GATE
    assert receipt.schema_version == gate.schema_version
    assert receipt.artifact_kind == BatchGPUQualificationArtifactKind.RECEIPT


def test_qualification_root_must_be_isolated(tmp_path) -> None:
    output = tmp_path / "formal"
    with pytest.raises(ValueError, match="inside batch output"):
        require_isolated_qualification_root(
            output / "qualification",
            output,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Shape:
    name: str
    axes: dict[str, int]


def test_risk_first_selection_covers_each_axis_extreme() -> None:
    shapes = (
        _Shape(name="small", axes={"M": 1, "N": 8}),
        _Shape(name="wide", axes={"M": 2, "N": 64}),
        _Shape(name="tall", axes={"M": 32, "N": 4}),
        _Shape(name="middle", axes={"M": 8, "N": 16}),
    )

    selected = select_risk_first_axis_extrema(
        shapes,
        item_id=lambda item: item.name,
        axes=lambda item: item.axes,
    )

    assert {item.name for item in selected} == {"small", "wide", "tall"}
    assert selected[0].name in {"wide", "tall"}
