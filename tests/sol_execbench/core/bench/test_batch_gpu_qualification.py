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


def test_qualification_root_must_be_isolated(tmp_path) -> None:
    output = tmp_path / "formal"
    with pytest.raises(ValueError, match="inside batch output"):
        require_isolated_qualification_root(
            output / "qualification",
            output,
        )


@dataclass(frozen=True)
class _Shape:
    name: str
    axes: dict[str, int]


def test_risk_first_selection_covers_each_axis_extreme() -> None:
    shapes = (
        _Shape("small", {"M": 1, "N": 8}),
        _Shape("wide", {"M": 2, "N": 64}),
        _Shape("tall", {"M": 32, "N": 4}),
        _Shape("middle", {"M": 8, "N": 16}),
    )

    selected = select_risk_first_axis_extrema(
        shapes,
        item_id=lambda item: item.name,
        axes=lambda item: item.axes,
    )

    assert {item.name for item in selected} == {"small", "wide", "tall"}
    assert selected[0].name in {"wide", "tall"}
