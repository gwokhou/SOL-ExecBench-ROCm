from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from sol_execbench.core.bench.performance_model.lifecycle import (
    CHAIN,
    DiagnosticLifecycleStage,
    DiagnosticRunManifest,
    DiagnosticStageStatus,
    StageCompletion,
    StageRunContext,
    diagnostic_lifecycle_status,
    resume_diagnostic_lifecycle,
    run_diagnostic_lifecycle,
    run_state_path,
    stage_receipt_path,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticRetentionClass,
)
from sol_execbench.core.bench.performance_model.lifecycle.models import (
    DiagnosticDesignManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.receipts import (
    DiagnosticStageReceipt,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleParent,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import runs_dir
from sol_execbench.core.data.json_utils import atomic_write_json_value

_NOW = "2026-01-01T00:00:00+00:00"


def _design(tmp_path: Path) -> Path:
    design = DiagnosticDesignManifest(
        stage=DiagnosticLifecycleStage.DESIGN,
        stage_id="a" * 64,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        source_revision="test",
        created_at=_NOW,
        universe_start=0,
        design_payload_sha256="b" * 64,
    )
    path = tmp_path / "design.json"
    atomic_write_json_value(path, design.model_dump(mode="json"))
    return path


class FakeHandler:
    """Configurable handler that records calls for the state-machine tests."""

    def __init__(
        self,
        stage: DiagnosticLifecycleStage,
        *,
        order: list[DiagnosticLifecycleStage] | None = None,
        fail_first: int = 0,
        verify_result: bool = True,
    ) -> None:
        self.stage = stage
        self.order = order if order is not None else []
        self.fail_first = fail_first
        self.verify_result = verify_result
        self.calls = 0

    def run(self, context: StageRunContext) -> StageCompletion:
        self.calls += 1
        self.order.append(self.stage)
        if self.calls <= self.fail_first:
            raise ValueError(f"fake failure for {self.stage.value}")
        return StageCompletion(stage_id=f"{self.stage.value}-id", outputs=())

    def prepare(
        self,
        context: StageRunContext,
        run_state: DiagnosticRunManifest,
    ) -> tuple[DiagnosticLifecycleParent, ...]:
        del context, run_state
        return ()

    def verify(
        self,
        context: StageRunContext,
        receipt: DiagnosticStageReceipt,
    ) -> bool:
        return self.verify_result


def _handlers(
    order: list[DiagnosticLifecycleStage],
    overrides: dict[DiagnosticLifecycleStage, FakeHandler] | None = None,
) -> dict[DiagnosticLifecycleStage, FakeHandler]:
    handlers: dict[DiagnosticLifecycleStage, FakeHandler] = {
        stage: FakeHandler(stage, order=order) for stage in CHAIN
    }
    if overrides:
        handlers.update(overrides)
    return handlers


def _run(
    design_path: Path,
    store_root: Path,
    handlers: dict[DiagnosticLifecycleStage, FakeHandler],
    *,
    stages: Sequence[DiagnosticLifecycleStage] | None = None,
    max_attempts: int = 3,
) -> DiagnosticRunManifest:
    return run_diagnostic_lifecycle(
        design_manifest_path=design_path,
        store_root_path=store_root,
        handlers=handlers,
        stages=stages,
        max_attempts=max_attempts,
        now_fn=lambda: _NOW,
    )


def test_run_executes_chain_in_order_and_persists(
    tmp_path: Path,
) -> None:
    design_path = _design(tmp_path)
    order: list[DiagnosticLifecycleStage] = []
    handlers = _handlers(order)
    run_state = _run(design_path, tmp_path, handlers)

    assert [item.stage for item in run_state.stages] == list(CHAIN)
    assert all(
        item.status is DiagnosticStageStatus.VERIFIED
        for item in run_state.stages
    )
    assert order == list(CHAIN)
    assert run_state.run_id == run_state.collection_run_id
    assert run_state.design_id == "a" * 64
    state_file = run_state_path(run_state.collection_run_id, tmp_path)
    assert state_file.is_file()
    for stage in CHAIN:
        receipt = stage_receipt_path(
            run_state.collection_run_id,
            stage,
            tmp_path,
        )
        assert receipt.is_file()


def test_run_illegal_transition_rejected(tmp_path: Path) -> None:
    design_path = _design(tmp_path)
    handlers = _handlers([])
    with pytest.raises(ValueError, match="illegal lifecycle transition"):
        _run(
            design_path,
            tmp_path,
            handlers,
            stages=[
                DiagnosticLifecycleStage.DESIGN,
                DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
            ],
        )


def test_run_requires_verified_predecessor(tmp_path: Path) -> None:
    design_path = _design(tmp_path)
    handlers = _handlers([])
    with pytest.raises(ValueError, match="requires verified"):
        _run(
            design_path,
            tmp_path,
            handlers,
            stages=[DiagnosticLifecycleStage.MODEL_BUILD],
        )


def test_bounded_retries_retry_within_budget(tmp_path: Path) -> None:
    design_path = _design(tmp_path)
    order: list[DiagnosticLifecycleStage] = []
    flaky = FakeHandler(
        DiagnosticLifecycleStage.MODEL_BUILD,
        order=order,
        fail_first=1,
    )
    handlers = _handlers(order, {DiagnosticLifecycleStage.MODEL_BUILD: flaky})
    run_state = _run(design_path, tmp_path, handlers)

    assert flaky.calls == 2
    model_build = run_state.stage_state(DiagnosticLifecycleStage.MODEL_BUILD)
    assert model_build is not None
    assert model_build.status is DiagnosticStageStatus.VERIFIED
    assert model_build.attempts == 2


def test_exhausted_attempts_mark_failed_and_stop(tmp_path: Path) -> None:
    design_path = _design(tmp_path)
    order: list[DiagnosticLifecycleStage] = []
    bad = FakeHandler(
        DiagnosticLifecycleStage.ACCEPTANCE,
        order=order,
        fail_first=10**6,
    )
    handlers = _handlers(order, {DiagnosticLifecycleStage.ACCEPTANCE: bad})
    run_state = _run(design_path, tmp_path, handlers, max_attempts=3)

    acceptance = run_state.stage_state(DiagnosticLifecycleStage.ACCEPTANCE)
    assert acceptance is not None
    assert acceptance.status is DiagnosticStageStatus.FAILED
    assert acceptance.attempts == 3
    assert bad.calls == 3
    assert run_state.stage_state(DiagnosticLifecycleStage.PUBLICATION) is None


def test_resume_reruns_stage_with_missing_receipt(tmp_path: Path) -> None:
    design_path = _design(tmp_path)
    order: list[DiagnosticLifecycleStage] = []
    handlers = _handlers(order)
    run_state = _run(design_path, tmp_path, handlers)

    model_build_receipt = stage_receipt_path(
        run_state.collection_run_id,
        DiagnosticLifecycleStage.MODEL_BUILD,
        tmp_path,
    )
    model_build_receipt.unlink()

    resumed_order: list[DiagnosticLifecycleStage] = []
    resumed_handlers = _handlers(resumed_order)
    resumed = resume_diagnostic_lifecycle(
        run_state_path=run_state_path(run_state.collection_run_id, tmp_path),
        handlers=resumed_handlers,
        now_fn=lambda: _NOW,
    )

    model_build = resumed.stage_state(DiagnosticLifecycleStage.MODEL_BUILD)
    assert model_build is not None
    assert model_build.status is DiagnosticStageStatus.VERIFIED
    assert model_build_receipt.is_file()
    # The drifted stage is re-run; later stages stay verified.
    assert DiagnosticLifecycleStage.MODEL_BUILD in resumed_order
    release = resumed.stage_state(DiagnosticLifecycleStage.RELEASE)
    assert release is not None
    assert release.status is DiagnosticStageStatus.VERIFIED


def test_resume_reruns_drifted_stage(tmp_path: Path) -> None:
    design_path = _design(tmp_path)
    order: list[DiagnosticLifecycleStage] = []
    drift = FakeHandler(
        DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
        order=order,
        verify_result=False,
    )
    handlers = _handlers(
        order,
        {DiagnosticLifecycleStage.CORPUS_SNAPSHOT: drift},
    )
    run_state = _run(design_path, tmp_path, handlers)

    resumed_order: list[DiagnosticLifecycleStage] = []
    resumed_handlers = _handlers(resumed_order)
    resumed = resume_diagnostic_lifecycle(
        run_state_path=run_state_path(run_state.collection_run_id, tmp_path),
        handlers=resumed_handlers,
        now_fn=lambda: _NOW,
    )

    snapshot = resumed.stage_state(DiagnosticLifecycleStage.CORPUS_SNAPSHOT)
    assert snapshot is not None
    assert snapshot.status is DiagnosticStageStatus.VERIFIED
    model_build = resumed.stage_state(DiagnosticLifecycleStage.MODEL_BUILD)
    assert model_build is not None
    assert model_build.status is DiagnosticStageStatus.VERIFIED


def test_interrupted_run_resumes_from_first_incomplete(tmp_path: Path) -> None:
    design_path = _design(tmp_path)
    order: list[DiagnosticLifecycleStage] = []
    failing = FakeHandler(
        DiagnosticLifecycleStage.MODEL_BUILD,
        order=order,
        fail_first=10**6,
    )
    handlers = _handlers(
        order,
        {DiagnosticLifecycleStage.MODEL_BUILD: failing},
    )
    run_state = _run(design_path, tmp_path, handlers, max_attempts=2)
    interrupted = run_state.stage_state(DiagnosticLifecycleStage.MODEL_BUILD)
    assert interrupted is not None
    assert interrupted.status is DiagnosticStageStatus.FAILED

    fixed_order: list[DiagnosticLifecycleStage] = []
    resumed_handlers = _handlers(fixed_order)
    resumed = resume_diagnostic_lifecycle(
        run_state_path=run_state_path(run_state.collection_run_id, tmp_path),
        handlers=resumed_handlers,
        now_fn=lambda: _NOW,
    )

    model_build = resumed.stage_state(DiagnosticLifecycleStage.MODEL_BUILD)
    assert model_build is not None
    assert model_build.status is DiagnosticStageStatus.VERIFIED
    release = resumed.stage_state(DiagnosticLifecycleStage.RELEASE)
    assert release is not None
    assert release.status is DiagnosticStageStatus.VERIFIED
    assert DiagnosticLifecycleStage.MODEL_BUILD in fixed_order


def test_status_reports_drift_and_next_stage(tmp_path: Path) -> None:
    design_path = _design(tmp_path)
    order: list[DiagnosticLifecycleStage] = []
    handlers = _handlers(order)
    run_state = _run(design_path, tmp_path, handlers)

    drift = FakeHandler(
        DiagnosticLifecycleStage.ACCEPTANCE,
        order=[],
        verify_result=False,
    )
    drifted_handlers = _handlers(
        [],
        {DiagnosticLifecycleStage.ACCEPTANCE: drift},
    )
    status = diagnostic_lifecycle_status(
        run_state_path=run_state_path(run_state.collection_run_id, tmp_path),
        handlers=drifted_handlers,
    )

    assert status["next_stage"] == DiagnosticLifecycleStage.ACCEPTANCE.value
    stages: object = status["stages"]
    assert isinstance(stages, list)
    by_stage: dict[str, str] = {}
    for item in stages:
        if isinstance(item, dict):
            entry = cast(dict[str, object], item)
            by_stage[str(entry["stage"])] = str(entry["status"])
    assert by_stage["acceptance"] == DiagnosticStageStatus.FAILED.value
    assert by_stage["design"] == DiagnosticStageStatus.VERIFIED.value
    status_file = (
        runs_dir(tmp_path) / run_state.collection_run_id / "status.json"
    )
    assert status_file.is_file()


def test_status_next_stage_none_when_complete(tmp_path: Path) -> None:
    design_path = _design(tmp_path)
    order: list[DiagnosticLifecycleStage] = []
    handlers = _handlers(order)
    run_state = _run(design_path, tmp_path, handlers)

    status = diagnostic_lifecycle_status(
        run_state_path=run_state_path(run_state.collection_run_id, tmp_path),
        handlers=_handlers([]),
    )
    assert status["next_stage"] is None
    assert status["collection_run_id"] == run_state.collection_run_id
    chain: object = status["parent_chain"]
    assert isinstance(chain, list)
    assert len(chain) == len(CHAIN)
    stages: object = status["stages"]
    assert isinstance(stages, list)
    for item in stages:
        if isinstance(item, dict):
            entry = cast(dict[str, object], item)
            if entry["status"] == DiagnosticStageStatus.VERIFIED.value:
                assert isinstance(entry["stage_id"], str)
