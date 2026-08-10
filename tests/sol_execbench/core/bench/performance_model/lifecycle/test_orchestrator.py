from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from sol_execbench.core.bench.performance_model.case_reuse import (
    AcceptancePreconditionError,
)
from sol_execbench.core.bench.performance_model.lifecycle import (
    CHAIN,
    BlobStore,
    DiagnosticCorpusSnapshotManifest,
    DiagnosticLifecycleArtifact,
    DiagnosticLifecycleParent,
    DiagnosticLifecyclePlan,
    DiagnosticLifecycleStage,
    DiagnosticRunManifest,
    DiagnosticStageAttempt,
    DiagnosticStageStatus,
    StageCompletion,
    StageRunContext,
    collection_run_id,
    diagnostic_lifecycle_status,
    orchestrator as lifecycle_orchestrator,
    resume_diagnostic_lifecycle,
    run_diagnostic_lifecycle,
    run_state_path,
    stage_attempt_path,
    stage_receipt_path,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
    DiagnosticRetentionClass,
)
from sol_execbench.core.bench.performance_model.lifecycle.models import (
    DiagnosticDesignManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.receipts import (
    DiagnosticStageReceipt,
)
from sol_execbench.core.bench.performance_model.lifecycle.run_state import (
    diagnostic_lifecycle_plan_payload,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import (
    designs_dir,
    orchestrations_dir,
    snapshots_dir,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file, stable_json_checksum

_NOW = "2026-01-01T00:00:00+00:00"


def _design(tmp_path: Path) -> Path:
    design = DiagnosticDesignManifest(
        stage=DiagnosticLifecycleStage.DESIGN,
        stage_id="a" * 64,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        purpose=DiagnosticEvidencePurpose.CONTROL_PLANE_CONFORMANCE,
        source_revision="test",
        created_at=_NOW,
        universe_start=0,
        design_payload_sha256="b" * 64,
    )
    path = designs_dir(tmp_path) / design.stage_id / "manifest.json"
    atomic_write_json_value(path, design.model_dump(mode="json"))
    BlobStore(tmp_path).put_file(path)
    return path


def _plan(design_path: Path, store_root: Path, max_attempts: int) -> Path:
    design = DiagnosticDesignManifest.model_validate_json(
        design_path.read_text(encoding="utf-8")
    )
    development = store_root / "development.json"
    development.write_text("{}", encoding="utf-8")
    development_digest = BlobStore(store_root).put_file(development)
    snapshot = DiagnosticCorpusSnapshotManifest(
        stage=DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
        stage_id="c" * 64,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        purpose=DiagnosticEvidencePurpose.CONTROL_PLANE_CONFORMANCE,
        source_revision="test",
        created_at=_NOW,
        role="development",
        corpus_file_sha256=development_digest,
        case_count=220,
        source_snapshot_ids=("d" * 64,),
    )
    snapshot_path = (
        snapshots_dir(store_root) / snapshot.stage_id / "manifest.json"
    )
    atomic_write_json_value(snapshot_path, snapshot.model_dump(mode="json"))
    BlobStore(store_root).put_file(snapshot_path)
    collection = store_root / "collection"
    collection.mkdir(exist_ok=True)
    held_out = collection / "held_out.json"
    held_out.write_text("{}", encoding="utf-8")
    held_out_artifact = DiagnosticLifecycleArtifact(
        relative_path="held_out.json",
        sha256=sha256_file(held_out),
        size_bytes=held_out.stat().st_size,
    )
    calibration = store_root / "calibration"
    calibration.mkdir(exist_ok=True)
    profile = calibration / "profile.json"
    audit = calibration / "profile.audit.json"
    profile.write_text("{}", encoding="utf-8")
    audit.write_text("{}", encoding="utf-8")
    run_id = collection_run_id(
        design_id=design.stage_id,
        generation=1,
        roles=("held_out",),
        frozen_held_out_sha256=held_out_artifact.sha256,
        source_revision="0" * 40,
        purpose=design.purpose,
    )
    values: dict[str, Any] = {
        "design": DiagnosticLifecycleParent(
            stage=DiagnosticLifecycleStage.DESIGN,
            purpose=design.purpose,
            stage_id=design.stage_id,
            sha256=sha256_file(design_path),
        ),
        "development_snapshot": DiagnosticLifecycleParent(
            stage=DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
            purpose=snapshot.purpose,
            stage_id=snapshot.stage_id,
            sha256=sha256_file(snapshot_path),
        ),
        "collection_root": str(collection),
        "collection_inventory": (held_out_artifact,),
        "collection_run_id": run_id,
        "generation": 1,
        "roles": ("held_out",),
        "calibration_profile_path": str(profile),
        "calibration_profile": DiagnosticLifecycleArtifact(
            relative_path=profile.name,
            sha256=sha256_file(profile),
            size_bytes=profile.stat().st_size,
        ),
        "calibration_audit_path": str(audit),
        "calibration_audit": DiagnosticLifecycleArtifact(
            relative_path=audit.name,
            sha256=sha256_file(audit),
            size_bytes=audit.stat().st_size,
        ),
        "held_out_corpus_path": str(held_out),
        "held_out_corpus": held_out_artifact,
        "output_root": str(store_root / "output"),
        "source_revision": "0" * 40,
        "purpose": design.purpose,
        "model_version": "test",
        "max_attempts": max_attempts,
    }
    provisional = DiagnosticLifecyclePlan.model_construct(
        plan_id="0" * 64, **values
    )
    plan = DiagnosticLifecyclePlan(
        plan_id=stable_json_checksum(
            diagnostic_lifecycle_plan_payload(provisional)
        ),
        **values,
    )
    path = store_root / "plan.json"
    atomic_write_json_value(path, plan.model_dump(mode="json"))
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
        verify_fail_first: int = 0,
    ) -> None:
        self.stage = stage
        self.order = order if order is not None else []
        self.fail_first = fail_first
        self.verify_result = verify_result
        self.verify_fail_first = verify_fail_first
        self.verify_calls = 0
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
        self.verify_calls += 1
        if self.verify_calls <= self.verify_fail_first:
            return False
        return self.verify_result


class PreconditionHandler(FakeHandler):
    """Acceptance handler that exposes one deterministic precondition."""

    def run(self, context: StageRunContext) -> StageCompletion:
        self.calls += 1
        self.order.append(self.stage)
        raise AcceptancePreconditionError(
            case_id="held_out-elementwise-01",
            workload_kind=WorkloadKind.ELEMENTWISE,
            reason_codes=("calibration_out_of_range:working_set_bytes",),
        )


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
    plan_path = _plan(design_path, store_root, max_attempts)
    return run_diagnostic_lifecycle(
        plan_path=plan_path,
        store_root_path=store_root,
        handlers=handlers,
        stages=stages,
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


def test_run_rejects_tampered_immutable_plan(tmp_path: Path) -> None:
    design_path = _design(tmp_path)
    plan_path = _plan(design_path, tmp_path, max_attempts=3)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["max_attempts"] = 4
    atomic_write_json_value(plan_path, payload)

    with pytest.raises(ValueError, match="plan_id is not canonical"):
        run_diagnostic_lifecycle(
            plan_path=plan_path,
            store_root_path=tmp_path,
            handlers=_handlers([]),
            now_fn=lambda: _NOW,
        )


def test_production_plan_requires_complete_pcie_identity(
    tmp_path: Path,
) -> None:
    design_path = _design(tmp_path)
    plan_path = _plan(design_path, tmp_path, max_attempts=3)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["purpose"] = DiagnosticEvidencePurpose.PRODUCTION.value
    payload["design"]["purpose"] = DiagnosticEvidencePurpose.PRODUCTION.value
    payload["development_snapshot"]["purpose"] = (
        DiagnosticEvidencePurpose.PRODUCTION.value
    )

    with pytest.raises(ValueError, match="requires a complete gpu_identity"):
        DiagnosticLifecyclePlan.model_validate(payload)


def test_conformance_plan_preserves_legacy_id_without_gpu_identity(
    tmp_path: Path,
) -> None:
    design_path = _design(tmp_path)
    plan_path = _plan(design_path, tmp_path, max_attempts=3)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload.pop("gpu_identity")
    identity_payload = {
        name: value for name, value in payload.items() if name != "plan_id"
    }
    payload["plan_id"] = stable_json_checksum(identity_payload)

    plan = DiagnosticLifecyclePlan.model_validate(payload)

    assert plan.gpu_identity is None
    assert plan.plan_id == stable_json_checksum(identity_payload)


def test_stage_manifest_preparation_runs_outside_registry_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design_path = _design(tmp_path)
    plan_path = _plan(design_path, tmp_path, max_attempts=3)
    plan = DiagnosticLifecyclePlan.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )
    context = lifecycle_orchestrator.build_run_context(
        plan=plan, store_root_path=tmp_path
    )
    output = tmp_path / "output.json"
    output.write_text("output", encoding="utf-8")
    completion = StageCompletion(
        stage_id="stage-id",
        outputs=(
            DiagnosticLifecycleArtifact(
                relative_path=output.name,
                sha256=sha256_file(output),
                size_bytes=output.stat().st_size,
            ),
        ),
        output_paths=(output,),
    )
    receipt = DiagnosticStageReceipt(
        stage=DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
        stage_id=completion.stage_id,
        command="test",
        started_at=_NOW,
        finished_at=_NOW,
        attempts=1,
    )

    def _prepare(*args: object) -> tuple[()]:
        del args
        BlobStore(tmp_path).put_bytes(b"nested CAS preparation")
        return ()

    monkeypatch.setattr(lifecycle_orchestrator, "_stage_manifests", _prepare)

    lifecycle_orchestrator._commit_stage_manifests(context, completion, receipt)


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
    attempts = [
        DiagnosticStageAttempt.model_validate_json(
            stage_attempt_path(
                run_state.collection_run_id,
                DiagnosticLifecycleStage.ACCEPTANCE,
                attempt,
                tmp_path,
            ).read_text(encoding="utf-8")
        )
        for attempt in range(1, 4)
    ]
    assert [item.failure_code for item in attempts] == [
        "stage_execution_error",
        "stage_execution_error",
        "stage_execution_error",
    ]
    assert all(len(item.detail) <= 4096 for item in attempts)


def test_acceptance_precondition_is_terminal_without_retry(
    tmp_path: Path,
) -> None:
    design_path = _design(tmp_path)
    order: list[DiagnosticLifecycleStage] = []
    handler = PreconditionHandler(
        DiagnosticLifecycleStage.ACCEPTANCE, order=order
    )
    handlers = _handlers(order, {DiagnosticLifecycleStage.ACCEPTANCE: handler})

    run_state = _run(design_path, tmp_path, handlers, max_attempts=3)

    acceptance = run_state.stage_state(DiagnosticLifecycleStage.ACCEPTANCE)
    assert acceptance is not None
    assert acceptance.status is DiagnosticStageStatus.FAILED
    assert acceptance.attempts == 1
    assert handler.calls == 1


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
    handlers = _handlers(order)
    run_state = _run(design_path, tmp_path, handlers)

    resumed_order: list[DiagnosticLifecycleStage] = []
    resumed_handlers = _handlers(
        resumed_order,
        {
            DiagnosticLifecycleStage.CORPUS_SNAPSHOT: FakeHandler(
                DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
                order=resumed_order,
                verify_fail_first=1,
            )
        },
    )
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


def test_resume_does_not_exceed_immutable_plan_budget(tmp_path: Path) -> None:
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
    assert model_build.status is DiagnosticStageStatus.FAILED
    assert model_build.attempts == 2
    assert fixed_order == []


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
        orchestrations_dir(tmp_path)
        / run_state.collection_run_id
        / "status.json"
    )
    assert status_file.is_file()


def test_status_reports_interrupted_running_stage_as_next(
    tmp_path: Path,
) -> None:
    design_path = _design(tmp_path)
    run_state = _run(design_path, tmp_path, _handlers([]))
    snapshot = run_state.stage_state(DiagnosticLifecycleStage.CORPUS_SNAPSHOT)
    assert snapshot is not None
    interrupted_snapshot = snapshot.model_copy(
        update={
            "status": DiagnosticStageStatus.RUNNING,
            "receipt_path": "",
            "outputs": (),
        }
    )
    interrupted = run_state.model_copy(
        update={
            "stages": (
                *run_state.stages[:3],
                interrupted_snapshot,
            )
        }
    )
    atomic_write_json_value(
        run_state_path(run_state.collection_run_id, tmp_path),
        interrupted.model_dump(mode="json"),
    )

    status = diagnostic_lifecycle_status(
        run_state_path=run_state_path(run_state.collection_run_id, tmp_path),
        handlers=_handlers([]),
    )

    assert (
        status["next_stage"] == DiagnosticLifecycleStage.CORPUS_SNAPSHOT.value
    )
    assert status["held_out_snapshot_id"] is None


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
    assert "parent_chain" not in status
    assert status["development_snapshot_id"] == "c" * 64
    assert isinstance(status["held_out_snapshot_id"], str)
    stages: object = status["stages"]
    assert isinstance(stages, list)
    for item in stages:
        if isinstance(item, dict):
            entry = cast(dict[str, object], item)
            if entry["status"] == DiagnosticStageStatus.VERIFIED.value:
                assert isinstance(entry["stage_id"], str)
