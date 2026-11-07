from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sol_execbench.core.bench.performance_model.lifecycle import planning
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
    DiagnosticLifecycleStage,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
)
from sol_execbench.core.bench.performance_model.lifecycle.models import (
    DiagnosticCorpusSnapshotManifest,
    DiagnosticDesignManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.planning import (
    LifecyclePlanInputs,
    author_lifecycle_plan,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleParent,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.platform.hardware import (
    HardwareExecutionIdentity,
    PCIeLinkIdentity,
    PCIeTopologyIdentity,
)
from sol_execbench.core.platform.source_state import GitSourceState

_CREATED_AT = "2026-08-09T00:00:00+00:00"
_DESIGN_ID = "a" * 64
_SNAPSHOT_ID = "b" * 64
_SOURCE_SNAPSHOT_ID = "c" * 64
_SOURCE_REVISION = "1" * 40


def _topology(width: int = 8) -> PCIeTopologyIdentity:
    link = PCIeLinkIdentity(
        bdf="0000:03:00.0",
        current_speed_gtps=32.0,
        max_speed_gtps=32.0,
        current_width=width,
        max_width=16,
    )
    return PCIeTopologyIdentity(
        links=(link,),
        bottleneck_bdf=link.bdf,
        effective_speed_gtps=link.current_speed_gtps,
        effective_width=link.current_width,
    )


def _gpu(width: int = 8) -> HardwareExecutionIdentity:
    return HardwareExecutionIdentity(
        gpu_architecture="gfx1200",
        gpu_id="a3ff7590-0000-1000-800f-a29c1cca1511",
        gpu_bdf="0000:03:00.0",
        pcie_topology=_topology(width),
        rocm_version="7.2.0",
        compiler_version="HIP version: 7.2.26015-fc0010cf6a",
        clock_mode="locked",
        power_profile="stable_peak",
    )


def _write_registry(store: Path) -> None:
    design = DiagnosticDesignManifest(
        stage=DiagnosticLifecycleStage.DESIGN,
        purpose=DiagnosticEvidencePurpose.PRODUCTION,
        stage_id=_DESIGN_ID,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        source_revision=_SOURCE_REVISION,
        created_at=_CREATED_AT,
        universe_start=160,
        design_payload_sha256="d" * 64,
    )
    snapshot = DiagnosticCorpusSnapshotManifest(
        stage=DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
        purpose=DiagnosticEvidencePurpose.PRODUCTION,
        stage_id=_SNAPSHOT_ID,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        source_revision=_SOURCE_REVISION,
        created_at=_CREATED_AT,
        role="development",
        corpus_file_sha256="e" * 64,
        case_count=880,
        source_snapshot_ids=(_SOURCE_SNAPSHOT_ID,),
        parents=(
            DiagnosticLifecycleParent(
                stage=DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
                purpose=DiagnosticEvidencePurpose.PRODUCTION,
                stage_id=_SOURCE_SNAPSHOT_ID,
                sha256="f" * 64,
            ),
        ),
    )
    atomic_write_json_value(
        store / "designs" / _DESIGN_ID / "manifest.json",
        design.model_dump(mode="json"),
    )
    atomic_write_json_value(
        store / "snapshots" / _SNAPSHOT_ID / "manifest.json",
        snapshot.model_dump(mode="json"),
    )


def _inputs(tmp_path: Path) -> LifecyclePlanInputs:
    collection = tmp_path / "collection"
    held_out = collection / "held_out.json"
    calibration_profile = tmp_path / "calibration-profile.json"
    calibration_audit = tmp_path / "calibration-audit.json"
    for path in (held_out, calibration_profile, calibration_audit):
        atomic_write_json_value(path, {})
    return LifecyclePlanInputs(
        design_id=_DESIGN_ID,
        development_snapshot_id=_SNAPSHOT_ID,
        collection_root=collection,
        held_out_corpus_path=held_out,
        calibration_profile_path=calibration_profile,
        calibration_audit_path=calibration_audit,
        output_root=tmp_path / "output",
        model_version="test-model-v1",
        max_attempts=2,
    )


def _mock_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planning,
        "capture_git_source_state",
        lambda *_args, **_kwargs: GitSourceState(
            revision=_SOURCE_REVISION,
            tracked_dirty=False,
            untracked_paths=(),
        ),
    )


def test_production_plan_binds_calibration_and_collection_pcie_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "store"
    _write_registry(store)
    inputs = _inputs(tmp_path)
    expected_gpu = _gpu()
    _mock_source(monkeypatch)
    monkeypatch.setattr(
        planning,
        "load_calibration_gpu_identity",
        lambda *_args, **_kwargs: expected_gpu,
    )
    monkeypatch.setattr(
        planning,
        "load_collection_gpu_identity",
        lambda *_args, **_kwargs: expected_gpu,
    )

    plan = author_lifecycle_plan(
        repository_root=tmp_path,
        store_root=store,
        inputs=inputs,
    )

    assert plan.gpu_identity == expected_gpu
    assert plan.gpu_identity is not None
    assert plan.gpu_identity.pcie_topology == _topology()


def test_production_plan_rejects_collection_pcie_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "store"
    _write_registry(store)
    inputs = _inputs(tmp_path)
    _mock_source(monkeypatch)
    monkeypatch.setattr(
        planning,
        "load_calibration_gpu_identity",
        lambda *_args, **_kwargs: _gpu(width=8),
    )
    monkeypatch.setattr(
        planning,
        "load_collection_gpu_identity",
        lambda *_args, **_kwargs: _gpu(width=4),
    )

    with pytest.raises(
        ValueError,
        match="collection/calibration GPU identity mismatch",
    ):
        author_lifecycle_plan(
            repository_root=tmp_path,
            store_root=store,
            inputs=inputs,
        )


def test_production_plan_rejects_reuse_fragment_from_other_design(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "store"
    _write_registry(store)
    inputs = _inputs(tmp_path)
    _mock_source(monkeypatch)
    monkeypatch.setattr(
        planning,
        "load_and_verify_case_reuse_bundle",
        lambda _path: SimpleNamespace(replacement_design_sha256="9" * 64),
    )

    with pytest.raises(
        ValueError, match="reuse fragment differs from lifecycle design"
    ):
        author_lifecycle_plan(
            repository_root=tmp_path,
            store_root=store,
            inputs=inputs,
        )


def test_capacity_governed_plan_binds_prefrozen_policy_and_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(tmp_path)
    policy_path = tmp_path / "vram-policy.json"
    inference_path = tmp_path / "inference.json"
    atomic_write_json_value(policy_path, {"policy": "frozen"})
    atomic_write_json_value(inference_path, {"inference": "frozen"})
    policy_digest = sha256_file(policy_path)
    design = DiagnosticDesignManifest(
        stage=DiagnosticLifecycleStage.DESIGN,
        purpose=DiagnosticEvidencePurpose.PRODUCTION,
        stage_id=_DESIGN_ID,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        source_revision=_SOURCE_REVISION,
        created_at=_CREATED_AT,
        universe_start=400,
        design_payload_sha256="d" * 64,
        vram_policy_sha256=policy_digest,
    )
    governed_inputs = replace(
        inputs,
        vram_policy_path=policy_path,
        frozen_inference_profile_path=inference_path,
    )

    def load_prefrozen(model, _path):
        if model.__name__ == "DiagnosticCalibrationProfile":
            return SimpleNamespace(probe_evidence_sha256=[policy_digest])
        if model.__name__ == "DiagnosticInferenceProfile":
            return SimpleNamespace(model_version=inputs.model_version)
        return SimpleNamespace()

    monkeypatch.setattr(planning, "load_json_file", load_prefrozen)

    policy, inference = planning._pre_frozen_inputs(design, governed_inputs)

    assert policy is not None
    assert policy.sha256 == policy_digest
    assert inference is not None
    assert inference.sha256 == sha256_file(inference_path)


def test_next_generation_counts_precollection_orchestration_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration = tmp_path / "orchestrations" / "failed" / "run.json"
    atomic_write_json_value(orchestration, {})
    monkeypatch.setattr(planning, "runs_dir", lambda _root: tmp_path / "runs")
    monkeypatch.setattr(
        planning,
        "orchestrations_dir",
        lambda _root: tmp_path / "orchestrations",
    )
    monkeypatch.setattr(
        planning,
        "load_json_file",
        lambda model, _path: (
            SimpleNamespace(
                design_id=_DESIGN_ID,
                generation=3,
            )
            if model.__name__ == "DiagnosticRunManifest"
            else pytest.fail(f"unexpected model {model}")
        ),
    )

    assert planning._next_generation(tmp_path, _DESIGN_ID) == 4
