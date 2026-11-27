# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Persistence and verification for manifests, receipts, and run state."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from pydantic import TypeAdapter

from sol_execbench.core.bench.performance_model.lifecycle.blob_store import (
    BlobStore,
)
from sol_execbench.core.bench.performance_model.lifecycle.calibration_identity import (
    load_calibration_gpu_identity,
)
from sol_execbench.core.bench.performance_model.lifecycle.corpus_registry import (
    snapshot_blob_inventory,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
    DiagnosticLifecycleStage,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
)
from sol_execbench.core.bench.performance_model.lifecycle.execution import (
    StageCompletion,
    StageRunContext,
)
from sol_execbench.core.bench.performance_model.lifecycle.identity import (
    recompute_stage_id,
)
from sol_execbench.core.bench.performance_model.lifecycle.models import (
    DIAGNOSTIC_LIFECYCLE_MANIFEST_ADAPTER,
    PRODUCER_VERSION,
    DiagnosticAcceptanceLifecycleManifest,
    DiagnosticCalibrationLifecycleManifest,
    DiagnosticCollectionRunManifest,
    DiagnosticCorpusSnapshotManifest,
    DiagnosticDesignManifest,
    DiagnosticLifecycleManifest,
    DiagnosticModelBuildManifest,
    DiagnosticPublicationLifecycleManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.receipts import (
    DiagnosticStageReceipt,
)
from sol_execbench.core.bench.performance_model.lifecycle.run_state import (
    DiagnosticLifecyclePlan,
    DiagnosticRunManifest,
    DiagnosticRunStageState,
    DiagnosticStageAttempt,
    lifecycle_plan_path,
    run_state_path,
    stage_attempt_path,
    stage_receipt_path,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
    DiagnosticLifecycleParent,
    SoftwareLifecycleIdentity,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import (
    acceptances_dir,
    builds_dir,
    calibrations_dir,
    designs_dir,
    orchestrations_dir,
    publication_registry_dir,
    runs_dir,
    snapshots_dir,
    store_lock_path,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.platform.hardware import HardwareExecutionIdentity
from sol_execbench.core.process import exclusive_file_lock

_RECEIPT_ADAPTER = TypeAdapter(DiagnosticStageReceipt)


def _required(path: Path | None, message: str) -> Path:
    if path is None:
        raise ValueError(message)
    return Path(path)


def _stage_manifest_path(
    root: Path, stage: DiagnosticLifecycleStage, stage_id: str
) -> Path:
    directories = {
        DiagnosticLifecycleStage.DESIGN: designs_dir(root),
        DiagnosticLifecycleStage.CALIBRATION: calibrations_dir(root),
        DiagnosticLifecycleStage.COLLECTION_RUN: runs_dir(root),
        DiagnosticLifecycleStage.CORPUS_SNAPSHOT: snapshots_dir(root),
        DiagnosticLifecycleStage.MODEL_BUILD: builds_dir(root),
        DiagnosticLifecycleStage.ACCEPTANCE: acceptances_dir(root),
        DiagnosticLifecycleStage.PUBLICATION: publication_registry_dir(root),
    }
    try:
        directory = directories[stage]
    except KeyError as error:
        raise ValueError(f"no manifest registry for {stage.value}") from error
    return directory / stage_id / "manifest.json"


def _recorded_parents(
    context: StageRunContext,
    state: DiagnosticRunStageState,
) -> list[dict[str, str]]:
    if state.status is not DiagnosticStageStatus.VERIFIED:
        return []
    receipt = _load_receipt(
        context.collection_run_id, state.stage, context.store_root
    )
    if receipt is None:
        return []
    return [
        {
            "stage": parent.stage.value,
            "stage_id": parent.stage_id,
            "sha256": parent.sha256,
        }
        for parent in receipt.input_identities
    ]


def _verified_stage_id(
    context: StageRunContext,
    run_state: DiagnosticRunManifest,
    stage: DiagnosticLifecycleStage,
) -> str | None:
    state = run_state.stage_state(stage)
    return _produced_stage_id(context, state) if state is not None else None


def _produced_stage_id(
    context: StageRunContext,
    state: DiagnosticRunStageState,
) -> str | None:
    if state.status is not DiagnosticStageStatus.VERIFIED:
        return None
    receipt = _load_receipt(
        context.collection_run_id,
        state.stage,
        context.store_root,
    )
    return receipt.stage_id if receipt is not None else None


def _write_status_json(
    context: StageRunContext,
    status: dict[str, object],
) -> None:
    path = (
        orchestrations_dir(context.store_root)
        / context.collection_run_id
        / "status.json"
    )
    atomic_write_json_value(path, status)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_design(context: StageRunContext) -> DiagnosticDesignManifest:
    return load_json_file(
        DiagnosticDesignManifest,
        context.design_manifest_path,
    )


def _load_design_or_none(
    context: StageRunContext,
) -> DiagnosticDesignManifest | None:
    try:
        return _load_design(context)
    except (OSError, ValueError):
        return None


def _initial_run_state(
    design: DiagnosticDesignManifest,
    context: StageRunContext,
    created_at: str,
    plan_sha256: str,
) -> DiagnosticRunManifest:
    return DiagnosticRunManifest(
        run_id=context.collection_run_id,
        collection_run_id=context.collection_run_id,
        design_id=design.stage_id,
        generation=context.generation,
        purpose=context.purpose,
        created_at=created_at,
        updated_at=created_at,
        plan_id=context.plan.plan_id,
        plan_sha256=plan_sha256,
    )


def _persist_plan(context: StageRunContext) -> str:
    """Write the canonical reviewed plan once and import it into CAS."""
    path = lifecycle_plan_path(context.collection_run_id, context.store_root)
    if path.is_file():
        existing = load_json_file(DiagnosticLifecyclePlan, path)
        if existing != context.plan:
            raise ValueError(
                "immutable lifecycle plan differs from stored plan"
            )
    else:
        with exclusive_file_lock(store_lock_path(context.store_root)):
            atomic_write_json_value(path, context.plan.model_dump(mode="json"))
    return BlobStore(context.store_root).put_file(path)


class _ManifestCommon(TypedDict):
    purpose: DiagnosticEvidencePurpose
    stage_id: str
    status: DiagnosticStageStatus
    retention_class: DiagnosticRetentionClass
    source_revision: str
    parents: tuple[DiagnosticLifecycleParent, ...]
    exact_inventory: tuple[DiagnosticLifecycleArtifact, ...]
    receipt: DiagnosticStageReceipt
    created_at: str


def _manifest_common(
    context: StageRunContext,
    receipt: DiagnosticStageReceipt,
    retention: DiagnosticRetentionClass,
) -> _ManifestCommon:
    return {
        "purpose": context.purpose,
        "stage_id": receipt.stage_id,
        "status": DiagnosticStageStatus.VERIFIED,
        "retention_class": retention,
        "source_revision": context.source_revision,
        "parents": receipt.input_identities,
        "exact_inventory": receipt.output_inventory,
        "receipt": receipt,
        "created_at": receipt.finished_at,
    }


def _commit_stage_manifests(
    context: StageRunContext,
    completion: StageCompletion,
    receipt: DiagnosticStageReceipt,
) -> None:
    """Materialize immutable stage objects only after verified receipt commit."""
    if not completion.output_paths:
        return
    paths: list[Path] = []
    manifests = _stage_manifests(context, completion, receipt)
    with exclusive_file_lock(store_lock_path(context.store_root)):
        for manifest in manifests:
            recomputed = recompute_stage_id(manifest)
            if recomputed != manifest.stage_id:
                raise ValueError(
                    f"lifecycle stage identity is not canonical: "
                    f"{manifest.stage.value}"
                )
            path = _stage_manifest_path(
                context.store_root, manifest.stage, manifest.stage_id
            )
            paths.append(path)
            if path.is_file():
                existing = DIAGNOSTIC_LIFECYCLE_MANIFEST_ADAPTER.validate_json(
                    path.read_text(encoding="utf-8")
                )
                if existing != manifest:
                    raise ValueError(
                        f"immutable lifecycle object differs: {path}"
                    )
                continue
            atomic_write_json_value(path, manifest.model_dump(mode="json"))
    for path in paths:
        BlobStore(context.store_root).put_file(path)


def _stage_manifests(
    context: StageRunContext,
    completion: StageCompletion,
    receipt: DiagnosticStageReceipt,
) -> tuple[DiagnosticLifecycleManifest, ...]:
    stage = receipt.stage
    if stage is DiagnosticLifecycleStage.DESIGN:
        design = _load_design(context)
        return (design.model_copy(update={"receipt": receipt}),)
    if stage is DiagnosticLifecycleStage.CALIBRATION:
        return (_calibration_manifest(context, receipt),)
    if stage is DiagnosticLifecycleStage.COLLECTION_RUN:
        return (_collection_manifest(context, receipt),)
    if stage is DiagnosticLifecycleStage.CORPUS_SNAPSHOT:
        return _snapshot_manifests(context, completion, receipt)
    if stage is DiagnosticLifecycleStage.MODEL_BUILD:
        return (_model_manifest(context, completion, receipt),)
    if stage is DiagnosticLifecycleStage.ACCEPTANCE:
        return (_acceptance_manifest(context, completion, receipt),)
    if stage is DiagnosticLifecycleStage.PUBLICATION:
        return (_publication_manifest(context, completion, receipt),)
    if stage is DiagnosticLifecycleStage.RELEASE:
        return ()
    raise ValueError(f"unsupported lifecycle manifest stage: {stage.value}")


def _calibration_manifest(
    context: StageRunContext, receipt: DiagnosticStageReceipt
) -> DiagnosticCalibrationLifecycleManifest:
    profile = _required(context.calibration_profile_path, "missing calibration")
    audit = _required(
        context.calibration_audit_path, "missing calibration audit"
    )
    gpu, software = _calibration_identities(
        profile,
        audit,
        purpose=context.purpose,
    )
    return DiagnosticCalibrationLifecycleManifest(
        **_manifest_common(
            context, receipt, DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE
        ),
        stage=DiagnosticLifecycleStage.CALIBRATION,
        gpu_identity=gpu,
        software_identity=software,
        calibration_profile_sha256=sha256_file(profile),
        calibration_audit_sha256=sha256_file(audit),
    )


def _collection_manifest(
    context: StageRunContext, receipt: DiagnosticStageReceipt
) -> DiagnosticCollectionRunManifest:
    predecessor = _latest_collection_run(context)
    return DiagnosticCollectionRunManifest(
        **_manifest_common(
            context, receipt, DiagnosticRetentionClass.PROCESS_EVIDENCE
        ),
        stage=DiagnosticLifecycleStage.COLLECTION_RUN,
        gpu_identity=context.plan.gpu_identity,
        generation=context.generation,
        roles=("held_out",),
        frozen_held_out_sha256=context.plan.held_out_corpus.sha256,
        supersedes=predecessor.stage_id if predecessor is not None else None,
    )


def _latest_collection_run(
    context: StageRunContext,
) -> DiagnosticCollectionRunManifest | None:
    """Return the immediate immutable predecessor and validate generation."""
    candidates: list[DiagnosticCollectionRunManifest] = []
    for path in sorted(runs_dir(context.store_root).glob("*/manifest.json")):
        manifest = load_json_file(DiagnosticCollectionRunManifest, path)
        if manifest.purpose is not context.purpose:
            continue
        if any(
            parent.stage_id == context.plan.design.stage_id
            for parent in manifest.parents
        ):
            candidates.append(manifest)
    predecessor = max(
        candidates, key=lambda item: item.generation, default=None
    )
    prior_generations = [item.generation for item in candidates]
    for path in sorted(
        orchestrations_dir(context.store_root).glob("*/run.json")
    ):
        run = load_json_file(DiagnosticRunManifest, path)
        if (
            run.design_id == context.plan.design.stage_id
            and run.run_id != context.collection_run_id
        ):
            prior_generations.append(run.generation)
    expected = max(prior_generations, default=0) + 1
    if context.generation != expected:
        raise ValueError("collection generation changed since plan authoring")
    return predecessor


def _snapshot_manifests(
    context: StageRunContext,
    completion: StageCompletion,
    receipt: DiagnosticStageReceipt,
) -> tuple[DiagnosticCorpusSnapshotManifest, ...]:
    from sol_execbench.core.bench.performance_model.validation_corpus import (
        DiagnosticValidationCorpus,
    )

    artifact = completion.outputs[0]
    corpus_path = completion.output_paths[0]
    corpus = load_json_file(DiagnosticValidationCorpus, corpus_path)
    common = _manifest_common(
        context,
        receipt,
        DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
    )
    common["exact_inventory"] = snapshot_blob_inventory(
        corpus_path,
        corpus,
        store=BlobStore(context.store_root),
    )
    return (
        DiagnosticCorpusSnapshotManifest(
            **common,
            stage=DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
            role="held_out",
            corpus_file_sha256=artifact.sha256,
            case_count=len(corpus.cases),
        ),
    )


def _model_manifest(
    context: StageRunContext,
    completion: StageCompletion,
    receipt: DiagnosticStageReceipt,
) -> DiagnosticModelBuildManifest:
    calibration = _required(
        context.calibration_profile_path, "missing calibration"
    )
    audit = _required(
        context.calibration_audit_path, "missing calibration audit"
    )
    inference = completion.output_paths[0]
    return DiagnosticModelBuildManifest(
        **_manifest_common(
            context, receipt, DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE
        ),
        stage=DiagnosticLifecycleStage.MODEL_BUILD,
        calibration_profile_sha256=sha256_file(calibration),
        calibration_audit_sha256=sha256_file(audit),
        inference_profile_sha256=sha256_file(inference),
        model_version=context.model_version,
    )


def _acceptance_manifest(
    context: StageRunContext,
    completion: StageCompletion,
    receipt: DiagnosticStageReceipt,
) -> DiagnosticAcceptanceLifecycleManifest:
    from sol_execbench.core.bench.performance_model.acceptance import (
        DiagnosticAcceptanceResult,
    )

    result_path = completion.output_paths[1]
    result = load_json_file(DiagnosticAcceptanceResult, result_path)
    held_out_id = _prior_receipt_stage_id(
        context, DiagnosticLifecycleStage.CORPUS_SNAPSHOT
    )
    return DiagnosticAcceptanceLifecycleManifest(
        **_manifest_common(
            context, receipt, DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE
        ),
        stage=DiagnosticLifecycleStage.ACCEPTANCE,
        held_out_corpus_snapshot_id=held_out_id,
        accepted=result.accepted,
        verdict_sha256=sha256_file(result_path),
    )


def _publication_manifest(
    context: StageRunContext,
    completion: StageCompletion,
    receipt: DiagnosticStageReceipt,
) -> DiagnosticPublicationLifecycleManifest:
    from sol_execbench.core.bench.performance_model.publication import (
        DiagnosticPublicationProjection,
    )

    manifest_path = completion.output_paths[0]
    projection = load_json_file(DiagnosticPublicationProjection, manifest_path)
    return DiagnosticPublicationLifecycleManifest(
        **_manifest_common(
            context, receipt, DiagnosticRetentionClass.PUBLICATION_RELEASE
        ),
        stage=DiagnosticLifecycleStage.PUBLICATION,
        source_corpus_sha256=projection.source_corpus_sha256,
        publication_manifest_sha256=sha256_file(manifest_path),
        uncompressed_size_bytes=projection.uncompressed_size_bytes,
        case_count=projection.case_count,
    )


def _calibration_identities(
    profile_path: Path,
    audit_path: Path,
    *,
    purpose: DiagnosticEvidencePurpose,
) -> tuple[HardwareExecutionIdentity, SoftwareLifecycleIdentity]:
    gpu = load_calibration_gpu_identity(
        profile_path,
        audit_path,
        expected_purpose=purpose,
        require_pcie_topology=(purpose is DiagnosticEvidencePurpose.PRODUCTION),
    )
    software = SoftwareLifecycleIdentity(
        sol_version=PRODUCER_VERSION,
        python_version=sys.version.split()[0],
    )
    return gpu, software


def _replace_stage(
    run_state: DiagnosticRunManifest,
    state: DiagnosticRunStageState,
) -> DiagnosticRunManifest:
    updated = run_state.set_stage(state)
    return updated.model_copy(update={"updated_at": _now()})


def _write_run_state(
    context: StageRunContext,
    run_state: DiagnosticRunManifest,
) -> None:
    with exclusive_file_lock(store_lock_path(context.store_root)):
        atomic_write_json_value(
            run_state_path(context.collection_run_id, context.store_root),
            run_state.model_dump(mode="json"),
        )


def _write_receipt(
    context: StageRunContext,
    stage: DiagnosticLifecycleStage,
    receipt: DiagnosticStageReceipt,
) -> None:
    path = stage_receipt_path(
        context.collection_run_id,
        stage,
        context.store_root,
    )
    with exclusive_file_lock(store_lock_path(context.store_root)):
        atomic_write_json_value(
            path,
            _RECEIPT_ADAPTER.dump_python(receipt, mode="json"),
        )
    BlobStore(context.store_root).put_file(path)


def _write_attempt(
    context: StageRunContext,
    attempt: DiagnosticStageAttempt,
) -> None:
    path = stage_attempt_path(
        context.collection_run_id,
        attempt.stage,
        attempt.attempt,
        context.store_root,
    )
    if path.is_file():
        existing = load_json_file(DiagnosticStageAttempt, path)
        if existing != attempt:
            raise ValueError(f"append-only lifecycle attempt differs: {path}")
        return
    with exclusive_file_lock(store_lock_path(context.store_root)):
        if path.exists():
            raise ValueError(f"lifecycle attempt path is not a file: {path}")
        atomic_write_json_value(path, attempt.model_dump(mode="json"))
    BlobStore(context.store_root).put_file(path)


def _build_receipt(
    stage: DiagnosticLifecycleStage,
    completion: StageCompletion,
    context: StageRunContext,
    input_identities: tuple[DiagnosticLifecycleParent, ...],
    attempts: int,
    started_at: str,
    finished_at: str,
) -> DiagnosticStageReceipt:
    return DiagnosticStageReceipt(
        stage=stage,
        purpose=context.purpose,
        stage_id=completion.stage_id,
        command=f"diagnostics lifecycle {stage.value}",
        started_at=started_at,
        finished_at=finished_at,
        attempts=attempts,
        input_identities=input_identities,
        output_inventory=completion.outputs,
        verification="receipt_verified",
    )


def _load_receipt(
    collection_run_id: str,
    stage: DiagnosticLifecycleStage,
    root: Path,
) -> DiagnosticStageReceipt | None:
    path = stage_receipt_path(collection_run_id, stage, root)
    if not path.is_file():
        return None
    try:
        return _RECEIPT_ADAPTER.validate_json(
            path.read_text(encoding="utf-8"),
        )
    except (OSError, ValueError):
        return None


def _prior_receipt_stage_id(
    context: StageRunContext,
    stage: DiagnosticLifecycleStage,
) -> str:
    """Return the recorded stage_id of one already-verified predecessor."""
    receipt = _load_receipt(
        context.collection_run_id,
        stage,
        context.store_root,
    )
    if receipt is None:
        raise ValueError(
            f"{stage.value} receipt is missing; cannot derive downstream identity",
        )
    return receipt.stage_id
