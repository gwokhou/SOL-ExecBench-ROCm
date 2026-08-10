# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Canonical authoring of immutable diagnostic lifecycle plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sol_execbench.core.bench.performance_model.case_reuse import (
    load_and_verify_case_reuse_bundle,
)
from sol_execbench.core.bench.performance_model.lifecycle.calibration_identity import (
    load_calibration_gpu_identity,
)
from sol_execbench.core.bench.performance_model.lifecycle.collection_identity import (
    load_collection_gpu_identity,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
    DiagnosticLifecycleStage,
)
from sol_execbench.core.bench.performance_model.lifecycle.identity import (
    collection_run_id,
)
from sol_execbench.core.bench.performance_model.lifecycle.inventory import (
    inventory_regular_tree,
)
from sol_execbench.core.bench.performance_model.lifecycle.models import (
    DiagnosticCollectionRunManifest,
    DiagnosticCorpusSnapshotManifest,
    DiagnosticDesignManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.run_state import (
    DiagnosticLifecyclePlan,
    diagnostic_lifecycle_plan_payload,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
    DiagnosticLifecycleParent,
    GpuLifecycleIdentity,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import (
    designs_dir,
    runs_dir,
    snapshots_dir,
)
from sol_execbench.core.bench.performance_model.models import (
    DiagnosticCalibrationProfile,
)
from sol_execbench.core.bench.performance_model.vram_policy import (
    DiagnosticVRAMWorkingSetPolicy,
)
from sol_execbench.core.data.json_utils import load_json_file
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.integrity.schema_versions import SchemaVersion
from sol_execbench.core.platform.source_state import (
    GitSourceState,
    capture_git_source_state,
)

_SOURCE_PATHS = ("src", "scripts", "pyproject.toml", "uv.lock")


@dataclass(frozen=True, slots=True)
class LifecyclePlanInputs:
    """Operator-selected inputs for one immutable lifecycle plan."""

    design_id: str
    development_snapshot_id: str
    collection_root: Path
    held_out_corpus_path: Path
    calibration_profile_path: Path
    calibration_audit_path: Path
    output_root: Path
    model_version: str
    max_attempts: int
    vram_policy_path: Path | None = None
    frozen_inference_profile_path: Path | None = None


def _artifact(path: Path, *, relative_path: str) -> DiagnosticLifecycleArtifact:
    provided = Path(path)
    if provided.is_symlink():
        raise ValueError(f"lifecycle plan input is not regular: {path}")
    resolved = provided.resolve()
    if not resolved.is_file():
        raise ValueError(f"lifecycle plan input is not regular: {path}")
    return DiagnosticLifecycleArtifact(
        relative_path=relative_path,
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _parent(
    stage: DiagnosticLifecycleStage,
    manifest: DiagnosticDesignManifest | DiagnosticCorpusSnapshotManifest,
    path: Path,
) -> DiagnosticLifecycleParent:
    return DiagnosticLifecycleParent(
        stage=stage,
        purpose=manifest.purpose,
        stage_id=manifest.stage_id,
        sha256=sha256_file(path),
    )


def _next_generation(root: Path, design_id: str) -> int:
    generations: list[int] = []
    for path in sorted(runs_dir(root).glob("*/manifest.json")):
        manifest = load_json_file(DiagnosticCollectionRunManifest, path)
        if any(parent.stage_id == design_id for parent in manifest.parents):
            generations.append(manifest.generation)
    return max(generations, default=0) + 1


def _load_registry_inputs(
    root: Path, inputs: LifecyclePlanInputs
) -> tuple[
    DiagnosticDesignManifest,
    Path,
    DiagnosticCorpusSnapshotManifest,
    Path,
]:
    design_path = designs_dir(root) / inputs.design_id / "manifest.json"
    snapshot_path = (
        snapshots_dir(root) / inputs.development_snapshot_id / "manifest.json"
    )
    design = load_json_file(DiagnosticDesignManifest, design_path)
    snapshot = load_json_file(DiagnosticCorpusSnapshotManifest, snapshot_path)
    if snapshot.role != "development" or not snapshot.source_snapshot_ids:
        raise ValueError(
            "lifecycle plan requires a promoted development snapshot"
        )
    if design.purpose is not snapshot.purpose:
        raise ValueError("design and development snapshot purpose mismatch")
    return design, design_path, snapshot, snapshot_path


def _collection_inputs(
    inputs: LifecyclePlanInputs,
    design: DiagnosticDesignManifest,
) -> tuple[
    Path, tuple[DiagnosticLifecycleArtifact, ...], DiagnosticLifecycleArtifact
]:
    collection = inputs.collection_root.resolve()
    output = inputs.output_root.resolve()
    if output.is_relative_to(collection) or collection.is_relative_to(output):
        raise ValueError(
            "lifecycle output and collection roots must be disjoint"
        )
    held_out = inputs.held_out_corpus_path.resolve()
    if not held_out.is_relative_to(collection):
        raise ValueError("held-out corpus must be beneath the collection root")
    inventory = inventory_regular_tree(collection)
    artifact = _artifact(
        inputs.held_out_corpus_path,
        relative_path=held_out.relative_to(collection).as_posix(),
    )
    if artifact not in inventory:
        raise ValueError("held-out corpus is absent from collection inventory")
    reuse = load_and_verify_case_reuse_bundle(inputs.held_out_corpus_path)
    if (
        reuse is not None
        and reuse.replacement_design_sha256 != design.design_payload_sha256
    ):
        raise ValueError("reuse fragment differs from lifecycle design")
    return collection, inventory, artifact


def _finalize_plan(
    provisional: DiagnosticLifecyclePlan,
) -> DiagnosticLifecyclePlan:
    payload = diagnostic_lifecycle_plan_payload(provisional)
    return DiagnosticLifecyclePlan.model_validate(
        {"plan_id": stable_json_checksum(payload), **payload}
    )


def _plan_gpu_identity(
    inputs: LifecyclePlanInputs,
    design: DiagnosticDesignManifest,
    collection: Path,
) -> GpuLifecycleIdentity:
    gpu_identity = load_calibration_gpu_identity(
        inputs.calibration_profile_path,
        inputs.calibration_audit_path,
        expected_purpose=design.purpose,
        require_pcie_topology=(
            design.purpose is DiagnosticEvidencePurpose.PRODUCTION
        ),
    )
    if design.purpose is DiagnosticEvidencePurpose.PRODUCTION:
        collected_gpu = load_collection_gpu_identity(
            inputs.held_out_corpus_path,
            corpus_root=collection,
        )
        if collected_gpu != gpu_identity:
            raise ValueError("collection/calibration GPU identity mismatch")
    return gpu_identity


def _build_plan(
    *,
    root: Path,
    inputs: LifecyclePlanInputs,
    design: DiagnosticDesignManifest,
    design_path: Path,
    snapshot: DiagnosticCorpusSnapshotManifest,
    snapshot_path: Path,
    source: GitSourceState,
) -> DiagnosticLifecyclePlan:
    collection, inventory, held_out = _collection_inputs(inputs, design)
    generation = _next_generation(root, design.stage_id)
    gpu_identity = _plan_gpu_identity(inputs, design, collection)
    policy_artifact, frozen_inference_artifact = _pre_frozen_inputs(
        design, inputs
    )
    run_id = collection_run_id(
        design_id=design.stage_id,
        generation=generation,
        roles=("held_out",),
        frozen_held_out_sha256=held_out.sha256,
        gpu_identity=gpu_identity,
        source_revision=source.revision,
        purpose=design.purpose,
    )
    provisional = DiagnosticLifecyclePlan.model_construct(
        schema_version=SchemaVersion.DIAGNOSTIC_LIFECYCLE_PLAN,
        plan_id="0" * 64,
        design=_parent(DiagnosticLifecycleStage.DESIGN, design, design_path),
        development_snapshot=_parent(
            DiagnosticLifecycleStage.CORPUS_SNAPSHOT, snapshot, snapshot_path
        ),
        collection_root=str(collection),
        collection_inventory=inventory,
        collection_run_id=run_id,
        gpu_identity=gpu_identity,
        generation=generation,
        roles=("held_out",),
        calibration_profile_path=str(inputs.calibration_profile_path.resolve()),
        calibration_profile=_artifact(
            inputs.calibration_profile_path,
            relative_path=inputs.calibration_profile_path.name,
        ),
        calibration_audit_path=str(inputs.calibration_audit_path.resolve()),
        calibration_audit=_artifact(
            inputs.calibration_audit_path,
            relative_path=inputs.calibration_audit_path.name,
        ),
        held_out_corpus_path=str(inputs.held_out_corpus_path.resolve()),
        held_out_corpus=held_out,
        output_root=str(inputs.output_root.resolve()),
        source_revision=source.revision,
        purpose=design.purpose,
        model_version=inputs.model_version,
        max_attempts=inputs.max_attempts,
        vram_policy_path=(
            str(inputs.vram_policy_path.resolve())
            if inputs.vram_policy_path is not None
            else None
        ),
        vram_policy=policy_artifact,
        frozen_inference_profile_path=(
            str(inputs.frozen_inference_profile_path.resolve())
            if inputs.frozen_inference_profile_path is not None
            else None
        ),
        frozen_inference_profile=frozen_inference_artifact,
    )
    return _finalize_plan(provisional)


def _pre_frozen_inputs(
    design: DiagnosticDesignManifest,
    inputs: LifecyclePlanInputs,
) -> tuple[
    DiagnosticLifecycleArtifact | None, DiagnosticLifecycleArtifact | None
]:
    from sol_execbench.core.bench.performance_model.inference import (
        DiagnosticInferenceProfile,
    )

    if design.vram_policy_sha256 is None:
        if (
            inputs.vram_policy_path is not None
            or inputs.frozen_inference_profile_path is not None
        ):
            raise ValueError("legacy design cannot acquire pre-frozen inputs")
        return None, None
    if (
        inputs.vram_policy_path is None
        or inputs.frozen_inference_profile_path is None
    ):
        raise ValueError(
            "capacity-governed design requires policy and frozen inference"
        )
    load_json_file(DiagnosticVRAMWorkingSetPolicy, inputs.vram_policy_path)
    profile = load_json_file(
        DiagnosticCalibrationProfile, inputs.calibration_profile_path
    )
    inference = load_json_file(
        DiagnosticInferenceProfile, inputs.frozen_inference_profile_path
    )
    policy_digest = sha256_file(inputs.vram_policy_path)
    if (
        policy_digest != design.vram_policy_sha256
        or policy_digest not in profile.probe_evidence_sha256
    ):
        raise ValueError("design/calibration VRAM policy identity mismatch")
    if inference.model_version != inputs.model_version:
        raise ValueError("frozen inference model version mismatch")
    return (
        _artifact(inputs.vram_policy_path, relative_path="vram-policy.json"),
        _artifact(
            inputs.frozen_inference_profile_path,
            relative_path="frozen-inference.json",
        ),
    )


def author_lifecycle_plan(
    *,
    repository_root: Path,
    store_root: Path,
    inputs: LifecyclePlanInputs,
) -> DiagnosticLifecyclePlan:
    """Build a complete plan from verified registry and filesystem inputs."""
    root = store_root.resolve()
    design, design_path, snapshot, snapshot_path = _load_registry_inputs(
        root, inputs
    )
    source = capture_git_source_state(
        repository_root.resolve(), paths=_SOURCE_PATHS
    )
    if not source.clean:
        raise ValueError("lifecycle source paths contain uncommitted changes")
    return _build_plan(
        root=root,
        inputs=inputs,
        design=design,
        design_path=design_path,
        snapshot=snapshot,
        snapshot_path=snapshot_path,
        source=source,
    )


__all__ = ["LifecyclePlanInputs", "author_lifecycle_plan"]
