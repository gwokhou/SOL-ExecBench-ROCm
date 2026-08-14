# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Collection and calibration lifecycle stage handlers."""

from __future__ import annotations

from sol_execbench.core.bench.performance_model.case_reuse import (
    load_and_verify_case_reuse_bundle,
)
from sol_execbench.core.bench.performance_model.lifecycle.collection_identity import (
    load_collection_gpu_identity,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
    DiagnosticLifecycleStage,
)
from sol_execbench.core.bench.performance_model.lifecycle.execution import (
    StageCompletion,
    StageRunContext,
    artifact_for_path,
    verify_artifacts,
)
from sol_execbench.core.bench.performance_model.lifecycle.identity import (
    calibration_id,
    corpus_snapshot_id,
)
from sol_execbench.core.bench.performance_model.lifecycle.inventory import (
    verify_regular_tree_inventory,
)
from sol_execbench.core.bench.performance_model.lifecycle.receipts import (
    DiagnosticStageReceipt,
)
from sol_execbench.core.bench.performance_model.lifecycle.records import (
    _calibration_identities,
    _load_design,
    _load_design_or_none,
    _required,
)
from sol_execbench.core.bench.performance_model.lifecycle.run_state import (
    DiagnosticRunManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleParent,
)
from sol_execbench.core.bench.performance_model.lifecycle.stage_support import (
    _parents_of,
)
from sol_execbench.core.integrity import sha256_file

DEPENDENCIES: dict[
    DiagnosticLifecycleStage, tuple[DiagnosticLifecycleStage, ...]
] = {
    DiagnosticLifecycleStage.DESIGN: (),
    DiagnosticLifecycleStage.CALIBRATION: (),
    DiagnosticLifecycleStage.COLLECTION_RUN: (DiagnosticLifecycleStage.DESIGN,),
    DiagnosticLifecycleStage.CORPUS_SNAPSHOT: (
        DiagnosticLifecycleStage.COLLECTION_RUN,
    ),
    DiagnosticLifecycleStage.MODEL_BUILD: (
        DiagnosticLifecycleStage.CALIBRATION,
    ),
    DiagnosticLifecycleStage.ACCEPTANCE: (
        DiagnosticLifecycleStage.MODEL_BUILD,
        DiagnosticLifecycleStage.CALIBRATION,
        DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
    ),
    DiagnosticLifecycleStage.PUBLICATION: (
        DiagnosticLifecycleStage.ACCEPTANCE,
        DiagnosticLifecycleStage.CALIBRATION,
        DiagnosticLifecycleStage.MODEL_BUILD,
    ),
    DiagnosticLifecycleStage.RELEASE: (DiagnosticLifecycleStage.PUBLICATION,),
}


class DesignHandler:
    """Verify the preregistered design that roots a lifecycle run."""

    stage = DiagnosticLifecycleStage.DESIGN

    def run(self, context: StageRunContext) -> StageCompletion:
        """Record the design manifest as the chain root."""
        design = _load_design(context)
        return StageCompletion(
            stage_id=design.stage_id,
            outputs=(),
        )

    def prepare(
        self,
        context: StageRunContext,
        run_state: DiagnosticRunManifest,
    ) -> tuple[DiagnosticLifecycleParent, ...]:
        """Return the design's input identities (none: it is the root)."""
        del context, run_state
        return ()

    def verify(
        self,
        context: StageRunContext,
        receipt: DiagnosticStageReceipt,
    ) -> bool:
        """Re-verify the design manifest against the recorded receipt."""
        design = _load_design_or_none(context)
        if design is None or design.stage_id != receipt.stage_id:
            return False
        return (
            not receipt.output_inventory
            and sha256_file(context.design_manifest_path)
            == context.plan.design.sha256
        )


class CollectionRunHandler:
    """Adopt one operator-collected generation beneath a frozen design."""

    stage = DiagnosticLifecycleStage.COLLECTION_RUN

    def run(self, context: StageRunContext) -> StageCompletion:
        """Record the collected generation from its evidence tree."""
        if context.corpus_root is None:
            raise ValueError("collection run requires --corpus-root")
        if not verify_regular_tree_inventory(
            context.corpus_root, context.plan.collection_inventory
        ):
            raise ValueError("collection inventory differs from reviewed plan")
        held_out = _required(
            context.held_out_corpus_path,
            "collection run is missing frozen held-out corpus",
        )
        load_and_verify_case_reuse_bundle(held_out)
        if not _collection_gpu_identity_matches_plan(context):
            raise ValueError(
                "collection GPU identity differs from reviewed plan"
            )
        return StageCompletion(
            stage_id=context.collection_run_id,
            outputs=context.plan.collection_inventory,
            output_paths=tuple(
                context.corpus_root / item.relative_path
                for item in context.plan.collection_inventory
            ),
        )

    def prepare(
        self,
        context: StageRunContext,
        run_state: DiagnosticRunManifest,
    ) -> tuple[DiagnosticLifecycleParent, ...]:
        """Return the frozen design as the collection input identity."""
        return _parents_of(context, run_state, self.stage)

    def verify(
        self,
        context: StageRunContext,
        receipt: DiagnosticStageReceipt,
    ) -> bool:
        """Re-check that the collected evidence tree still exists."""
        if context.corpus_root is None:
            return False
        return (
            verify_regular_tree_inventory(
                context.corpus_root, receipt.output_inventory
            )
            and _collection_gpu_identity_matches_plan(context)
            and _case_reuse_bundle_is_valid(context)
        )


def _collection_gpu_identity_matches_plan(context: StageRunContext) -> bool:
    if context.purpose is not DiagnosticEvidencePurpose.PRODUCTION:
        return True
    if context.corpus_root is None or context.held_out_corpus_path is None:
        return False
    try:
        observed = load_collection_gpu_identity(
            context.held_out_corpus_path,
            corpus_root=context.corpus_root,
        )
    except (OSError, ValueError):
        return False
    return observed == context.plan.gpu_identity


def _case_reuse_bundle_is_valid(context: StageRunContext) -> bool:
    path = context.held_out_corpus_path
    if path is None:
        return False
    try:
        load_and_verify_case_reuse_bundle(path)
    except (OSError, ValueError):
        return False
    return True


class CalibrationHandler:
    """Adopt and verify one immutable hardware calibration pair."""

    stage = DiagnosticLifecycleStage.CALIBRATION

    def run(self, context: StageRunContext) -> StageCompletion:
        """Validate the calibration pair and derive its bound identity."""
        profile_path = _required(
            context.calibration_profile_path,
            "calibration requires --calibration-profile",
        )
        audit_path = _required(
            context.calibration_audit_path,
            "calibration requires --calibration-audit",
        )
        for path, expected in (
            (profile_path, context.plan.calibration_profile),
            (audit_path, context.plan.calibration_audit),
        ):
            if (
                not path.is_file()
                or path.stat().st_size != expected.size_bytes
                or sha256_file(path) != expected.sha256
            ):
                raise ValueError("calibration input differs from reviewed plan")
        if context.plan.vram_policy is not None:
            policy_path = _required(
                context.vram_policy_path,
                "calibration requires the reviewed VRAM policy",
            )
            expected = context.plan.vram_policy
            if (
                not policy_path.is_file()
                or policy_path.stat().st_size != expected.size_bytes
                or sha256_file(policy_path) != expected.sha256
            ):
                raise ValueError("VRAM policy differs from reviewed plan")
        gpu, software = _calibration_identities(
            profile_path,
            audit_path,
            purpose=context.purpose,
        )
        stage_id = calibration_id(
            calibration_profile_sha256=sha256_file(profile_path),
            calibration_audit_sha256=sha256_file(audit_path),
            gpu_identity=gpu,
            software_identity=software,
            source_revision=context.source_revision,
            purpose=context.purpose,
        )
        return StageCompletion(
            stage_id=stage_id,
            outputs=(
                artifact_for_path(profile_path),
                artifact_for_path(audit_path),
            ),
            output_paths=(profile_path, audit_path),
        )

    def prepare(
        self,
        context: StageRunContext,
        run_state: DiagnosticRunManifest,
    ) -> tuple[DiagnosticLifecycleParent, ...]:
        """Return the immutable design dependency."""
        return _parents_of(context, run_state, self.stage)

    def verify(
        self,
        context: StageRunContext,
        receipt: DiagnosticStageReceipt,
    ) -> bool:
        """Re-verify both calibration artifacts from the receipt."""
        profile = context.calibration_profile_path
        audit = context.calibration_audit_path
        if profile is None or audit is None or profile.parent != audit.parent:
            return False
        return verify_artifacts(receipt.output_inventory, profile.parent)


class CorpusSnapshotHandler:
    """Validate the fresh held-out corpus snapshot for this generation."""

    stage = DiagnosticLifecycleStage.CORPUS_SNAPSHOT

    def run(self, context: StageRunContext) -> StageCompletion:
        """Record the held-out corpus and derive its direct-child identity."""
        if context.corpus_root is None:
            raise ValueError("corpus snapshot requires --corpus-root")
        path = _required(
            context.held_out_corpus_path,
            "frozen held-out corpus is missing",
        )
        artifact = context.plan.held_out_corpus
        if (
            not path.is_file()
            or path.stat().st_size != artifact.size_bytes
            or sha256_file(path) != artifact.sha256
        ):
            raise ValueError("held-out corpus differs from reviewed plan")
        digest = corpus_snapshot_id(
            collection_run_id=context.collection_run_id,
            role="held_out",
            corpus_sha256=artifact.sha256,
            source_revision=context.source_revision,
            purpose=context.purpose,
        )
        return StageCompletion(
            stage_id=digest,
            outputs=(artifact,),
            output_paths=(path,),
        )

    def prepare(
        self,
        context: StageRunContext,
        run_state: DiagnosticRunManifest,
    ) -> tuple[DiagnosticLifecycleParent, ...]:
        """Return the collection generation as the snapshot input identity."""
        return _parents_of(context, run_state, self.stage)

    def verify(
        self,
        context: StageRunContext,
        receipt: DiagnosticStageReceipt,
    ) -> bool:
        """Re-verify the frozen corpus files against the receipt."""
        if context.corpus_root is None:
            return False
        return verify_artifacts(
            receipt.output_inventory,
            context.corpus_root,
        )
