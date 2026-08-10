# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Resumable lifecycle DAG orchestration with verification-based status.

The orchestrator owns the monotonic chain, bounded attempts, typed receipts,
legal transitions, and atomic run-state persistence. It never re-implements
hardware collection or model-fitting logic: each stage delegates to a thin
``DiagnosticStageHandler`` that wraps an existing low-level entry point.
Status and resume re-verify every recorded stage through the handler rather
than trusting that an output filename exists.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

from pydantic import TypeAdapter

if TYPE_CHECKING:
    from sol_execbench.core.bench.performance_model.acceptance import (
        DiagnosticAcceptanceManifest,
        DiagnosticAcceptanceResult,
    )
    from sol_execbench.core.bench.performance_model.builder import (
        SemanticCharacterizationLoader,
    )
    from sol_execbench.core.bench.performance_model.lifecycle.resolver import (
        ReferenceResolver,
    )
    from sol_execbench.core.bench.performance_model.publication import (
        SolarManifestProjectionVerifier,
        SolarManifestProjector,
    )

from sol_execbench.core.bench.performance_model.case_reuse import (
    EXPOSURE_RECEIPT_NAME,
    AcceptancePreconditionError,
    DiagnosticAcceptanceExposureReceipt,
    load_and_verify_case_reuse_bundle,
    persist_acceptance_exposure,
)
from sol_execbench.core.bench.performance_model.lifecycle.blob_store import (
    BlobStore,
)
from sol_execbench.core.bench.performance_model.lifecycle.calibration_identity import (
    load_calibration_gpu_identity,
)
from sol_execbench.core.bench.performance_model.lifecycle.collection_identity import (
    load_collection_gpu_identity,
)
from sol_execbench.core.bench.performance_model.lifecycle.corpus_registry import (
    snapshot_blob_inventory,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticAttemptFailureCode,
    DiagnosticAttemptStatus,
    DiagnosticEvidencePurpose,
    DiagnosticLifecycleStage,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
)
from sol_execbench.core.bench.performance_model.lifecycle.identity import (
    acceptance_id,
    calibration_id,
    corpus_snapshot_id,
    model_build_id,
    publication_id,
    recompute_stage_id,
)
from sol_execbench.core.bench.performance_model.lifecycle.inventory import (
    verify_regular_tree_inventory,
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
    GpuLifecycleIdentity,
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
    store_root,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.process import exclusive_file_lock, redacted_text_tail

CHAIN: tuple[DiagnosticLifecycleStage, ...] = (
    DiagnosticLifecycleStage.DESIGN,
    DiagnosticLifecycleStage.CALIBRATION,
    DiagnosticLifecycleStage.COLLECTION_RUN,
    DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
    DiagnosticLifecycleStage.MODEL_BUILD,
    DiagnosticLifecycleStage.ACCEPTANCE,
    DiagnosticLifecycleStage.PUBLICATION,
    DiagnosticLifecycleStage.RELEASE,
)

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

_CHAIN_INDEX = {stage: index for index, stage in enumerate(CHAIN)}

_RECEIPT_ADAPTER = TypeAdapter(DiagnosticStageReceipt)


@dataclass
class StageRunContext:
    """Mutable carrier of the fixed run inputs and produced output paths."""

    store_root: Path
    plan: DiagnosticLifecyclePlan
    design_manifest_path: Path
    collection_run_id: str
    generation: int
    purpose: DiagnosticEvidencePurpose = DiagnosticEvidencePurpose.PRODUCTION
    corpus_root: Path | None = None
    calibration_profile_path: Path | None = None
    calibration_audit_path: Path | None = None
    development_corpus_path: Path | None = None
    held_out_corpus_path: Path | None = None
    output_root: Path | None = None
    source_revision: str = "unknown"
    model_version: str = "gfx1200_diagnostic.v7"
    paths: dict[str, Path] = field(default_factory=dict)

    def output(self, stage: DiagnosticLifecycleStage) -> Path | None:
        """Return the recorded primary output path for one stage."""
        return self.paths.get(stage.value)

    def set_output(self, stage: DiagnosticLifecycleStage, path: Path) -> None:
        """Record the primary output path for one completed stage."""
        self.paths[stage.value] = Path(path)


@dataclass(frozen=True)
class StageCompletion:
    """One completed stage's produced identity and exact output inventory."""

    stage_id: str
    outputs: tuple[DiagnosticLifecycleArtifact, ...]
    output_paths: tuple[Path, ...] = ()


@runtime_checkable
class DiagnosticStageHandler(Protocol):
    """One lifecycle stage adapter: perform, prepare, and re-verify."""

    stage: DiagnosticLifecycleStage

    def run(self, context: StageRunContext) -> StageCompletion:
        """Execute the stage and return its produced identity and outputs."""
        ...

    def prepare(
        self,
        context: StageRunContext,
        run_state: DiagnosticRunManifest,
    ) -> tuple[DiagnosticLifecycleParent, ...]:
        """Return the immutable input identities the stage consumes."""
        ...

    def verify(
        self,
        context: StageRunContext,
        receipt: DiagnosticStageReceipt,
    ) -> bool:
        """Re-verify the completed stage; returns whether it still holds."""
        ...


def _artifact(path: Path) -> DiagnosticLifecycleArtifact:
    return DiagnosticLifecycleArtifact(
        relative_path=path.name,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _verify_artifacts(
    artifacts: Sequence[DiagnosticLifecycleArtifact],
    base: Path,
) -> bool:
    """Re-check that every recorded artifact still verifies under *base*."""
    resolved_base = base.resolve()
    for item in artifacts:
        if item.relative_path == "":
            continue
        candidate = resolved_base / item.relative_path
        if candidate.is_symlink():
            return False
        path = candidate.resolve()
        if not path.is_relative_to(resolved_base) or not path.is_file():
            return False
        if path.stat().st_size != item.size_bytes:
            return False
        if sha256_file(path) != item.sha256:
            return False
    return True


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
            outputs=(_artifact(profile_path), _artifact(audit_path)),
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
        return _verify_artifacts(receipt.output_inventory, profile.parent)


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
        return _verify_artifacts(
            receipt.output_inventory,
            context.corpus_root,
        )


class ModelBuildHandler:
    """Fit the frozen inference model build from development evidence."""

    stage = DiagnosticLifecycleStage.MODEL_BUILD

    def __init__(
        self,
        semantic_loader: SemanticCharacterizationLoader,
        blob_resolver: ReferenceResolver | None = None,
    ) -> None:
        """Bind the semantic loader and blob resolver for fitting."""
        self._semantic_loader = semantic_loader
        self._blob_resolver = blob_resolver

    def run(self, context: StageRunContext) -> StageCompletion:
        """Fit the inference profile and record it as the build output."""
        from sol_execbench.core.bench.performance_model.authoring import (
            fit_diagnostic_inference_profile,
        )

        development = _required(
            context.development_corpus_path,
            "model build requires --development-corpus",
        )
        calibration = _required(
            context.calibration_profile_path,
            "model build requires --calibration-profile",
        )
        calibration_audit = _required(
            context.calibration_audit_path,
            "model build requires --calibration-audit",
        )
        output = (
            _require_output_root(context) / "model-build" / "inference.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        profile = fit_diagnostic_inference_profile(
            development_corpus_path=development,
            calibration_profile_path=calibration,
            semantic_loader=self._semantic_loader,
            blob_resolver=self._blob_resolver,
        )
        atomic_write_json_value(output, profile.model_dump(mode="json"))
        context.set_output(self.stage, output)
        stage_id = model_build_id(
            calibration_id=_prior_receipt_stage_id(
                context,
                DiagnosticLifecycleStage.CALIBRATION,
            ),
            development_snapshot_id=(
                context.plan.development_snapshot.stage_id
            ),
            calibration_profile_sha256=sha256_file(calibration),
            calibration_audit_sha256=sha256_file(calibration_audit),
            inference_profile_sha256=sha256_file(output),
            model_version=context.model_version,
            source_revision=context.source_revision,
            purpose=context.purpose,
        )
        return StageCompletion(
            stage_id=stage_id,
            outputs=(_artifact(output),),
            output_paths=(output,),
        )

    def prepare(
        self,
        context: StageRunContext,
        run_state: DiagnosticRunManifest,
    ) -> tuple[DiagnosticLifecycleParent, ...]:
        """Return the corpus snapshot as the model-build input identity."""
        return _parents_of(context, run_state, self.stage)

    def verify(
        self,
        context: StageRunContext,
        receipt: DiagnosticStageReceipt,
    ) -> bool:
        """Re-verify the fitted inference profile against the receipt."""
        output = context.output(self.stage)
        if output is None:
            return False
        return _verify_artifacts(
            receipt.output_inventory,
            output.parent,
        )


class AcceptanceHandler:
    """Evaluate the frozen profile once against disjoint held-out evidence."""

    stage = DiagnosticLifecycleStage.ACCEPTANCE

    def __init__(
        self,
        semantic_loader: SemanticCharacterizationLoader,
        blob_resolver: ReferenceResolver | None = None,
    ) -> None:
        """Bind the semantic loader and blob resolver for acceptance."""
        self._semantic_loader = semantic_loader
        self._blob_resolver = blob_resolver

    def run(self, context: StageRunContext) -> StageCompletion:
        """Run held-out acceptance and record its manifest and result."""
        development = _required(
            context.development_corpus_path,
            "acceptance requires --development-corpus",
        )
        held_out = _required(
            context.held_out_corpus_path,
            "acceptance requires --held-out-corpus",
        )
        calibration = _required(
            context.calibration_profile_path,
            "acceptance requires --calibration-profile",
        )
        inference = _required(
            context.output(DiagnosticLifecycleStage.MODEL_BUILD),
            "acceptance requires the model-build inference profile",
        )
        load_and_verify_case_reuse_bundle(held_out)
        manifest_output, result_output, manifest, result = self._build_outputs(
            context, development, held_out, calibration, inference
        )
        atomic_write_json_value(
            manifest_output,
            manifest.model_dump(mode="json"),
        )
        atomic_write_json_value(result_output, result.model_dump(mode="json"))
        context.set_output(self.stage, manifest_output)
        outputs = (_artifact(manifest_output), _artifact(result_output))
        held_out_snapshot_id = corpus_snapshot_id(
            collection_run_id=context.collection_run_id,
            role="held_out",
            corpus_sha256=manifest.held_out_corpus_sha256,
            source_revision=context.source_revision,
            purpose=context.purpose,
        )
        stage_id = acceptance_id(
            calibration_id=_prior_receipt_stage_id(
                context,
                DiagnosticLifecycleStage.CALIBRATION,
            ),
            development_snapshot_id=(
                context.plan.development_snapshot.stage_id
            ),
            model_build_id=_prior_receipt_stage_id(
                context,
                DiagnosticLifecycleStage.MODEL_BUILD,
            ),
            held_out_corpus_snapshot_id=held_out_snapshot_id,
            accepted=result.accepted,
            verdict_sha256=sha256_file(result_output),
            source_revision=context.source_revision,
            purpose=context.purpose,
        )
        return StageCompletion(
            stage_id=stage_id,
            outputs=outputs,
            output_paths=(manifest_output, result_output),
        )

    def _build_outputs(
        self,
        context: StageRunContext,
        development: Path,
        held_out: Path,
        calibration: Path,
        inference: Path,
    ) -> tuple[
        Path,
        Path,
        DiagnosticAcceptanceManifest,
        DiagnosticAcceptanceResult,
    ]:
        """Build acceptance outputs or persist the exact precondition leak."""
        from sol_execbench.core.bench.performance_model.authoring import (
            build_diagnostic_acceptance,
        )

        root = _require_output_root(context) / "acceptance"
        root.mkdir(parents=True, exist_ok=True)
        try:
            manifest, result = build_diagnostic_acceptance(
                development_corpus_path=development,
                held_out_corpus_path=held_out,
                calibration_profile_path=calibration,
                inference_profile_path=inference,
                semantic_loader=self._semantic_loader,
                blob_resolver=self._blob_resolver,
            )
        except AcceptancePreconditionError as error:
            self._record_exposure(context, held_out, root, error)
            raise
        return (
            root / "acceptance-manifest.json",
            root / "acceptance.json",
            manifest,
            result,
        )

    @staticmethod
    def _record_exposure(
        context: StageRunContext,
        held_out: Path,
        root: Path,
        error: AcceptancePreconditionError,
    ) -> None:
        """Persist the pre-verdict release as content-addressed evidence."""
        receipt = DiagnosticAcceptanceExposureReceipt(
            purpose=context.purpose,
            run_id=context.collection_run_id,
            held_out_corpus_sha256=sha256_file(held_out),
            source_revision=context.source_revision,
            evaluated_case_ids_before_failure=(
                error.evaluated_case_ids_before_failure
            ),
            released_case_id=error.case_id,
            released_workload_kind=error.workload_kind,
            released_reason_codes=error.reason_codes,
            created_at=_now(),
        )
        exposure_output = root / EXPOSURE_RECEIPT_NAME
        atomic_write_json_value(
            exposure_output, receipt.model_dump(mode="json")
        )
        error.bind_exposure_receipt(
            persist_acceptance_exposure(
                receipt, exposure_output, context.store_root
            )
        )

    def prepare(
        self,
        context: StageRunContext,
        run_state: DiagnosticRunManifest,
    ) -> tuple[DiagnosticLifecycleParent, ...]:
        """Return the model build as the acceptance input identity."""
        return _parents_of(context, run_state, self.stage)

    def verify(
        self,
        context: StageRunContext,
        receipt: DiagnosticStageReceipt,
    ) -> bool:
        """Re-verify the acceptance manifest and result files."""
        output = context.output(self.stage)
        if output is None:
            return False
        return _verify_artifacts(
            receipt.output_inventory,
            output.parent,
        )


class PublicationHandler:
    """Project the compact, self-verifying publication tree."""

    stage = DiagnosticLifecycleStage.PUBLICATION

    def __init__(
        self,
        semantic_loader: SemanticCharacterizationLoader,
        solar_projector: SolarManifestProjector,
        solar_verifier: SolarManifestProjectionVerifier,
        blob_resolver: ReferenceResolver | None = None,
    ) -> None:
        """Bind the SOLAR projection tools and blob resolver."""
        self._semantic_loader = semantic_loader
        self._solar_projector = solar_projector
        self._solar_verifier = solar_verifier
        self._blob_resolver = blob_resolver

    def run(self, context: StageRunContext) -> StageCompletion:
        """Build the publication tree and record its projection identity."""
        from sol_execbench.core.bench.performance_model.acceptance import (
            DiagnosticAcceptanceResult,
        )
        from sol_execbench.core.bench.performance_model.publication import (
            DiagnosticPublicationProjection,
            build_diagnostic_publication_projection,
        )

        acceptance_manifest = _required(
            context.output(DiagnosticLifecycleStage.ACCEPTANCE),
            "publication requires the acceptance manifest",
        )
        acceptance_result = acceptance_manifest.with_name("acceptance.json")
        verdict = load_json_file(DiagnosticAcceptanceResult, acceptance_result)
        if not verdict.accepted:
            raise ValueError(
                "publication refused: held-out acceptance is terminally rejected"
            )

        development = _required(
            context.development_corpus_path,
            "publication requires --development-corpus",
        )
        calibration = _required(
            context.calibration_profile_path,
            "publication requires --calibration-profile",
        )
        source_inference = _required(
            context.output(DiagnosticLifecycleStage.MODEL_BUILD),
            "publication requires the model-build inference profile",
        )
        output = _require_output_root(context) / "publication"
        manifest_path = build_diagnostic_publication_projection(
            development_corpus_path=development,
            calibration_profile_path=calibration,
            source_inference_profile_path=source_inference,
            output_root=output,
            semantic_loader=self._semantic_loader,
            solar_projector=self._solar_projector,
            solar_verifier=self._solar_verifier,
            blob_resolver=self._blob_resolver,
        )
        context.set_output(self.stage, output)
        projection = load_json_file(
            DiagnosticPublicationProjection,
            manifest_path,
        )
        stage_id = publication_id(
            acceptance_id=_prior_receipt_stage_id(
                context,
                DiagnosticLifecycleStage.ACCEPTANCE,
            ),
            calibration_id=_prior_receipt_stage_id(
                context,
                DiagnosticLifecycleStage.CALIBRATION,
            ),
            development_snapshot_id=(
                context.plan.development_snapshot.stage_id
            ),
            model_build_id=_prior_receipt_stage_id(
                context,
                DiagnosticLifecycleStage.MODEL_BUILD,
            ),
            source_corpus_sha256=projection.source_corpus_sha256,
            publication_manifest_sha256=sha256_file(manifest_path),
            uncompressed_size_bytes=projection.uncompressed_size_bytes,
            case_count=projection.case_count,
            source_revision=context.source_revision,
            purpose=context.purpose,
        )
        return StageCompletion(
            stage_id=stage_id,
            outputs=(_artifact(manifest_path),),
            output_paths=(manifest_path,),
        )

    def prepare(
        self,
        context: StageRunContext,
        run_state: DiagnosticRunManifest,
    ) -> tuple[DiagnosticLifecycleParent, ...]:
        """Return the acceptance verdict as the publication input identity."""
        return _parents_of(context, run_state, self.stage)

    def verify(
        self,
        context: StageRunContext,
        receipt: DiagnosticStageReceipt,
    ) -> bool:
        """Re-run the publication projection verifier on the recorded tree."""
        from sol_execbench.core.bench.performance_model.publication import (
            verify_diagnostic_publication_projection,
        )

        del receipt
        tree = context.output(self.stage)
        if tree is None:
            return False
        manifest = tree / "publication.json"
        if not manifest.is_file():
            return False
        try:
            verify_diagnostic_publication_projection(
                manifest_path=manifest,
                semantic_loader=self._semantic_loader,
                solar_verifier=self._solar_verifier,
                blob_resolver=self._blob_resolver,
            )
        except (OSError, ValueError):
            return False
        return True


class ReleaseHandler:
    """Package a verified publication into a deterministic release object."""

    stage = DiagnosticLifecycleStage.RELEASE

    def __init__(
        self,
        semantic_loader: SemanticCharacterizationLoader,
        solar_verifier: SolarManifestProjectionVerifier,
    ) -> None:
        """Bind the publication verification tools for packaging."""
        self._semantic_loader = semantic_loader
        self._solar_verifier = solar_verifier

    def run(self, context: StageRunContext) -> StageCompletion:
        """Package the publication archive and record its release identity."""
        from sol_execbench.core.bench.performance_model.release import (
            package_diagnostic_publication,
        )

        tree = _required(
            context.output(DiagnosticLifecycleStage.PUBLICATION),
            "release requires the publication tree",
        )
        manifest = tree / "publication.json"
        if not manifest.is_file():
            raise ValueError(f"publication manifest is missing: {manifest}")
        root = _require_output_root(context)
        archive = root / "release.tar.zst"
        attestation_path = archive.with_suffix(".attestation.json")
        attestation = package_diagnostic_publication(
            manifest_path=manifest,
            archive_output=archive,
            attestation_output=attestation_path,
            source_revision=context.source_revision,
            semantic_loader=self._semantic_loader,
            solar_verifier=self._solar_verifier,
            store_root_path=context.store_root,
            purpose=context.purpose,
        )
        context.set_output(self.stage, archive)
        return StageCompletion(
            stage_id=attestation.release_id,
            outputs=(_artifact(archive), _artifact(attestation_path)),
            output_paths=(archive, attestation_path),
        )

    def prepare(
        self,
        context: StageRunContext,
        run_state: DiagnosticRunManifest,
    ) -> tuple[DiagnosticLifecycleParent, ...]:
        """Return the publication as the release input identity."""
        return _parents_of(context, run_state, self.stage)

    def verify(
        self,
        context: StageRunContext,
        receipt: DiagnosticStageReceipt,
    ) -> bool:
        """Re-run the release archive verifier on the recorded archive."""
        from sol_execbench.core.bench.performance_model.release import (
            verify_diagnostic_release_archive,
        )

        del receipt
        archive = context.output(self.stage)
        if archive is None or not archive.is_file():
            return False
        try:
            verify_diagnostic_release_archive(
                archive_path=archive,
                semantic_loader=self._semantic_loader,
                solar_verifier=self._solar_verifier,
            )
        except (OSError, ValueError):
            return False
        return True


def _required(path: Path | None, message: str) -> Path:
    if path is None:
        raise ValueError(message)
    return Path(path)


def _require_output_root(context: StageRunContext) -> Path:
    if context.output_root is None:
        raise ValueError("this lifecycle stage requires --output-root")
    root = context.output_root
    root.mkdir(parents=True, exist_ok=True)
    return root


def _parents_of(
    context: StageRunContext,
    run_state: DiagnosticRunManifest,
    stage: DiagnosticLifecycleStage,
) -> tuple[DiagnosticLifecycleParent, ...]:
    parents: list[DiagnosticLifecycleParent] = []
    for dependency in DEPENDENCIES[stage]:
        prior = run_state.stage_state(dependency)
        if prior is None or prior.receipt_path == "":
            raise ValueError(
                f"{stage.value} requires verified {dependency.value}"
            )
        receipt = _load_receipt(
            context.collection_run_id, dependency, context.store_root
        )
        if receipt is None or receipt.purpose is not context.purpose:
            raise ValueError(
                f"{stage.value} has missing or cross-purpose {dependency.value}"
            )
        manifest_path = _stage_manifest_path(
            context.store_root, dependency, receipt.stage_id
        )
        if not manifest_path.is_file():
            raise ValueError(
                f"{stage.value} dependency manifest is missing: {manifest_path}"
            )
        parents.append(
            DiagnosticLifecycleParent(
                stage=dependency,
                purpose=context.purpose,
                stage_id=receipt.stage_id,
                sha256=sha256_file(manifest_path),
            )
        )
    if stage in {
        DiagnosticLifecycleStage.MODEL_BUILD,
        DiagnosticLifecycleStage.ACCEPTANCE,
        DiagnosticLifecycleStage.PUBLICATION,
    }:
        development = context.plan.development_snapshot
        manifest_path = _stage_manifest_path(
            context.store_root, development.stage, development.stage_id
        )
        if (
            not manifest_path.is_file()
            or sha256_file(manifest_path) != development.sha256
        ):
            raise ValueError("promoted development snapshot identity drifted")
        manifest = load_json_file(
            DiagnosticCorpusSnapshotManifest, manifest_path
        )
        if (
            manifest.role != "development"
            or not manifest.source_snapshot_ids
            or manifest.purpose is not context.purpose
        ):
            raise ValueError("development snapshot is not a valid promotion")
        parents.append(development)
    return tuple(parents)


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


def build_stage_handlers(
    *,
    semantic_loader: SemanticCharacterizationLoader,
    solar_verifier: SolarManifestProjectionVerifier,
    solar_projector: SolarManifestProjector,
    blob_resolver: ReferenceResolver | None = None,
) -> Mapping[DiagnosticLifecycleStage, DiagnosticStageHandler]:
    """Return the real handler registry bound to the caller's tools.

    The SOLAR bridge and blob resolver are injected by the CLI so the
    lifecycle package never imports them directly.
    """
    return {
        DiagnosticLifecycleStage.DESIGN: DesignHandler(),
        DiagnosticLifecycleStage.CALIBRATION: CalibrationHandler(),
        DiagnosticLifecycleStage.COLLECTION_RUN: CollectionRunHandler(),
        DiagnosticLifecycleStage.CORPUS_SNAPSHOT: CorpusSnapshotHandler(),
        DiagnosticLifecycleStage.MODEL_BUILD: ModelBuildHandler(
            semantic_loader,
            blob_resolver,
        ),
        DiagnosticLifecycleStage.ACCEPTANCE: AcceptanceHandler(
            semantic_loader,
            blob_resolver,
        ),
        DiagnosticLifecycleStage.PUBLICATION: PublicationHandler(
            semantic_loader,
            solar_projector,
            solar_verifier,
            blob_resolver,
        ),
        DiagnosticLifecycleStage.RELEASE: ReleaseHandler(
            semantic_loader,
            solar_verifier,
        ),
    }


def build_run_context(
    *,
    plan: DiagnosticLifecyclePlan,
    store_root_path: Path | None = None,
) -> StageRunContext:
    """Build fixed run inputs from one validated immutable plan."""
    root = Path(store_root_path).resolve() if store_root_path else store_root()
    design_path = _stage_manifest_path(
        root, DiagnosticLifecycleStage.DESIGN, plan.design.stage_id
    )
    development_path = _stage_manifest_path(
        root,
        DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
        plan.development_snapshot.stage_id,
    )
    if (
        not design_path.is_file()
        or sha256_file(design_path) != plan.design.sha256
    ):
        raise ValueError("lifecycle plan design identity drifted")
    if (
        not development_path.is_file()
        or sha256_file(development_path) != plan.development_snapshot.sha256
    ):
        raise ValueError("lifecycle plan development snapshot identity drifted")
    development = load_json_file(
        DiagnosticCorpusSnapshotManifest, development_path
    )
    development_corpus = BlobStore(root).get(development.corpus_file_sha256)
    return StageRunContext(
        store_root=root,
        plan=plan,
        design_manifest_path=design_path,
        collection_run_id=plan.collection_run_id,
        generation=plan.generation,
        purpose=plan.purpose,
        corpus_root=Path(plan.collection_root),
        calibration_profile_path=Path(plan.calibration_profile_path),
        calibration_audit_path=Path(plan.calibration_audit_path),
        development_corpus_path=development_corpus,
        held_out_corpus_path=Path(plan.held_out_corpus_path),
        output_root=Path(plan.output_root),
        source_revision=plan.source_revision,
        model_version=plan.model_version,
    )


def run_diagnostic_lifecycle(
    *,
    plan_path: Path,
    store_root_path: Path | None = None,
    stages: Sequence[DiagnosticLifecycleStage] | None = None,
    handlers: Mapping[
        DiagnosticLifecycleStage,
        DiagnosticStageHandler,
    ],
    now_fn: Callable[[], str] | None = None,
) -> DiagnosticRunManifest:
    """Execute the selected chain stages and persist the run-state object.

    Stages run in monotonic chain order. Each scheduled stage requires its
    immediate chain predecessor to be verified (already recorded or running
    earlier in the same invocation); a selection that skips a chain link is
    rejected as an illegal transition. Every stage executes under a bounded
    attempt budget and writes its typed receipt and exact output inventory
    before it may be recorded ``verified``.
    """
    root = Path(store_root_path).resolve() if store_root_path else store_root()
    clock = now_fn or _now
    plan = load_json_file(DiagnosticLifecyclePlan, plan_path)
    run_context = build_run_context(plan=plan, store_root_path=root)
    plan_sha256 = _persist_plan(run_context)
    design = _load_design(run_context)
    requested = tuple(stages) if stages is not None else CHAIN
    ordered = _ordered_stages(requested)
    run_state = _initial_run_state(
        design,
        run_context,
        clock(),
        plan_sha256,
    )
    for stage in ordered:
        _validate_predecessor(run_state, stage)
        run_state = _execute_stage(
            run_state,
            run_context,
            stage,
            handlers,
            plan.max_attempts,
            clock,
        )
        if _stage_status(run_state, stage) is DiagnosticStageStatus.FAILED:
            break
    _write_run_state(run_context, run_state)
    return run_state


def resume_diagnostic_lifecycle(
    *,
    run_state_path: Path,
    handlers: Mapping[
        DiagnosticLifecycleStage,
        DiagnosticStageHandler,
    ],
    now_fn: Callable[[], str] | None = None,
) -> DiagnosticRunManifest:
    """Re-verify and continue a previously interrupted run-state object.

    Every recorded stage is re-verified first. A stage whose receipt is
    missing, whose inputs drifted, or whose outputs no longer verify is marked
    failed and re-executed with a fresh attempt budget; execution continues in
    chain order from the first incomplete stage.
    """
    run_state = load_json_file(DiagnosticRunManifest, run_state_path)
    context = _context_from_run_state(run_state, run_state_path)
    clock = now_fn or _now
    resumed = _reverify_past_stages(run_state, context, handlers)
    for stage in CHAIN:
        status = _stage_status(resumed, stage)
        if status in {
            DiagnosticStageStatus.VERIFIED,
            DiagnosticStageStatus.SUPERSEDED,
        }:
            continue
        resumed = _execute_stage(
            resumed,
            context,
            stage,
            handlers,
            context.plan.max_attempts,
            clock,
        )
        if _stage_status(resumed, stage) is DiagnosticStageStatus.FAILED:
            break
    _write_run_state(context, resumed)
    return resumed


def diagnostic_lifecycle_status(
    *,
    run_state_path: Path,
    handlers: Mapping[
        DiagnosticLifecycleStage,
        DiagnosticStageHandler,
    ],
) -> dict[str, object]:
    """Re-verify every recorded stage and report the current run state.

    A stage recorded as ``verified`` whose receipt or inputs no longer verify
    is reported as ``failed`` (drift), never as complete.
    """
    run_state = load_json_file(DiagnosticRunManifest, run_state_path)
    context = _context_from_run_state(run_state, run_state_path)
    verified = _reverify_past_stages(run_state, context, handlers)
    stages = [
        {
            "stage": item.stage.value,
            "status": item.status.value,
            "attempts": item.attempts,
            "stage_id": _produced_stage_id(context, item),
            "parents": _recorded_parents(context, item),
        }
        for item in verified.stages
    ]
    next_stage = _next_pending(verified)
    status = {
        "run_id": verified.run_id,
        "collection_run_id": verified.collection_run_id,
        "design_id": verified.design_id,
        "generation": verified.generation,
        "stages": stages,
        "development_snapshot_id": context.plan.development_snapshot.stage_id,
        "held_out_snapshot_id": _verified_stage_id(
            context, verified, DiagnosticLifecycleStage.CORPUS_SNAPSHOT
        ),
        "next_stage": next_stage.value if next_stage is not None else None,
    }
    _write_status_json(context, status)
    return status


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


def _ordered_stages(
    requested: Sequence[DiagnosticLifecycleStage],
) -> tuple[DiagnosticLifecycleStage, ...]:
    if len(set(requested)) != len(requested):
        raise ValueError("lifecycle stage list repeats a stage")
    if any(stage not in _CHAIN_INDEX for stage in requested):
        raise ValueError("lifecycle stage list contains an unknown stage")
    return tuple(sorted(requested, key=lambda stage: _CHAIN_INDEX[stage]))


def _validate_predecessor(
    run_state: DiagnosticRunManifest,
    stage: DiagnosticLifecycleStage,
) -> None:
    for dependency in DEPENDENCIES[stage]:
        if (
            _stage_status(run_state, dependency)
            is not DiagnosticStageStatus.VERIFIED
        ):
            raise ValueError(
                f"illegal lifecycle transition: {stage.value} requires "
                f"verified {dependency.value}",
            )


def _execute_stage(
    run_state: DiagnosticRunManifest,
    context: StageRunContext,
    stage: DiagnosticLifecycleStage,
    handlers: Mapping[DiagnosticLifecycleStage, DiagnosticStageHandler],
    max_attempts: int,
    clock: Callable[[], str],
) -> DiagnosticRunManifest:
    handler = handlers[stage]
    current = run_state
    prior = run_state.stage_state(stage)
    attempts = prior.attempts if prior is not None else 0
    while attempts < max_attempts:
        attempts += 1
        running = _replace_stage(
            current,
            DiagnosticRunStageState(
                stage=stage,
                status=DiagnosticStageStatus.RUNNING,
                attempts=attempts,
                receipt_path="",
            ),
        )
        _write_run_state(context, running)
        started = clock()
        try:
            completion = _complete_stage_attempt(
                handler,
                context,
                running,
                stage,
                attempts,
                started,
                clock,
            )
        except _StageAttemptError as error:
            current = _record_failed_attempt(
                running,
                context,
                stage,
                attempts,
                started,
                clock(),
                error.failure_code,
                error,
            )
            if error.terminal:
                return current
            continue
        current = _replace_stage(
            running,
            DiagnosticRunStageState(
                stage=stage,
                status=DiagnosticStageStatus.VERIFIED,
                attempts=attempts,
                receipt_path=_receipt_name(stage),
                outputs=completion.outputs,
            ),
        )
        _write_attempt(
            context,
            DiagnosticStageAttempt(
                run_id=context.collection_run_id,
                stage=stage,
                attempt=attempts,
                status=DiagnosticAttemptStatus.VERIFIED,
                started_at=started,
                finished_at=clock(),
            ),
        )
        _write_run_state(context, current)
        return current
    return current


class _StageAttemptError(ValueError):
    """One classified failure raised while completing a stage attempt."""

    def __init__(
        self,
        failure_code: DiagnosticAttemptFailureCode,
        detail: str,
        *,
        terminal: bool = False,
    ) -> None:
        super().__init__(detail)
        self.failure_code = failure_code
        self.terminal = terminal


def _complete_stage_attempt(
    handler: DiagnosticStageHandler,
    context: StageRunContext,
    running: DiagnosticRunManifest,
    stage: DiagnosticLifecycleStage,
    attempts: int,
    started_at: str,
    clock: Callable[[], str],
) -> StageCompletion:
    try:
        prepared = handler.prepare(context, running)
    except (OSError, ValueError) as error:
        raise _StageAttemptError(
            DiagnosticAttemptFailureCode.INPUT_PREPARATION_ERROR, str(error)
        ) from error
    try:
        completion = handler.run(context)
        _import_completion_outputs(context, completion)
    except AcceptancePreconditionError as error:
        if stage is DiagnosticLifecycleStage.ACCEPTANCE:
            raise _StageAttemptError(
                DiagnosticAttemptFailureCode.STAGE_EXECUTION_ERROR,
                str(error),
                terminal=True,
            ) from error
        raise _StageAttemptError(
            DiagnosticAttemptFailureCode.STAGE_EXECUTION_ERROR, str(error)
        ) from error
    except (OSError, ValueError) as error:
        raise _StageAttemptError(
            DiagnosticAttemptFailureCode.STAGE_EXECUTION_ERROR, str(error)
        ) from error
    receipt = _build_receipt(
        stage,
        completion,
        context,
        prepared,
        attempts,
        started_at,
        clock(),
    )
    try:
        unchanged = handler.prepare(context, running)
    except (OSError, ValueError) as error:
        raise _StageAttemptError(
            DiagnosticAttemptFailureCode.INPUT_PREPARATION_ERROR, str(error)
        ) from error
    if unchanged != prepared:
        raise _StageAttemptError(
            DiagnosticAttemptFailureCode.INPUT_IDENTITY_CHANGED,
            "stage inputs changed during execution",
        )
    try:
        if not handler.verify(context, receipt):
            raise ValueError("stage verifier rejected output inventory")
    except (OSError, ValueError) as error:
        raise _StageAttemptError(
            DiagnosticAttemptFailureCode.STAGE_VERIFICATION_FAILED, str(error)
        ) from error
    try:
        _write_receipt(context, stage, receipt)
        _commit_stage_manifests(context, completion, receipt)
    except (OSError, ValueError) as error:
        raise _StageAttemptError(
            DiagnosticAttemptFailureCode.STAGE_COMMIT_ERROR, str(error)
        ) from error
    return completion


def _record_failed_attempt(
    run_state: DiagnosticRunManifest,
    context: StageRunContext,
    stage: DiagnosticLifecycleStage,
    attempt: int,
    started_at: str,
    finished_at: str,
    failure_code: DiagnosticAttemptFailureCode,
    error: Exception,
) -> DiagnosticRunManifest:
    failed = _failed_stage(run_state, stage, attempt)
    _write_attempt(
        context,
        DiagnosticStageAttempt(
            run_id=context.collection_run_id,
            stage=stage,
            attempt=attempt,
            status=DiagnosticAttemptStatus.FAILED,
            started_at=started_at,
            finished_at=finished_at,
            failure_code=failure_code,
            detail=redacted_text_tail(str(error), limit=4096),
        ),
    )
    _write_run_state(context, failed)
    return failed


def _failed_stage(
    run_state: DiagnosticRunManifest,
    stage: DiagnosticLifecycleStage,
    attempts: int,
) -> DiagnosticRunManifest:
    return _replace_stage(
        run_state,
        DiagnosticRunStageState(
            stage=stage,
            status=DiagnosticStageStatus.FAILED,
            attempts=attempts,
            receipt_path="",
        ),
    )


def _import_completion_outputs(
    context: StageRunContext,
    completion: StageCompletion,
) -> None:
    """Import every declared output into CAS before committing a receipt."""
    if len(completion.output_paths) != len(completion.outputs):
        if completion.outputs:
            raise ValueError("stage completion omitted output paths")
        return
    store = BlobStore(context.store_root)
    for path, artifact in zip(
        completion.output_paths, completion.outputs, strict=True
    ):
        store.put_file(path, expected_sha256=artifact.sha256)


class _ManifestCommon(TypedDict):
    stage: DiagnosticLifecycleStage
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
        "stage": receipt.stage,
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
    expected = 1 if predecessor is None else predecessor.generation + 1
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
) -> tuple[GpuLifecycleIdentity, SoftwareLifecycleIdentity]:
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


def _receipt_name(stage: DiagnosticLifecycleStage) -> str:
    return f"{stage.value}.json"


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


def _reverify_past_stages(
    run_state: DiagnosticRunManifest,
    context: StageRunContext,
    handlers: Mapping[DiagnosticLifecycleStage, DiagnosticStageHandler],
) -> DiagnosticRunManifest:
    current = run_state
    invalid: set[DiagnosticLifecycleStage] = set()
    for stage in CHAIN:
        state = run_state.stage_state(stage)
        if state is None:
            invalid.add(stage)
            continue
        if state.status is not DiagnosticStageStatus.VERIFIED:
            invalid.add(stage)
            continue
        if any(dependency in invalid for dependency in DEPENDENCIES[stage]):
            invalid.add(stage)
            current = current.set_stage(
                DiagnosticRunStageState(
                    stage=state.stage,
                    status=DiagnosticStageStatus.FAILED,
                    attempts=state.attempts,
                )
            )
            continue
        receipt = _load_receipt(
            context.collection_run_id,
            state.stage,
            context.store_root,
        )
        try:
            prepared = handlers[state.stage].prepare(context, current)
        except (OSError, ValueError):
            prepared = ()
            inputs_match = False
        else:
            inputs_match = (
                receipt is not None and prepared == receipt.input_identities
            )
        verified = (
            receipt is not None
            and inputs_match
            and handlers[state.stage].verify(context, receipt)
        )
        if not verified:
            invalid.add(stage)
            current = current.set_stage(
                DiagnosticRunStageState(
                    stage=state.stage,
                    status=DiagnosticStageStatus.FAILED,
                    attempts=state.attempts,
                    receipt_path="",
                    outputs=(),
                ),
            )
    return current


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


def _stage_status(
    run_state: DiagnosticRunManifest,
    stage: DiagnosticLifecycleStage,
) -> DiagnosticStageStatus | None:
    state = run_state.stage_state(stage)
    return state.status if state is not None else None


def _immediate_predecessor(
    stage: DiagnosticLifecycleStage,
) -> DiagnosticLifecycleStage | None:
    index = _CHAIN_INDEX[stage]
    return CHAIN[index - 1] if index > 0 else None


def _next_pending(
    run_state: DiagnosticRunManifest,
) -> DiagnosticLifecycleStage | None:
    for stage in CHAIN:
        status = _stage_status(run_state, stage)
        if status not in {
            DiagnosticStageStatus.VERIFIED,
            DiagnosticStageStatus.SUPERSEDED,
        }:
            return stage
    return None


def _context_from_run_state(
    run_state: DiagnosticRunManifest,
    state_path: Path,
) -> StageRunContext:
    root = state_path.parents[2]
    plan_path = lifecycle_plan_path(run_state.collection_run_id, root)
    if (
        not plan_path.is_file()
        or sha256_file(plan_path) != run_state.plan_sha256
    ):
        raise ValueError("lifecycle run plan is missing or drifted")
    plan = load_json_file(DiagnosticLifecyclePlan, plan_path)
    if (
        plan.plan_id != run_state.plan_id
        or plan.collection_run_id != run_state.collection_run_id
    ):
        raise ValueError("lifecycle run does not match its immutable plan")
    context = build_run_context(plan=plan, store_root_path=root)
    if context.output_root is not None:
        context.paths.update(
            {
                DiagnosticLifecycleStage.MODEL_BUILD.value: (
                    context.output_root / "model-build" / "inference.json"
                ),
                DiagnosticLifecycleStage.ACCEPTANCE.value: (
                    context.output_root
                    / "acceptance"
                    / "acceptance-manifest.json"
                ),
                DiagnosticLifecycleStage.PUBLICATION.value: (
                    context.output_root / "publication"
                ),
                DiagnosticLifecycleStage.RELEASE.value: (
                    context.output_root / "release.tar.zst"
                ),
            }
        )
    return context


__all__ = [
    "CHAIN",
    "AcceptanceHandler",
    "CalibrationHandler",
    "CollectionRunHandler",
    "CorpusSnapshotHandler",
    "DesignHandler",
    "DiagnosticStageHandler",
    "ModelBuildHandler",
    "PublicationHandler",
    "ReleaseHandler",
    "StageCompletion",
    "StageRunContext",
    "build_run_context",
    "build_stage_handlers",
    "diagnostic_lifecycle_status",
    "resume_diagnostic_lifecycle",
    "run_diagnostic_lifecycle",
]
