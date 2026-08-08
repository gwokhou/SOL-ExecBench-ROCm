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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import TypeAdapter

if TYPE_CHECKING:
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

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticLifecycleStage,
    DiagnosticStageStatus,
)
from sol_execbench.core.bench.performance_model.lifecycle.identity import (
    acceptance_id,
    collection_run_id as derive_collection_run_id,
    corpus_snapshot_id,
    model_build_id,
    publication_id,
)
from sol_execbench.core.bench.performance_model.lifecycle.models import (
    DiagnosticDesignManifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.receipts import (
    DiagnosticStageReceipt,
)
from sol_execbench.core.bench.performance_model.lifecycle.run_state import (
    DiagnosticRunManifest,
    DiagnosticRunStageState,
    run_state_path,
    stage_receipt_path,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
    DiagnosticLifecycleParent,
)
from sol_execbench.core.bench.performance_model.lifecycle.store import (
    runs_dir,
    store_root,
)
from sol_execbench.core.bench.performance_model.lifecycle.transitions import (
    require_legal_transition,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
)
from sol_execbench.core.integrity import sha256_file

CHAIN: tuple[DiagnosticLifecycleStage, ...] = (
    DiagnosticLifecycleStage.DESIGN,
    DiagnosticLifecycleStage.COLLECTION_RUN,
    DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
    DiagnosticLifecycleStage.MODEL_BUILD,
    DiagnosticLifecycleStage.ACCEPTANCE,
    DiagnosticLifecycleStage.PUBLICATION,
    DiagnosticLifecycleStage.RELEASE,
)

_CHAIN_INDEX = {stage: index for index, stage in enumerate(CHAIN)}

_RECEIPT_ADAPTER = TypeAdapter(DiagnosticStageReceipt)


@dataclass
class StageRunContext:
    """Mutable carrier of the fixed run inputs and produced output paths."""

    store_root: Path
    design_manifest_path: Path
    collection_run_id: str
    generation: int
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

    def inputs_record(self) -> dict[str, str]:
        """Return the operational input paths persisted in the run-state."""
        record: dict[str, str] = {"source_revision": self.source_revision}
        for name, path in (
            ("corpus_root", self.corpus_root),
            ("calibration_profile", self.calibration_profile_path),
            ("calibration_audit", self.calibration_audit_path),
            ("development_corpus", self.development_corpus_path),
            ("held_out_corpus", self.held_out_corpus_path),
            ("output_root", self.output_root),
        ):
            if path is not None:
                record[name] = str(path)
        return record


@dataclass(frozen=True)
class StageCompletion:
    """One completed stage's produced identity and exact output inventory."""

    stage_id: str
    outputs: tuple[DiagnosticLifecycleArtifact, ...]


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
    for item in artifacts:
        if item.relative_path == "":
            continue
        path = (base / item.relative_path).resolve()
        if path.is_symlink() or not path.is_file():
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
            outputs=(_artifact(context.design_manifest_path),),
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
        return _verify_artifacts(
            receipt.output_inventory,
            context.design_manifest_path.parent,
        )


class CollectionRunHandler:
    """Adopt one operator-collected generation beneath a frozen design."""

    stage = DiagnosticLifecycleStage.COLLECTION_RUN

    def run(self, context: StageRunContext) -> StageCompletion:
        """Record the collected generation from its evidence tree."""
        if context.corpus_root is None:
            raise ValueError("collection run requires --corpus-root")
        evidence = context.corpus_root / "cases"
        if not evidence.is_dir():
            raise ValueError(
                f"no collected evidence tree beneath {context.corpus_root}",
            )
        return StageCompletion(stage_id=context.collection_run_id, outputs=())

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
        del receipt
        if context.corpus_root is None:
            return False
        return (context.corpus_root / "cases").is_dir()


class CorpusSnapshotHandler:
    """Validate one frozen development and held-out corpus snapshot."""

    stage = DiagnosticLifecycleStage.CORPUS_SNAPSHOT

    def run(self, context: StageRunContext) -> StageCompletion:
        """Record the frozen corpus files and derive the snapshot identity.

        The stage produces both the development and held-out corpus files.
        Its stage_id is the development snapshot identity (the canonical
        role for the chain); the held-out snapshot identity is recomputed
        deterministically by the acceptance handler from the held-out file.
        """
        if context.corpus_root is None:
            raise ValueError("corpus snapshot requires --corpus-root")
        outputs: list[DiagnosticLifecycleArtifact] = []
        digests: dict[str, str] = {}
        for role in ("development", "held_out"):
            path = context.corpus_root / f"{role}.json"
            if not path.is_file():
                raise ValueError(f"frozen {role} corpus is missing: {path}")
            artifact = _artifact(path)
            outputs.append(artifact)
            digests[role] = artifact.sha256
        digest = corpus_snapshot_id(
            collection_run_id=context.collection_run_id,
            role="development",
            corpus_sha256=digests["development"],
        )
        return StageCompletion(stage_id=digest, outputs=tuple(outputs))

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
        output = _require_output_root(context) / "inference.json"
        profile = fit_diagnostic_inference_profile(
            development_corpus_path=development,
            calibration_profile_path=calibration,
            semantic_loader=self._semantic_loader,
            blob_resolver=self._blob_resolver,
        )
        atomic_write_json_value(output, profile.model_dump(mode="json"))
        context.set_output(self.stage, output)
        stage_id = model_build_id(
            calibration_profile_sha256=sha256_file(calibration),
            calibration_audit_sha256=sha256_file(calibration_audit),
            inference_profile_sha256=sha256_file(output),
            model_version=context.model_version,
        )
        return StageCompletion(
            stage_id=stage_id,
            outputs=(_artifact(output),),
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
        if context.output_root is None:
            return False
        return _verify_artifacts(
            receipt.output_inventory,
            context.output_root,
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
        from sol_execbench.core.bench.performance_model.authoring import (
            build_diagnostic_acceptance,
        )

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
        root = _require_output_root(context)
        manifest_output = root / "acceptance-manifest.json"
        result_output = root / "acceptance.json"
        manifest, result = build_diagnostic_acceptance(
            development_corpus_path=development,
            held_out_corpus_path=held_out,
            calibration_profile_path=calibration,
            inference_profile_path=inference,
            semantic_loader=self._semantic_loader,
            blob_resolver=self._blob_resolver,
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
        )
        stage_id = acceptance_id(
            model_build_id=_prior_receipt_stage_id(
                context,
                DiagnosticLifecycleStage.MODEL_BUILD,
            ),
            held_out_corpus_snapshot_id=held_out_snapshot_id,
            accepted=result.accepted,
            verdict_sha256=sha256_file(result_output),
        )
        return StageCompletion(
            stage_id=stage_id,
            outputs=outputs,
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
        if context.output_root is None:
            return False
        return _verify_artifacts(
            receipt.output_inventory,
            context.output_root,
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
        from sol_execbench.core.bench.performance_model.publication import (
            DiagnosticPublicationProjection,
            build_diagnostic_publication_projection,
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
            source_corpus_sha256=projection.source_corpus_sha256,
            publication_manifest_sha256=sha256_file(manifest_path),
            uncompressed_size_bytes=projection.uncompressed_size_bytes,
            case_count=projection.case_count,
        )
        return StageCompletion(
            stage_id=stage_id,
            outputs=(_artifact(manifest_path),),
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
        )
        context.set_output(self.stage, archive)
        return StageCompletion(
            stage_id=attestation.release_id,
            outputs=(_artifact(archive), _artifact(attestation_path)),
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
    previous = _immediate_predecessor(stage)
    if previous is None:
        return ()
    prior = run_state.stage_state(previous)
    if prior is None or prior.receipt_path == "":
        return ()
    receipt = _load_receipt(
        context.collection_run_id,
        previous,
        context.store_root,
    )
    if receipt is None:
        return ()
    path = stage_receipt_path(
        context.collection_run_id,
        previous,
        context.store_root,
    )
    return (
        DiagnosticLifecycleParent(
            stage=previous,
            stage_id=receipt.stage_id,
            sha256=sha256_file(path),
        ),
    )


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
    design_manifest_path: Path,
    store_root_path: Path | None = None,
    corpus_root: Path | None = None,
    calibration_profile_path: Path | None = None,
    calibration_audit_path: Path | None = None,
    development_corpus_path: Path | None = None,
    held_out_corpus_path: Path | None = None,
    output_root: Path | None = None,
    source_revision: str = "unknown",
    model_version: str = "gfx1200_diagnostic.v7",
) -> StageRunContext:
    """Build the run context for a fresh lifecycle run from a design.

    The collection generation identity is derived deterministically from the
    design at generation one, matching the corpus authoring script.
    """
    design = _load_design_manual(design_manifest_path)
    root = Path(store_root_path).resolve() if store_root_path else store_root()
    return StageRunContext(
        store_root=root,
        design_manifest_path=Path(design_manifest_path).resolve(),
        collection_run_id=derive_collection_run_id(
            design_id=design.stage_id,
            generation=1,
        ),
        generation=1,
        corpus_root=corpus_root,
        calibration_profile_path=calibration_profile_path,
        calibration_audit_path=calibration_audit_path,
        development_corpus_path=development_corpus_path,
        held_out_corpus_path=held_out_corpus_path,
        output_root=output_root,
        source_revision=source_revision,
        model_version=model_version,
    )


def _base_context(
    design_manifest_path: Path,
    root: Path,
) -> StageRunContext:
    return build_run_context(
        design_manifest_path=design_manifest_path,
        store_root_path=root,
    )


def run_diagnostic_lifecycle(
    *,
    design_manifest_path: Path,
    store_root_path: Path | None = None,
    stages: Sequence[DiagnosticLifecycleStage] | None = None,
    max_attempts: int = 3,
    handlers: Mapping[
        DiagnosticLifecycleStage,
        DiagnosticStageHandler,
    ],
    context: StageRunContext | None = None,
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
    run_context = context or _base_context(design_manifest_path, root)
    design = _load_design(run_context)
    requested = tuple(stages) if stages is not None else CHAIN
    ordered = _ordered_stages(requested)
    run_state = _initial_run_state(
        design,
        run_context,
        clock(),
    )
    for stage in ordered:
        _validate_predecessor(run_state, stage)
        run_state = _execute_stage(
            run_state,
            run_context,
            stage,
            handlers,
            max_attempts,
            clock,
        )
        if _stage_status(run_state, stage) is DiagnosticStageStatus.FAILED:
            break
    _write_run_state(run_context, run_state)
    return run_state


def resume_diagnostic_lifecycle(
    *,
    run_state_path: Path,
    max_attempts: int = 3,
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
            max_attempts,
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
        "parent_chain": [
            entry["stage_id"]
            for entry in stages
            if entry["status"] == DiagnosticStageStatus.VERIFIED.value
            and entry["stage_id"] is not None
        ],
        "next_stage": next_stage.value if next_stage is not None else None,
    }
    _write_status_json(context, status)
    return status


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
        runs_dir(context.store_root) / context.collection_run_id / "status.json"
    )
    atomic_write_json_value(path, status)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_design_manual(path: Path) -> DiagnosticDesignManifest:
    return load_json_file(DiagnosticDesignManifest, path)


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
) -> DiagnosticRunManifest:
    return DiagnosticRunManifest(
        run_id=context.collection_run_id,
        collection_run_id=context.collection_run_id,
        design_id=design.stage_id,
        generation=context.generation,
        created_at=created_at,
        updated_at=created_at,
        design_manifest_path=str(context.design_manifest_path),
        inputs=context.inputs_record(),
    )


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
    previous = _immediate_predecessor(stage)
    if previous is None:
        return
    if _stage_status(run_state, previous) is not DiagnosticStageStatus.VERIFIED:
        raise ValueError(
            f"illegal lifecycle transition: {stage.value} requires "
            f"verified {previous.value}",
        )
    require_legal_transition(previous, stage)


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
    attempts = 0
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
            completion = handler.run(context)
        except (OSError, ValueError):
            current = _replace_stage(
                running,
                DiagnosticRunStageState(
                    stage=stage,
                    status=DiagnosticStageStatus.FAILED,
                    attempts=attempts,
                    receipt_path="",
                ),
            )
            _write_run_state(context, current)
            continue
        receipt = _build_receipt(
            handler,
            completion,
            context,
            running,
            attempts,
            started,
            clock(),
        )
        _write_receipt(context, stage, receipt)
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
        _write_run_state(context, current)
        return current
    return current


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
    atomic_write_json_value(
        run_state_path(context.collection_run_id, context.store_root),
        run_state.model_dump(mode="json"),
    )


def _write_receipt(
    context: StageRunContext,
    stage: DiagnosticLifecycleStage,
    receipt: DiagnosticStageReceipt,
) -> None:
    atomic_write_json_value(
        stage_receipt_path(
            context.collection_run_id,
            stage,
            context.store_root,
        ),
        _RECEIPT_ADAPTER.dump_python(receipt, mode="json"),
    )


def _receipt_name(stage: DiagnosticLifecycleStage) -> str:
    return f"{stage.value}.json"


def _build_receipt(
    handler: DiagnosticStageHandler,
    completion: StageCompletion,
    context: StageRunContext,
    run_state: DiagnosticRunManifest,
    attempts: int,
    started_at: str,
    finished_at: str,
) -> DiagnosticStageReceipt:
    return DiagnosticStageReceipt(
        stage=handler.stage,
        stage_id=completion.stage_id,
        command=f"diagnostics lifecycle {handler.stage.value}",
        started_at=started_at,
        finished_at=finished_at,
        attempts=attempts,
        input_identities=handler.prepare(context, run_state),
        output_inventory=completion.outputs,
        verification="receipt_verified",
    )


def _reverify_past_stages(
    run_state: DiagnosticRunManifest,
    context: StageRunContext,
    handlers: Mapping[DiagnosticLifecycleStage, DiagnosticStageHandler],
) -> DiagnosticRunManifest:
    current = run_state
    for state in run_state.stages:
        if state.status is not DiagnosticStageStatus.VERIFIED:
            continue
        receipt = _load_receipt(
            context.collection_run_id,
            state.stage,
            context.store_root,
        )
        verified = receipt is not None and handlers[state.stage].verify(
            context, receipt
        )
        if not verified:
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
        if status is None or status is DiagnosticStageStatus.FAILED:
            return stage
    return None


def _context_from_run_state(
    run_state: DiagnosticRunManifest,
    state_path: Path,
) -> StageRunContext:
    root = state_path.parents[2]
    design_path = (
        Path(run_state.design_manifest_path)
        if run_state.design_manifest_path
        else state_path.parents[3] / "design.json"
    )
    inputs = run_state.inputs
    return StageRunContext(
        store_root=root,
        design_manifest_path=design_path,
        collection_run_id=run_state.collection_run_id,
        generation=run_state.generation,
        corpus_root=_optional_path(inputs.get("corpus_root")),
        calibration_profile_path=_optional_path(
            inputs.get("calibration_profile")
        ),
        development_corpus_path=_optional_path(
            inputs.get("development_corpus")
        ),
        held_out_corpus_path=_optional_path(inputs.get("held_out_corpus")),
        output_root=_optional_path(inputs.get("output_root")),
        source_revision=inputs.get("source_revision", "unknown"),
    )


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


__all__ = [
    "CHAIN",
    "AcceptanceHandler",
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
