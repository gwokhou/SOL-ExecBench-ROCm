# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Model, acceptance, publication, and release lifecycle handlers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticLifecycleStage,
)
from sol_execbench.core.bench.performance_model.lifecycle.execution import (
    StageCompletion,
    StageRunContext,
    artifact_for_path,
    verify_artifacts,
)
from sol_execbench.core.bench.performance_model.lifecycle.identity import (
    acceptance_id,
    corpus_snapshot_id,
    model_build_id,
    publication_id,
)
from sol_execbench.core.bench.performance_model.lifecycle.receipts import (
    DiagnosticStageReceipt,
)
from sol_execbench.core.bench.performance_model.lifecycle.records import (
    _now,
    _prior_receipt_stage_id,
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
    _require_output_root,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
)
from sol_execbench.core.integrity import sha256_file


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
        frozen = context.frozen_inference_profile_path
        if frozen is not None:
            expected = context.plan.frozen_inference_profile
            if (
                expected is None
                or not frozen.is_file()
                or frozen.stat().st_size != expected.size_bytes
                or sha256_file(frozen) != expected.sha256
                or sha256_file(output) != expected.sha256
            ):
                raise ValueError(
                    "model build differs from pre-frozen inference profile"
                )
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
            outputs=(artifact_for_path(output),),
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
        return verify_artifacts(
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
        outputs = (
            artifact_for_path(manifest_output),
            artifact_for_path(result_output),
        )
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
        return verify_artifacts(
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
            outputs=(artifact_for_path(manifest_path),),
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
        hardware_validation = context.hardware_validation
        if hardware_validation is None:
            raise ValueError("release requires exact-SHA hardware validation")
        attestation = package_diagnostic_publication(
            manifest_path=manifest,
            archive_output=archive,
            attestation_output=attestation_path,
            source_revision=context.source_revision,
            hardware_validation=hardware_validation,
            semantic_loader=self._semantic_loader,
            solar_verifier=self._solar_verifier,
            store_root_path=context.store_root,
            purpose=context.purpose,
        )
        context.set_output(self.stage, archive)
        return StageCompletion(
            stage_id=attestation.release_id,
            outputs=(
                artifact_for_path(archive),
                artifact_for_path(attestation_path),
            ),
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
