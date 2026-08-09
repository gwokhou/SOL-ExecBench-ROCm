"""Identity closure: every stage_id recomputes from its manifest inputs.

The identity functions and ``recompute_stage_id`` must agree for every
stage: a manifest whose stored ``stage_id`` was produced by an identity
function must yield the same digest when recomputed from its own fields
and cited parents. This is the single-source-of-truth contract that lets
the consistency gate detect drift.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.performance_model.lifecycle import (
    DiagnosticAcceptanceLifecycleManifest,
    DiagnosticCalibrationLifecycleManifest,
    DiagnosticCollectionRunManifest,
    DiagnosticCorpusSnapshotManifest,
    DiagnosticDesignManifest,
    DiagnosticEvidencePurpose,
    DiagnosticLifecycleParent,
    DiagnosticLifecycleStage,
    DiagnosticModelBuildManifest,
    DiagnosticPublicationLifecycleManifest,
    DiagnosticReleaseLifecycleManifest,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
    GpuLifecycleIdentity,
    SoftwareLifecycleIdentity,
    acceptance_id,
    calibration_id,
    collection_run_id,
    corpus_snapshot_id,
    design_id,
    diagnostic_lifecycle_id,
    model_build_id,
    publication_id,
    recompute_stage_id,
    release_id,
)
from sol_execbench.core.platform.runtime import (
    PCIeLinkIdentity,
    PCIeTopologyIdentity,
)

_SOURCE = "19f195a8"
_CREATED = "2026-08-07T00:00:00+00:00"

# Distinct 64-char lowercase-hex digests (the SHA256Digest validator rejects
# anything outside [0-9a-f]).
D_DESIGN_PAYLOAD = "b" * 64
D_DESIGN_PARENT = "a" * 64
D_RUN_PARENT = "2a" * 32
D_CORPUS_HELD = "1a" * 32
D_CORPUS_PROMO = "1b" * 32
D_SOURCE_ONE = "3a" * 32
D_SOURCE_TWO = "3b" * 32
D_CAL_PROFILE = "0a" * 32
D_CAL_AUDIT = "0b" * 32
D_INFER = "0c" * 32
D_VERDICT = "0d" * 32
D_MODEL_BUILD = "1c" * 32
D_SOURCE_CORPUS = "0e" * 32
D_PUB_MANIFEST = "0f" * 32
D_PUB = "1d" * 32
D_ARCHIVE = "1e" * 32
D_ATTEST = "1f" * 32

_LINK = PCIeLinkIdentity(
    bdf="0000:03:00.0",
    current_speed_gtps=32.0,
    max_speed_gtps=32.0,
    current_width=8,
    max_width=16,
)
_TOPOLOGY = PCIeTopologyIdentity(
    links=(_LINK,),
    bottleneck_bdf=_LINK.bdf,
    effective_speed_gtps=_LINK.current_speed_gtps,
    effective_width=_LINK.current_width,
)

_GPU = GpuLifecycleIdentity(
    gpu_architecture="gfx1200",
    gpu_id="a3ff7590-0000-1000-800f-a29c1cca1511",
    gpu_bdf="0000:03:00.0",
    pcie_topology=_TOPOLOGY,
    rocm_version="7.2.0",
    compiler_version="HIP version: 7.2.26015-fc0010cf6a",
    clock_mode="locked",
    power_profile="stable_peak",
)
_SW = SoftwareLifecycleIdentity(sol_version="4.0.0", python_version="3.13")


_SCHEMA_BY_STAGE: dict[DiagnosticLifecycleStage, str] = {
    DiagnosticLifecycleStage.DESIGN: "sol_execbench.diagnostic_lifecycle_design.v2",
    DiagnosticLifecycleStage.COLLECTION_RUN: (
        "sol_execbench.diagnostic_lifecycle_collection_run.v2"
    ),
    DiagnosticLifecycleStage.CORPUS_SNAPSHOT: (
        "sol_execbench.diagnostic_lifecycle_corpus_snapshot.v2"
    ),
    DiagnosticLifecycleStage.CALIBRATION: (
        "sol_execbench.diagnostic_lifecycle_calibration.v2"
    ),
    DiagnosticLifecycleStage.MODEL_BUILD: (
        "sol_execbench.diagnostic_lifecycle_model_build.v2"
    ),
    DiagnosticLifecycleStage.ACCEPTANCE: (
        "sol_execbench.diagnostic_lifecycle_acceptance.v2"
    ),
    DiagnosticLifecycleStage.PUBLICATION: (
        "sol_execbench.diagnostic_lifecycle_publication.v2"
    ),
    DiagnosticLifecycleStage.RELEASE: (
        "sol_execbench.diagnostic_lifecycle_release.v2"
    ),
}


def _manifest_data(
    stage: DiagnosticLifecycleStage,
    stage_id: str,
    **extra: object,
) -> dict[str, object]:
    """Return a base manifest dict merged with stage-specific fields."""
    return {
        "stage": stage,
        "stage_id": stage_id,
        "schema_version": _SCHEMA_BY_STAGE[stage],
        "status": DiagnosticStageStatus.VERIFIED,
        "retention_class": DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        "source_revision": _SOURCE,
        "created_at": _CREATED,
        **extra,
    }


def _parent(
    stage: DiagnosticLifecycleStage, stage_id: str
) -> DiagnosticLifecycleParent:
    return DiagnosticLifecycleParent(
        stage=stage,
        stage_id=stage_id,
        sha256="0" * 64,
    )


def test_design_identity_recomputes() -> None:
    did = design_id(
        universe_start=160,
        design_payload_sha256=D_DESIGN_PAYLOAD,
        source_revision=_SOURCE,
    )
    manifest = DiagnosticDesignManifest.model_validate(
        _manifest_data(
            DiagnosticLifecycleStage.DESIGN,
            did,
            universe_start=160,
            design_payload_sha256=D_DESIGN_PAYLOAD,
        ),
    )
    assert recompute_stage_id(manifest) == did


def test_collection_run_identity_recomputes() -> None:
    cid = collection_run_id(
        design_id=D_DESIGN_PARENT,
        generation=1,
        source_revision=_SOURCE,
    )
    manifest = DiagnosticCollectionRunManifest.model_validate(
        _manifest_data(
            DiagnosticLifecycleStage.COLLECTION_RUN,
            cid,
            generation=1,
            parents=(
                _parent(DiagnosticLifecycleStage.DESIGN, D_DESIGN_PARENT),
            ),
        ),
    )
    assert recompute_stage_id(manifest) == cid


def test_collection_run_identity_binds_pcie_topology() -> None:
    cid = collection_run_id(
        design_id=D_DESIGN_PARENT,
        generation=1,
        gpu_identity=_GPU,
        source_revision=_SOURCE,
    )
    manifest = DiagnosticCollectionRunManifest.model_validate(
        _manifest_data(
            DiagnosticLifecycleStage.COLLECTION_RUN,
            cid,
            generation=1,
            gpu_identity=_GPU,
            parents=(
                _parent(DiagnosticLifecycleStage.DESIGN, D_DESIGN_PARENT),
            ),
        ),
    )

    assert recompute_stage_id(manifest) == cid
    assert cid != collection_run_id(
        design_id=D_DESIGN_PARENT,
        generation=1,
        source_revision=_SOURCE,
    )


def test_calibration_identity_preserves_legacy_gpu_without_pcie() -> None:
    legacy_gpu = _GPU.model_copy(update={"pcie_topology": None})
    legacy_payload = legacy_gpu.model_dump(mode="json")
    legacy_payload.pop("pcie_topology")

    observed = calibration_id(
        calibration_profile_sha256=D_CAL_PROFILE,
        calibration_audit_sha256=D_CAL_AUDIT,
        gpu_identity=legacy_gpu,
        software_identity=_SW,
        source_revision=_SOURCE,
    )
    expected = diagnostic_lifecycle_id(
        "calibration",
        {
            "calibration_profile_sha256": D_CAL_PROFILE,
            "calibration_audit_sha256": D_CAL_AUDIT,
            "gpu_identity": legacy_payload,
            "software_identity": _SW.model_dump(mode="json"),
            "purpose": DiagnosticEvidencePurpose.PRODUCTION,
            "source_revision": _SOURCE,
        },
    )

    assert observed == expected


def test_corpus_snapshot_direct_identity_recomputes() -> None:
    sid = corpus_snapshot_id(
        collection_run_id=D_RUN_PARENT,
        role="held_out",
        corpus_sha256=D_CORPUS_HELD,
        source_revision=_SOURCE,
    )
    manifest = DiagnosticCorpusSnapshotManifest.model_validate(
        _manifest_data(
            DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
            sid,
            role="held_out",
            corpus_file_sha256=D_CORPUS_HELD,
            case_count=220,
            parents=(
                _parent(DiagnosticLifecycleStage.COLLECTION_RUN, D_RUN_PARENT),
            ),
        ),
    )
    assert recompute_stage_id(manifest) == sid


def test_corpus_snapshot_promoted_identity_recomputes() -> None:
    sid = corpus_snapshot_id(
        role="development",
        corpus_sha256=D_CORPUS_PROMO,
        source_snapshot_ids=(D_SOURCE_ONE, D_SOURCE_TWO),
        source_revision=_SOURCE,
    )
    manifest = DiagnosticCorpusSnapshotManifest.model_validate(
        _manifest_data(
            DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
            sid,
            role="development",
            corpus_file_sha256=D_CORPUS_PROMO,
            case_count=880,
            source_snapshot_ids=(D_SOURCE_ONE, D_SOURCE_TWO),
            parents=(
                _parent(DiagnosticLifecycleStage.CORPUS_SNAPSHOT, D_SOURCE_ONE),
                _parent(DiagnosticLifecycleStage.CORPUS_SNAPSHOT, D_SOURCE_TWO),
            ),
        ),
    )
    assert recompute_stage_id(manifest) == sid


def test_promoted_snapshot_is_distinct_from_direct_child() -> None:
    """Promotion must not collapse into the direct-collection identity."""
    direct = corpus_snapshot_id(
        collection_run_id=D_RUN_PARENT,
        role="development",
        corpus_sha256=D_CORPUS_PROMO,
        source_revision=_SOURCE,
    )
    promoted = corpus_snapshot_id(
        role="development",
        corpus_sha256=D_CORPUS_PROMO,
        source_snapshot_ids=(D_RUN_PARENT,),
        source_revision=_SOURCE,
    )
    assert direct != promoted


def test_corpus_snapshot_rejects_both_kinds() -> None:
    with pytest.raises(ValueError, match="not both"):
        corpus_snapshot_id(
            collection_run_id=D_RUN_PARENT,
            role="development",
            corpus_sha256=D_CORPUS_PROMO,
            source_snapshot_ids=(D_SOURCE_ONE,),
            source_revision=_SOURCE,
        )


def test_calibration_identity_recomputes_and_binds_hardware() -> None:
    cal = calibration_id(
        calibration_profile_sha256=D_CAL_PROFILE,
        calibration_audit_sha256=D_CAL_AUDIT,
        gpu_identity=_GPU,
        software_identity=_SW,
        source_revision=_SOURCE,
    )
    manifest = DiagnosticCalibrationLifecycleManifest.model_validate(
        _manifest_data(
            DiagnosticLifecycleStage.CALIBRATION,
            cal,
            calibration_profile_sha256=D_CAL_PROFILE,
            calibration_audit_sha256=D_CAL_AUDIT,
            gpu_identity=_GPU,
            software_identity=_SW,
        ),
    )
    assert recompute_stage_id(manifest) == cal


def test_calibration_rejects_partial_gpu_identity() -> None:
    partial_gpu = GpuLifecycleIdentity(gpu_architecture="gfx1200")
    with pytest.raises(ValidationError, match="gpu_identity is missing"):
        DiagnosticCalibrationLifecycleManifest.model_validate(
            _manifest_data(
                DiagnosticLifecycleStage.CALIBRATION,
                D_CAL_PROFILE,
                calibration_profile_sha256=D_CAL_PROFILE,
                calibration_audit_sha256=D_CAL_AUDIT,
                gpu_identity=partial_gpu,
                software_identity=_SW,
            ),
        )


def test_model_build_identity_recomputes() -> None:
    mb = model_build_id(
        calibration_id=D_CAL_PROFILE,
        development_snapshot_id=D_CORPUS_PROMO,
        calibration_profile_sha256=D_CAL_PROFILE,
        calibration_audit_sha256=D_CAL_AUDIT,
        inference_profile_sha256=D_INFER,
        model_version="gfx1200_diagnostic.v7",
        source_revision=_SOURCE,
    )
    manifest = DiagnosticModelBuildManifest.model_validate(
        _manifest_data(
            DiagnosticLifecycleStage.MODEL_BUILD,
            mb,
            calibration_profile_sha256=D_CAL_PROFILE,
            calibration_audit_sha256=D_CAL_AUDIT,
            inference_profile_sha256=D_INFER,
            model_version="gfx1200_diagnostic.v7",
            parents=(
                _parent(DiagnosticLifecycleStage.CALIBRATION, D_CAL_PROFILE),
                _parent(
                    DiagnosticLifecycleStage.CORPUS_SNAPSHOT, D_CORPUS_PROMO
                ),
            ),
        ),
    )
    assert recompute_stage_id(manifest) == mb


def test_acceptance_identity_recomputes() -> None:
    aid = acceptance_id(
        calibration_id=D_CAL_PROFILE,
        development_snapshot_id=D_CORPUS_PROMO,
        model_build_id=D_MODEL_BUILD,
        held_out_corpus_snapshot_id=D_CORPUS_HELD,
        accepted=True,
        verdict_sha256=D_VERDICT,
        source_revision=_SOURCE,
    )
    manifest = DiagnosticAcceptanceLifecycleManifest.model_validate(
        _manifest_data(
            DiagnosticLifecycleStage.ACCEPTANCE,
            aid,
            held_out_corpus_snapshot_id=D_CORPUS_HELD,
            accepted=True,
            verdict_sha256=D_VERDICT,
            parents=(
                _parent(DiagnosticLifecycleStage.MODEL_BUILD, D_MODEL_BUILD),
                _parent(DiagnosticLifecycleStage.CALIBRATION, D_CAL_PROFILE),
                _parent(
                    DiagnosticLifecycleStage.CORPUS_SNAPSHOT, D_CORPUS_PROMO
                ),
                _parent(
                    DiagnosticLifecycleStage.CORPUS_SNAPSHOT, D_CORPUS_HELD
                ),
            ),
        ),
    )
    assert recompute_stage_id(manifest) == aid


def test_publication_identity_recomputes() -> None:
    pid = publication_id(
        acceptance_id=D_VERDICT,
        calibration_id=D_CAL_PROFILE,
        development_snapshot_id=D_CORPUS_PROMO,
        model_build_id=D_MODEL_BUILD,
        source_corpus_sha256=D_SOURCE_CORPUS,
        publication_manifest_sha256=D_PUB_MANIFEST,
        uncompressed_size_bytes=1234,
        case_count=880,
        source_revision=_SOURCE,
    )
    manifest = DiagnosticPublicationLifecycleManifest.model_validate(
        _manifest_data(
            DiagnosticLifecycleStage.PUBLICATION,
            pid,
            source_corpus_sha256=D_SOURCE_CORPUS,
            publication_manifest_sha256=D_PUB_MANIFEST,
            uncompressed_size_bytes=1234,
            case_count=880,
            parents=(
                _parent(DiagnosticLifecycleStage.ACCEPTANCE, D_VERDICT),
                _parent(DiagnosticLifecycleStage.CALIBRATION, D_CAL_PROFILE),
                _parent(
                    DiagnosticLifecycleStage.CORPUS_SNAPSHOT, D_CORPUS_PROMO
                ),
                _parent(DiagnosticLifecycleStage.MODEL_BUILD, D_MODEL_BUILD),
            ),
        ),
    )
    assert recompute_stage_id(manifest) == pid


def test_release_identity_recomputes() -> None:
    rid = release_id(
        publication_id=D_PUB,
        archive_sha256=D_ARCHIVE,
        source_revision=_SOURCE,
        producer_version="4.0.0",
        archive_size_bytes=99,
    )
    manifest = DiagnosticReleaseLifecycleManifest.model_validate(
        _manifest_data(
            DiagnosticLifecycleStage.RELEASE,
            rid,
            archive_sha256=D_ARCHIVE,
            archive_size_bytes=99,
            attestation_sha256=D_ATTEST,
            producer_version="4.0.0",
            parents=(_parent(DiagnosticLifecycleStage.PUBLICATION, D_PUB),),
        ),
    )
    assert recompute_stage_id(manifest) == rid


def test_source_revision_changes_every_stage_identity() -> None:
    """No immutable node may be reused across producer source revisions."""
    revisions = ("a" * 40, "b" * 40)
    assert (
        len(
            {
                design_id(
                    universe_start=0,
                    design_payload_sha256=D_DESIGN_PAYLOAD,
                    source_revision=revision,
                )
                for revision in revisions
            }
        )
        == 2
    )
    assert (
        len(
            {
                calibration_id(
                    calibration_profile_sha256=D_CAL_PROFILE,
                    calibration_audit_sha256=D_CAL_AUDIT,
                    gpu_identity=_GPU,
                    software_identity=_SW,
                    source_revision=revision,
                )
                for revision in revisions
            }
        )
        == 2
    )
    assert (
        len(
            {
                collection_run_id(
                    design_id=D_DESIGN_PARENT,
                    generation=1,
                    source_revision=revision,
                )
                for revision in revisions
            }
        )
        == 2
    )
    assert (
        len(
            {
                corpus_snapshot_id(
                    collection_run_id=D_RUN_PARENT,
                    role="held_out",
                    corpus_sha256=D_CORPUS_HELD,
                    source_revision=revision,
                )
                for revision in revisions
            }
        )
        == 2
    )
    assert (
        len(
            {
                model_build_id(
                    calibration_id=D_CAL_PROFILE,
                    development_snapshot_id=D_CORPUS_PROMO,
                    calibration_profile_sha256=D_CAL_PROFILE,
                    calibration_audit_sha256=D_CAL_AUDIT,
                    inference_profile_sha256=D_INFER,
                    model_version="gfx1200_diagnostic.v7",
                    source_revision=revision,
                )
                for revision in revisions
            }
        )
        == 2
    )
    assert (
        len(
            {
                acceptance_id(
                    calibration_id=D_CAL_PROFILE,
                    development_snapshot_id=D_CORPUS_PROMO,
                    model_build_id=D_MODEL_BUILD,
                    held_out_corpus_snapshot_id=D_CORPUS_HELD,
                    accepted=True,
                    verdict_sha256=D_VERDICT,
                    source_revision=revision,
                )
                for revision in revisions
            }
        )
        == 2
    )
    assert (
        len(
            {
                publication_id(
                    acceptance_id=D_VERDICT,
                    calibration_id=D_CAL_PROFILE,
                    development_snapshot_id=D_CORPUS_PROMO,
                    model_build_id=D_MODEL_BUILD,
                    source_corpus_sha256=D_SOURCE_CORPUS,
                    publication_manifest_sha256=D_PUB_MANIFEST,
                    uncompressed_size_bytes=1234,
                    case_count=880,
                    source_revision=revision,
                )
                for revision in revisions
            }
        )
        == 2
    )
    assert (
        len(
            {
                release_id(
                    publication_id=D_PUB,
                    archive_sha256=D_ARCHIVE,
                    source_revision=revision,
                    producer_version="4.0.0",
                    archive_size_bytes=99,
                )
                for revision in revisions
            }
        )
        == 2
    )
