from __future__ import annotations

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.performance_model.lifecycle import (
    DIAGNOSTIC_LIFECYCLE_MANIFEST_ADAPTER,
    DiagnosticAcceptanceLifecycleManifest,
    DiagnosticCalibrationLifecycleManifest,
    DiagnosticCollectionRunManifest,
    DiagnosticCorpusSnapshotManifest,
    DiagnosticDesignManifest,
    DiagnosticLifecycleStage,
    DiagnosticModelBuildManifest,
    DiagnosticPublicationLifecycleManifest,
    DiagnosticReleaseLifecycleManifest,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
)

_SCHEMA_BY_STAGE = {
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


def _base(stage: DiagnosticLifecycleStage) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_BY_STAGE[stage],
        "stage": stage,
        "stage_id": "a" * 64,
        "status": DiagnosticStageStatus.VERIFIED,
        "retention_class": DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        "source_revision": "19f195a8",
        "created_at": "2026-08-07T00:00:00+00:00",
    }


def _design_payload() -> dict[str, object]:
    return {
        **_base(DiagnosticLifecycleStage.DESIGN),
        "universe_start": 160,
        "design_payload_sha256": "b" * 64,
    }


@pytest.mark.parametrize(
    ("payload", "model_type"),
    [
        (_design_payload(), DiagnosticDesignManifest),
        (
            {
                **_base(DiagnosticLifecycleStage.COLLECTION_RUN),
                "roles": ("development", "held_out"),
                "generation": 1,
            },
            DiagnosticCollectionRunManifest,
        ),
        (
            {
                **_base(DiagnosticLifecycleStage.CORPUS_SNAPSHOT),
                "role": "development",
                "corpus_file_sha256": "c" * 64,
                "case_count": 880,
            },
            DiagnosticCorpusSnapshotManifest,
        ),
        (
            {
                **_base(DiagnosticLifecycleStage.CALIBRATION),
                "calibration_profile_sha256": "0a" * 32,
                "calibration_audit_sha256": "0b" * 32,
                "gpu_identity": {
                    "gpu_architecture": "gfx1200",
                    "gpu_id": "a3ff7590-0000-1000-800f-a29c1cca1511",
                    "gpu_bdf": "0000:03:00.0",
                    "rocm_version": "7.2.0",
                    "compiler_version": "HIP version: 7.2.26015-fc0010cf6a",
                    "clock_mode": "locked",
                    "power_profile": "stable_peak",
                },
                "software_identity": {
                    "sol_version": "4.0.0",
                    "python_version": "3.13",
                },
            },
            DiagnosticCalibrationLifecycleManifest,
        ),
        (
            {
                **_base(DiagnosticLifecycleStage.MODEL_BUILD),
                "calibration_profile_sha256": "d" * 64,
                "calibration_audit_sha256": "e" * 64,
                "inference_profile_sha256": "f" * 64,
                "model_version": "gfx1200_diagnostic.v7",
            },
            DiagnosticModelBuildManifest,
        ),
        (
            {
                **_base(DiagnosticLifecycleStage.ACCEPTANCE),
                "held_out_corpus_snapshot_id": "g" * 64,
                "accepted": True,
                "verdict_sha256": "a1" * 32,
            },
            DiagnosticAcceptanceLifecycleManifest,
        ),
        (
            {
                **_base(DiagnosticLifecycleStage.PUBLICATION),
                "source_corpus_sha256": "e1" * 32,
                "publication_manifest_sha256": "b1" * 32,
                "uncompressed_size_bytes": 1234,
                "case_count": 880,
            },
            DiagnosticPublicationLifecycleManifest,
        ),
        (
            {
                **_base(DiagnosticLifecycleStage.RELEASE),
                "archive_sha256": "c1" * 32,
                "archive_size_bytes": 1234,
                "attestation_sha256": "d1" * 32,
            },
            DiagnosticReleaseLifecycleManifest,
        ),
    ],
)
def test_lifecycle_manifest_dispatch_matches_its_stage(
    payload: dict[str, object],
    model_type: type,
) -> None:
    manifest = DIAGNOSTIC_LIFECYCLE_MANIFEST_ADAPTER.validate_python(payload)
    assert isinstance(manifest, model_type)
    data = manifest.model_dump(mode="json")
    # Round-trip through the discriminator returns the same concrete type.
    again = DIAGNOSTIC_LIFECYCLE_MANIFEST_ADAPTER.validate_python(data)
    assert type(again) is type(manifest)


def test_lifecycle_manifest_rejects_wrong_schema_version() -> None:
    payload = _design_payload()
    payload["schema_version"] = "sol_execbench.diagnostic_lifecycle_release.v2"
    with pytest.raises(ValidationError, match="archive_sha256"):
        DIAGNOSTIC_LIFECYCLE_MANIFEST_ADAPTER.validate_python(payload)


def test_stage_field_must_match_schema_family() -> None:
    payload = _design_payload()
    payload["stage"] = DiagnosticLifecycleStage.RELEASE
    # The stage field is not the discriminator; construction still requires a
    # consistent, schema-current object. The design family rejects a foreign
    # stage value by a plain validation failure.
    with pytest.raises(ValidationError):
        DiagnosticDesignManifest.model_validate(payload)


def test_human_alias_never_defines_identity() -> None:
    payload = _design_payload()
    payload["human_alias"] = "cycle3"
    manifest = DiagnosticDesignManifest.model_validate(payload)
    # The alias is advisory metadata; identity stays the content-addressed
    # stage_id. An alias equal to some other stage's ID is still allowed and
    # cannot redirect identity.
    assert manifest.human_alias == "cycle3"
    assert manifest.stage_id == "a" * 64


def test_receipt_verification_marker_is_literal() -> None:
    payload = _design_payload()
    payload["receipt"] = {
        "stage": "design",
        "stage_id": "a" * 64,
        "command": "sol-execbench diagnostics lifecycle run",
        "started_at": "2026-08-07T00:00:00+00:00",
        "finished_at": "2026-08-07T00:00:01+00:00",
        "attempts": 1,
    }
    manifest = DiagnosticDesignManifest.model_validate(payload)
    assert manifest.receipt is not None
    assert manifest.receipt.verification == "receipt_verified"
