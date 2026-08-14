from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
from click.testing import CliRunner
from sol_execbench_type_helpers import make_hardware_validation_binding

from sol_execbench.cli.commands import diagnostics as diagnostics_commands
from sol_execbench.cli.main import cli
from sol_execbench.core.bench.performance_model import authoring
from sol_execbench.core.bench.performance_model.acceptance import (
    DiagnosticAcceptanceCase,
    DiagnosticAcceptanceManifest,
    DiagnosticAcceptanceResult,
)
from sol_execbench.core.bench.performance_model.inference import (
    InferenceObservation,
)
from sol_execbench.core.bench.performance_model.models import (
    CalibrationIdentity,
    CalibrationParameter,
    CalibrationParameterName,
    CalibrationUnit,
    DiagnosticCalibrationProfile,
    WorkloadKind,
)
from sol_execbench.core.bench.performance_model.publication import (
    DiagnosticPublicationArtifact,
    DiagnosticPublicationProjection,
)
from sol_execbench.core.bench.performance_model.release import (
    DiagnosticReleaseArchive,
    DiagnosticReleaseAttestation,
)
from sol_execbench.core.bench.performance_model.validation_corpus import (
    DiagnosticValidationCase,
    DiagnosticValidationCorpus,
    ValidationArtifactReference,
    validation_pair_id,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
)
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.solar_bridge.performance import (
    load_manifest_semantic_characterization,
)

_FAMILIES = (
    WorkloadKind.ELEMENTWISE,
    WorkloadKind.TRANSPOSE,
    WorkloadKind.REDUCTION,
    WorkloadKind.MATMUL,
    WorkloadKind.SOFTMAX,
    WorkloadKind.CROSS_ENTROPY,
    WorkloadKind.INDEXED_READ,
    WorkloadKind.INDEXED_UPDATE,
    WorkloadKind.COMPOSITE,
    WorkloadKind.TRANSFORMER,
    WorkloadKind.CONCURRENT,
)


def _corpus(
    role: Literal["development", "held_out"],
    prefix: str,
) -> DiagnosticValidationCorpus:
    return DiagnosticValidationCorpus(
        role=role,
        cases=[
            DiagnosticValidationCase(
                case_id=f"{prefix}:{kind}:{index}",
                pair_id=validation_pair_id(
                    workload_sha256=stable_json_checksum(
                        [prefix, kind, index, "workload"]
                    ),
                    candidate_sha256=stable_json_checksum(
                        [prefix, kind, index, "candidate"]
                    ),
                ),
                workload_kind=kind,
                evidence_manifest=ValidationArtifactReference(
                    path=f"{prefix}/{kind}-{index}.evidence.json",
                    sha256=stable_json_checksum(
                        [prefix, kind, index, "evidence"]
                    ),
                    size_bytes=8,
                ),
                solar_manifest=ValidationArtifactReference(
                    path=f"{prefix}/{kind}-{index}.solar.yaml",
                    sha256=stable_json_checksum([prefix, kind, index, "solar"]),
                    size_bytes=8,
                ),
            )
            for kind in _FAMILIES
            for index in range(40 if role == "development" else 20)
        ],
    )


def _calibration() -> DiagnosticCalibrationProfile:
    return DiagnosticCalibrationProfile(
        identity=CalibrationIdentity(
            gpu_architecture="gfx1200",
            gpu_id="gpu-0",
            gpu_bdf="0000:03:00.0",
            rocm_version="7.2",
            compiler_version="hipcc-7.2",
            clock_mode="locked",
            power_profile="stable_peak",
        ),
        parameters=[
            CalibrationParameter(
                name=CalibrationParameterName.DISPATCH_FLOOR_MS,
                value=0.01,
                unit=CalibrationUnit.MS,
                confidence_interval=(0.009, 0.011),
            )
        ],
        tuning_evidence_sha256=["a" * 64],
        parameter_estimation_evidence_sha256=["b" * 64],
        probe_evidence_sha256=["c" * 64],
        bootstrap_seed=1,
        bootstrap_replicates=10_000,
    )


def _development_observation(
    case: DiagnosticValidationCase,
    **_kwargs,
) -> InferenceObservation:
    action_positive = int(case.case_id.rsplit(":", maxsplit=1)[1]) < 20
    return InferenceObservation(
        case_id=case.case_id,
        workload_kind=case.workload_kind,
        measured_ms=1.0,
        base_predicted_ms=1.0,
        base_lower_ms=0.9,
        base_upper_ms=1.1,
        point_features={
            "solar_lower_bound_ms": 1.0,
            "width_64": 0.0,
            "width_128": 0.0,
            "width_256": 0.0,
            "width_512": 0.0,
            "width_1024": 0.0,
            "outer_rows_width_32": 1.0,
            "outer_rows_width_64": 0.0,
            "outer_rows_width_128": 0.0,
            "outer_rows_width_256": 0.0,
            "outer_rows_width_512": 0.0,
            "outer_rows_width_1024": 0.0,
        },
        action_scores={"wmma_missing": float(action_positive)},
        gold_action_codes=(["restore_wmma_path"] if action_positive else []),
    )


def _acceptance_case(
    case: DiagnosticValidationCase,
    **_kwargs,
) -> DiagnosticAcceptanceCase:
    action_positive = int(case.case_id.rsplit(":", maxsplit=1)[1]) < 10
    return DiagnosticAcceptanceCase(
        case_id=case.case_id,
        pair_id=case.pair_id,
        workload_kind=case.workload_kind,
        evidence_manifest_sha256=case.evidence_manifest.sha256,
        performance_diagnostic_sha256=stable_json_checksum(
            [case.case_id, "diagnostic"]
        ),
        predicted_ms=1.0,
        lower_ms=0.9,
        upper_ms=1.1,
        measured_ms=1.0,
        predicted_action_codes=(
            ["restore_wmma_path"] if action_positive else []
        ),
        gold_action_codes=(["restore_wmma_path"] if action_positive else []),
    )


def _publication(root: Path) -> DiagnosticPublicationProjection:
    paths = (
        "development.json",
        "calibration/profile.json",
        "calibration/profile.audit.json",
        "source-inference.json",
        "inference.json",
    )
    artifacts: list[DiagnosticPublicationArtifact] = []
    for relative in sorted(paths):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        artifacts.append(
            DiagnosticPublicationArtifact(
                path=relative,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    indexed = {item.path: item for item in artifacts}
    return DiagnosticPublicationProjection(
        case_count=220,
        source_corpus_sha256="a" * 64,
        corpus=indexed[paths[0]],
        calibration_profile=indexed[paths[1]],
        calibration_audit=indexed[paths[2]],
        source_inference_profile=indexed[paths[3]],
        inference_profile=indexed[paths[4]],
        artifacts=artifacts,
        uncompressed_size_bytes=sum(item.size_bytes for item in artifacts),
    )


def test_publication_projection_cli_commands(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = [tmp_path / f"input-{index}.json" for index in range(3)]
    for path in inputs:
        path.write_text("{}", encoding="utf-8")
    output = tmp_path / "publication"
    manifest_path = output / "publication.json"

    def fake_build(**kwargs) -> Path:
        assert (
            kwargs["semantic_loader"] is load_manifest_semantic_characterization
        )
        assert callable(kwargs["solar_projector"])
        assert callable(kwargs["solar_verifier"])
        projection = _publication(kwargs["output_root"])
        atomic_write_json_value(
            manifest_path, projection.model_dump(mode="json")
        )
        return manifest_path

    monkeypatch.setattr(
        diagnostics_commands,
        "build_diagnostic_publication_projection",
        fake_build,
    )
    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "diagnostics",
            "build-publication-projection",
            "--development-corpus",
            str(inputs[0]),
            "--calibration-profile",
            str(inputs[1]),
            "--source-inference-profile",
            str(inputs[2]),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["data"]["cases"] == 220

    projection = load_json_file(DiagnosticPublicationProjection, manifest_path)

    def fake_verify(_path: Path, **kwargs) -> DiagnosticPublicationProjection:
        assert (
            kwargs["semantic_loader"] is load_manifest_semantic_characterization
        )
        assert callable(kwargs["solar_verifier"])
        return projection

    monkeypatch.setattr(
        diagnostics_commands,
        "verify_diagnostic_publication_projection",
        fake_verify,
    )
    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "diagnostics",
            "verify-publication-projection",
            "--manifest",
            str(manifest_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["data"]["verified"] is True


def test_inference_and_acceptance_authoring_cli_workflow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    development_path = tmp_path / "development.json"
    held_out_path = tmp_path / "held-out.json"
    calibration_path = tmp_path / "calibration.json"
    audit_path = tmp_path / "calibration.audit.json"
    inference_path = tmp_path / "inference.json"
    acceptance_manifest_path = tmp_path / "acceptance-manifest.json"
    acceptance_path = tmp_path / "acceptance.json"
    atomic_write_json_value(
        development_path,
        _corpus("development", "dev").model_dump(mode="json"),
    )
    atomic_write_json_value(
        held_out_path,
        _corpus("held_out", "held").model_dump(mode="json"),
    )
    atomic_write_json_value(
        calibration_path,
        _calibration().model_dump(mode="json"),
    )
    atomic_write_json_value(audit_path, {"content": "content-addressed"})
    monkeypatch.setattr(
        authoring,
        "_development_observation",
        _development_observation,
    )
    monkeypatch.setattr(authoring, "_acceptance_case", _acceptance_case)

    fit_result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "diagnostics",
            "fit-performance-inference",
            "--development-corpus",
            str(development_path),
            "--calibration-profile",
            str(calibration_path),
            "--output",
            str(inference_path),
        ],
    )

    assert fit_result.exit_code == 0, fit_result.output
    assert json.loads(fit_result.output)["ok"] is True
    acceptance_result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "diagnostics",
            "accept-performance-model",
            "--development-corpus",
            str(development_path),
            "--held-out-corpus",
            str(held_out_path),
            "--calibration-profile",
            str(calibration_path),
            "--inference-profile",
            str(inference_path),
            "--manifest-output",
            str(acceptance_manifest_path),
            "--output",
            str(acceptance_path),
        ],
    )

    assert acceptance_result.exit_code == 0, acceptance_result.output
    response = json.loads(acceptance_result.output)
    assert response["ok"] is True
    assert response["data"] == {"accepted": True, "case_count": 220}
    assert acceptance_manifest_path.is_file()
    assert acceptance_path.is_file()

    manifest = load_json_file(
        DiagnosticAcceptanceManifest,
        acceptance_manifest_path,
    )
    accepted = load_json_file(DiagnosticAcceptanceResult, acceptance_path)
    authoring.verify_diagnostic_acceptance(
        acceptance=accepted,
        manifest=manifest,
        development_corpus_path=development_path,
        held_out_corpus_path=held_out_path,
        calibration_profile_path=calibration_path,
        inference_profile_path=inference_path,
        semantic_loader=load_manifest_semantic_characterization,
    )

    path_alias_case = manifest.cases[0].model_copy(
        update={"performance_diagnostic_sha256": "f" * 64}
    )
    path_alias_manifest = manifest.model_copy(
        update={"cases": [path_alias_case, *manifest.cases[1:]]}
    )
    authoring.verify_diagnostic_acceptance(
        acceptance=accepted,
        manifest=path_alias_manifest,
        development_corpus_path=development_path,
        held_out_corpus_path=held_out_path,
        calibration_profile_path=calibration_path,
        inference_profile_path=inference_path,
        semantic_loader=load_manifest_semantic_characterization,
    )

    forged_case = manifest.cases[0].model_copy(update={"measured_ms": 1.05})
    forged_manifest = manifest.model_copy(
        update={"cases": [forged_case, *manifest.cases[1:]]}
    )
    with pytest.raises(ValueError, match="source corpus evidence"):
        authoring.verify_diagnostic_acceptance(
            acceptance=accepted,
            manifest=forged_manifest,
            development_corpus_path=development_path,
            held_out_corpus_path=held_out_path,
            calibration_profile_path=calibration_path,
            inference_profile_path=inference_path,
            semantic_loader=load_manifest_semantic_characterization,
        )


def _release_attestation() -> DiagnosticReleaseAttestation:
    source_revision = "1" * 40
    return DiagnosticReleaseAttestation(
        release_id="aa" * 32,
        publication_id="ab" * 32,
        archive=DiagnosticReleaseArchive(
            name="release.tar.zst",
            sha256="ac" * 32,
            size_bytes=1024,
            publication_manifest_sha256="ad" * 32,
            source_revision=source_revision,
        ),
        uncompressed_size_bytes=2048,
        case_count=880,
        inventory_sha256="ae" * 32,
        source_revision=source_revision,
        hardware_validation=make_hardware_validation_binding(
            source_revision=source_revision,
        ),
        created_at="2026-08-07T00:00:00+00:00",
    )


def test_release_package_cli_command(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "publication.json"
    manifest.write_text("{}", encoding="utf-8")
    archive_output = tmp_path / "release.tar.zst"
    attestation_output = tmp_path / "attestation.json"
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    source_revision = "1" * 40

    def fake_package(**kwargs) -> DiagnosticReleaseAttestation:
        assert kwargs["manifest_path"] == manifest
        assert kwargs["archive_output"] == archive_output
        assert kwargs["attestation_output"] == attestation_output
        assert kwargs["source_revision"] == source_revision
        return _release_attestation()

    monkeypatch.setattr(
        diagnostics_commands,
        "package_diagnostic_publication",
        fake_package,
    )
    monkeypatch.setattr(
        diagnostics_commands,
        "verify_validation_receipt",
        lambda *args, **kwargs: make_hardware_validation_binding(
            source_revision=source_revision,
        ),
    )
    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "diagnostics",
            "release",
            "package",
            "--manifest",
            str(manifest),
            "--archive-output",
            str(archive_output),
            "--attestation-output",
            str(attestation_output),
            "--source-revision",
            source_revision,
            "--hardware-validation-receipt",
            str(receipt),
            "--hardware-evidence-dir",
            str(evidence),
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)["data"]
    assert data["release_id"] == "aa" * 32
    assert data["archive_sha256"] == "ac" * 32
    assert data["case_count"] == 880
    assert data["diagnostic_only"] is True


def test_release_verify_cli_command(tmp_path: Path, monkeypatch) -> None:
    archive = tmp_path / "release.tar.zst"
    archive.write_bytes(b"not-a-real-archive")

    def fake_verify(**kwargs) -> DiagnosticPublicationProjection:
        assert kwargs["archive_path"] == archive
        assert kwargs["expected_sha256"] is None
        return _publication(tmp_path / "publication")

    monkeypatch.setattr(
        diagnostics_commands,
        "verify_diagnostic_release_archive",
        fake_verify,
    )
    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "diagnostics",
            "release",
            "verify",
            "--archive",
            str(archive),
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)["data"]
    assert data["verified"] is True
    assert data["cases"] == 220
