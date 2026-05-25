from __future__ import annotations

import hashlib
import shutil

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.static_kernel_evidence import (
    STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION,
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceClassification,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceSidecar,
    StaticKernelEvidenceStatus,
    StaticKernelEvidenceToolRun,
    build_static_kernel_evidence_failed,
    build_static_kernel_evidence_partial,
    build_static_kernel_evidence_sidecar,
    build_static_kernel_evidence_skipped,
    build_static_kernel_evidence_unavailable,
    build_static_kernel_evidence_unsupported,
    collect_static_kernel_artifacts,
)


EXPECTED_STATUSES = {
    "collected",
    "partial",
    "unavailable",
    "unsupported",
    "failed",
    "skipped",
}

EXPECTED_REASON_CODES = {
    "static_evidence_not_requested",
    "static_evidence_collected",
    "partial_artifact_metadata",
    "partial_disassembly_only",
    "partial_metadata_only",
    "artifact_unavailable",
    "toolchain_unavailable",
    "unsupported_solution_type",
    "unsupported_architecture",
    "unsupported_artifact_type",
    "extractor_failed",
    "extractor_timeout",
    "parser_failed",
}


def _representative_sidecar() -> StaticKernelEvidenceSidecar:
    return build_static_kernel_evidence_sidecar(
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        artifacts=[
            StaticKernelEvidenceArtifact(
                artifact_id="kernel-object",
                artifact_type="elf_object",
                status=StaticKernelEvidenceStatus.COLLECTED,
                reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
                source_path="build/kernel.o",
                size_bytes=4096,
                classification=StaticKernelEvidenceClassification(
                    metadata_present=True,
                    disassembly_present=True,
                    detected_architectures=["gfx1200", "gfx942"],
                    symbol_count=3,
                ),
            )
        ],
        tool_runs=[
            StaticKernelEvidenceToolRun(
                tool_id="llvm-objdump",
                command=["llvm-objdump", "--disassemble", "kernel.o"],
                status=StaticKernelEvidenceStatus.COLLECTED,
                reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
                returncode=0,
                stdout_tail="kernel symbols",
            )
        ],
    )


def test_static_kernel_evidence_round_trips_strict_json_payload():
    sidecar = _representative_sidecar()
    payload = sidecar.model_dump(mode="json")

    assert payload["schema_version"] == STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION
    assert payload["schema_version"] == "sol_execbench.static_kernel_evidence.v1"
    assert StaticKernelEvidenceSidecar.model_validate(payload) == sidecar


def test_static_kernel_evidence_rejects_unknown_top_level_and_nested_fields():
    payload = _representative_sidecar().model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        StaticKernelEvidenceSidecar.model_validate(payload)

    nested_payload = _representative_sidecar().model_dump(mode="json")
    nested_payload["artifacts"][0]["unexpected"] = True

    with pytest.raises(ValidationError):
        StaticKernelEvidenceSidecar.model_validate(nested_payload)


def test_status_and_reason_code_vocabularies_are_locked():
    assert {status.value for status in StaticKernelEvidenceStatus} == EXPECTED_STATUSES
    assert {reason.value for reason in StaticKernelEvidenceReasonCode} == (
        EXPECTED_REASON_CODES
    )


def test_authority_fields_are_diagnostic_only_and_false_for_benchmark_truth():
    payload = _representative_sidecar().model_dump(mode="json")

    assert payload["diagnostic_only"] is True
    assert payload["correctness_authority"] is False
    assert payload["performance_authority"] is False
    assert payload["timing_authority"] is False
    assert payload["score_authority"] is False
    assert payload["paper_parity_authority"] is False
    assert payload["leaderboard_authority"] is False


@pytest.mark.parametrize(
    ("builder", "status", "reason_code"),
    [
        (
            build_static_kernel_evidence_skipped,
            "skipped",
            "static_evidence_not_requested",
        ),
        (
            build_static_kernel_evidence_unavailable,
            "unavailable",
            "toolchain_unavailable",
        ),
        (
            build_static_kernel_evidence_unsupported,
            "unsupported",
            "unsupported_solution_type",
        ),
        (build_static_kernel_evidence_failed, "failed", "extractor_failed"),
        (build_static_kernel_evidence_partial, "partial", "partial_artifact_metadata"),
    ],
)
def test_non_collected_helpers_return_full_sidecars_with_stable_empty_sections(
    builder,
    status,
    reason_code,
):
    sidecar = builder()
    payload = sidecar.model_dump(mode="json")

    assert StaticKernelEvidenceSidecar.model_validate(payload) == sidecar
    assert payload["status"] == status
    assert payload["reason_code"] == reason_code
    assert payload["artifacts"] == []
    assert payload["tool_runs"] == []
    assert payload["kernels"] == []
    assert payload["warnings"] == []
    assert payload["source_references"] == []


def test_artifact_entries_carry_per_artifact_status_reason_and_classification():
    payload = _representative_sidecar().model_dump(mode="json")
    artifact = payload["artifacts"][0]
    classification = artifact["classification"]

    assert artifact["status"] == "collected"
    assert artifact["reason_code"] == "static_evidence_collected"
    assert classification["metadata_present"] is True
    assert classification["disassembly_present"] is True
    assert classification["detected_architectures"] == ["gfx1200", "gfx942"]
    assert classification["symbol_count"] == 3


def test_static_artifact_collection_persists_current_build_manifest(tmp_path):
    build_dir = tmp_path / "staging"
    evidence_dir = tmp_path / "evidence"
    build_dir.mkdir()
    files = {
        "benchmark_kernel.so": b"shared",
        "kernel.hsaco": b"hsaco",
        "kernel.co": b"code-object",
        "obj/kernel.o": b"object",
        "build.log": b"compile output",
    }
    for relative_path, content in files.items():
        path = build_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (build_dir / "notes.md").write_text("not a static artifact")

    sidecar = collect_static_kernel_artifacts(
        build_directory=build_dir,
        evidence_directory=evidence_dir,
        target_architecture="gfx1200",
    )
    payload = sidecar.model_dump(mode="json")

    assert payload["status"] == "collected"
    assert payload["reason_code"] == "static_evidence_collected"
    artifacts = {artifact["source_path"]: artifact for artifact in payload["artifacts"]}
    assert set(artifacts) == set(files)
    assert artifacts["benchmark_kernel.so"]["artifact_type"] == "shared_library"
    assert artifacts["kernel.hsaco"]["artifact_type"] == "hsaco"
    assert artifacts["kernel.co"]["artifact_type"] == "code_object"
    assert artifacts["obj/kernel.o"]["artifact_type"] == "object_file"
    assert artifacts["build.log"]["artifact_type"] == "compiler_output"

    for source_path, content in files.items():
        artifact = artifacts[source_path]
        persisted_path = evidence_dir / artifact["persisted_path"]
        assert persisted_path.read_bytes() == content
        assert artifact["sha256"] == hashlib.sha256(content).hexdigest()
        assert artifact["size_bytes"] == len(content)
        assert artifact["producer"] == "hip_cpp_build"
        assert artifact["target_architecture"] == "gfx1200"
        assert artifact["classification"]["metadata_present"] is True
        assert artifact["classification"]["detected_architectures"] == ["gfx1200"]
        assert artifact["persisted_path"].startswith("artifacts/")

    assert artifacts["benchmark_kernel.so"]["inspectable"] is True
    assert artifacts["build.log"]["inspectable"] is False


def test_static_artifact_collection_survives_staging_cleanup(tmp_path):
    build_dir = tmp_path / "staging"
    evidence_dir = tmp_path / "evidence"
    build_dir.mkdir()
    (build_dir / "benchmark_kernel.so").write_bytes(b"shared")

    sidecar = collect_static_kernel_artifacts(
        build_directory=build_dir,
        evidence_directory=evidence_dir,
    )
    persisted_path = evidence_dir / sidecar.artifacts[0].persisted_path

    shutil.rmtree(build_dir)

    assert persisted_path.read_bytes() == b"shared"


def test_static_artifact_collection_ignores_outside_files_and_symlink_escapes(
    tmp_path,
):
    build_dir = tmp_path / "staging"
    evidence_dir = tmp_path / "evidence"
    outside_dir = tmp_path / "outside"
    build_dir.mkdir()
    outside_dir.mkdir()
    (build_dir / "benchmark_kernel.so").write_bytes(b"shared")
    (outside_dir / "kernel.hsaco").write_bytes(b"outside")
    (build_dir / "escape.hsaco").symlink_to(outside_dir / "kernel.hsaco")

    sidecar = collect_static_kernel_artifacts(
        build_directory=build_dir,
        evidence_directory=evidence_dir,
    )
    source_paths = {artifact.source_path for artifact in sidecar.artifacts}

    assert source_paths == {"benchmark_kernel.so"}
    assert not (evidence_dir / "artifacts" / "escape.hsaco").exists()


def test_static_artifact_collection_skips_nested_evidence_directory(tmp_path):
    build_dir = tmp_path / "staging"
    evidence_dir = build_dir / "static-evidence"
    build_dir.mkdir()
    (build_dir / "benchmark_kernel.so").write_bytes(b"shared")
    (evidence_dir / "artifacts").mkdir(parents=True)
    (evidence_dir / "artifacts" / "stale.hsaco").write_bytes(b"stale")

    sidecar = collect_static_kernel_artifacts(
        build_directory=build_dir,
        evidence_directory=evidence_dir,
        sidecar_base_directory=evidence_dir,
    )

    assert [artifact.source_path for artifact in sidecar.artifacts] == [
        "benchmark_kernel.so"
    ]


def test_static_artifact_collection_reports_unavailable_without_primary_artifact(
    tmp_path,
):
    build_dir = tmp_path / "staging"
    evidence_dir = tmp_path / "evidence"
    build_dir.mkdir()
    (build_dir / "kernel.hsaco").write_bytes(b"orphan")

    sidecar = collect_static_kernel_artifacts(
        build_directory=build_dir,
        evidence_directory=evidence_dir,
    )
    payload = sidecar.model_dump(mode="json")

    assert payload["status"] == "unavailable"
    assert payload["reason_code"] == "artifact_unavailable"
    assert payload["artifacts"] == []
    assert not evidence_dir.exists()
