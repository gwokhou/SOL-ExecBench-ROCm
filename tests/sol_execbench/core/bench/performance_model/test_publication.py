from __future__ import annotations

from pathlib import Path

import pytest

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model import publication
from sol_execbench.core.bench.performance_model.evidence_manifest import (
    PerformanceEvidenceArtifact,
    PerformanceEvidenceArtifactKind,
    PerformanceEvidenceManifest,
    PerformanceRunIdentity,
)
from sol_execbench.core.bench.performance_model.replay_evidence import (
    PerformanceReplayEvidenceSidecar,
)
from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceSidecar,
    StaticKernelEvidenceStatus,
    StaticKernelEvidenceToolRun,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
)
from sol_execbench.core.integrity import sha256_file

_COMPLETE_ARTIFACT_KINDS = (
    PerformanceEvidenceArtifactKind.TRACE,
    PerformanceEvidenceArtifactKind.TIMING,
    PerformanceEvidenceArtifactKind.ACCESS_PATTERN,
    PerformanceEvidenceArtifactKind.PROFILE_SUMMARY,
    PerformanceEvidenceArtifactKind.STATIC_EVIDENCE,
    PerformanceEvidenceArtifactKind.COUNTER_PROVENANCE,
    PerformanceEvidenceArtifactKind.COUNTER_CSV,
    PerformanceEvidenceArtifactKind.ROCPD,
    PerformanceEvidenceArtifactKind.REPLAY_EVIDENCE,
)


def _static_evidence(path: Path) -> None:
    sidecar = StaticKernelEvidenceSidecar(
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=(StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED),
        artifacts=[
            StaticKernelEvidenceArtifact(
                artifact_id="code-object",
                artifact_type="rocm_binary",
                status=StaticKernelEvidenceStatus.COLLECTED,
                source_path="/private/build/kernel.hsaco",
                persisted_path="nested/kernel.hsaco",
                sha256="a" * 64,
                inspectable=True,
            )
        ],
        tool_runs=[
            StaticKernelEvidenceToolRun(
                tool_id="reader",
                command=["reader", "/private/build/kernel.hsaco"],
                status=StaticKernelEvidenceStatus.COLLECTED,
                stderr_tail="opened /private/build/kernel.hsaco",
            )
        ],
    )
    atomic_write_json_value(path, sidecar.to_dict())


def _performance_manifest(root: Path) -> Path:
    artifacts: list[PerformanceEvidenceArtifact] = []
    for kind in _COMPLETE_ARTIFACT_KINDS:
        path = root / f"{kind.value}.artifact"
        if kind is PerformanceEvidenceArtifactKind.STATIC_EVIDENCE:
            _static_evidence(path)
        elif kind is PerformanceEvidenceArtifactKind.REPLAY_EVIDENCE:
            replay = PerformanceReplayEvidenceSidecar(
                status=DiagnosticSidecarStatus.AVAILABLE,
                run_id="b" * 64,
                candidate_sha256="f" * 64,
                canonical_input_sha256="1" * 64,
                alignment_digest="2" * 64,
            )
            atomic_write_json_value(path, replay.to_dict())
        else:
            path.write_text(kind.value, encoding="utf-8")
        artifacts.append(
            PerformanceEvidenceArtifact(
                kind=kind,
                path=path.name,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    manifest = PerformanceEvidenceManifest(
        status=DiagnosticSidecarStatus.AVAILABLE,
        identity=PerformanceRunIdentity(
            run_id="b" * 64,
            definition="vector_add",
            definition_sha256="c" * 64,
            workload_uuid="case-0",
            workload_sha256="d" * 64,
            solution_sha256="e" * 64,
            candidate_sha256="f" * 64,
            gpu_architecture="gfx1200",
            clock_mode="locked",
            timing_protocol="device_event_v1",
        ),
        artifacts=artifacts,
    )
    path = root / "manifest.json"
    atomic_write_json_value(path, manifest.to_dict())
    return path


def test_performance_projection_verifies_then_omits_rocpd_and_private_paths(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "process" / "performance"
    source_root.mkdir(parents=True)
    source = _performance_manifest(source_root)

    output, manifest = publication._project_performance_manifest(
        source, tmp_path / "publication" / "performance"
    )

    assert not manifest.artifacts_of_kind(PerformanceEvidenceArtifactKind.ROCPD)
    assert not (output.parent / "rocpd.artifact").exists()
    static_artifact = manifest.artifact(
        PerformanceEvidenceArtifactKind.STATIC_EVIDENCE
    )
    assert static_artifact is not None
    static = load_json_file(
        StaticKernelEvidenceSidecar, output.parent / static_artifact.path
    )
    assert static.tool_runs == []
    assert static.artifacts[0].source_path is None
    assert static.artifacts[0].persisted_path is None

    (source_root / "rocpd.artifact").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match=r"(size|SHA-256) mismatch"):
        publication._project_performance_manifest(
            source, tmp_path / "other-publication" / "performance"
        )


def test_publication_and_process_directories_must_be_isolated(
    tmp_path: Path,
) -> None:
    process = tmp_path / "process"
    process.mkdir()
    inputs = tuple(
        process / name for name in ("corpus", "profile", "inference")
    )
    for path in inputs:
        path.touch()

    with pytest.raises(ValueError, match="must be isolated"):
        publication._require_new_output(process / "release", inputs)

    output = tmp_path / "publications" / "release"
    publication._require_new_output(output, inputs)
    assert output.parent.is_dir()


def test_publication_inventory_rejects_extra_files(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    root.mkdir()
    paths = (
        "development.json",
        "calibration/profile.json",
        "calibration/profile.audit.json",
        "source-inference.json",
        "inference.json",
    )
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    artifacts = publication._artifact_inventory(root)
    indexed = {item.path: item for item in artifacts}
    projection = publication.DiagnosticPublicationProjection(
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
    publication._verify_inventory(root, projection)

    unexpected = root / "unexpected.txt"
    unexpected.write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory mismatch"):
        publication._verify_inventory(root, projection)

    unexpected.unlink()
    (root / "symlink").symlink_to(root / paths[0])
    with pytest.raises(ValueError, match="contains a symlink"):
        publication._verify_inventory(root, projection)
