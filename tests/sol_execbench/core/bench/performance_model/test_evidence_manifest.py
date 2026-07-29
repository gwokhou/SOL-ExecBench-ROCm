from __future__ import annotations

from pathlib import Path

import pytest

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.evidence_manifest import (
    PerformanceEvidenceArtifactKind,
    PerformanceEvidenceManifest,
    PerformanceRunIdentity,
    artifact_reference,
    load_and_verify_performance_evidence_manifest,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value


def test_manifest_verifies_all_content_addressed_artifacts(
    tmp_path: Path,
) -> None:
    paths = {}
    for kind in (
        PerformanceEvidenceArtifactKind.TRACE,
        PerformanceEvidenceArtifactKind.TIMING,
        PerformanceEvidenceArtifactKind.PROFILE_SUMMARY,
        PerformanceEvidenceArtifactKind.STATIC_EVIDENCE,
    ):
        path = tmp_path / f"{kind}.json"
        path.write_text(str(kind), encoding="utf-8")
        paths[kind] = path
    manifest = PerformanceEvidenceManifest(
        status=DiagnosticSidecarStatus.PARTIAL,
        identity=PerformanceRunIdentity(
            run_id="a" * 64,
            definition="toy",
            definition_sha256="b" * 64,
            workload_uuid="w0",
            workload_sha256="c" * 64,
            solution_sha256="d" * 64,
            candidate_sha256="e" * 64,
            gpu_architecture="gfx1200",
            clock_mode="locked",
            timing_protocol="device_event_v1",
        ),
        artifacts=[
            artifact_reference(kind=kind, path=path, root=tmp_path)
            for kind, path in paths.items()
        ],
        reason_codes=["counter_evidence_missing"],
    )
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json_value(manifest_path, manifest.to_dict())

    assert (
        load_and_verify_performance_evidence_manifest(manifest_path) == manifest
    )

    paths[PerformanceEvidenceArtifactKind.TRACE].write_text(
        "tampered",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="(size|SHA-256) mismatch"):
        load_and_verify_performance_evidence_manifest(manifest_path)
