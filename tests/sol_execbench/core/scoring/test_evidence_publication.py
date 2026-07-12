from __future__ import annotations

import json
from pathlib import Path

import pytest

from sol_execbench.core.scoring.release_baseline import (
    CandidateIdentity,
    EvidencePublicationManifest,
    PublishedArtifact,
    evidence_publication_manifest_from_dict,
)
from sol_execbench.core.integrity.checksums import sha256_file


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest(root: Path) -> EvidencePublicationManifest:
    solution = root / "candidate.json"
    trace = root / "candidate.trace.jsonl"
    timing = root / "candidate.timing.json"
    suite = root / "suite.json"
    baseline = root / "baseline.json"
    bound = root / "bound.json"
    hardware = root / "hardware.json"
    solution.write_text('{"name":"candidate"}\n', encoding="utf-8")
    trace.write_text(
        '{"definition":"gemm","workload":{"uuid":"w1"},'
        '"evaluation":{"status":"PASSED","performance":{"latency_ms":1.0}}}\n'
        '{"definition":"gemm","workload":{"uuid":"w2"},'
        '"evaluation":{"status":"PASSED","performance":{"latency_ms":1.1}}}\n',
        encoding="utf-8",
    )
    _write_json(timing, {"policy": "latency_ms"})
    _write_json(suite, {"workloads": [{"definition": "gemm", "workload_uuid": "w1"}]})
    _write_json(baseline, {"schema_version": "sol_execbench.scoring_baseline.v1"})
    _write_json(bound, {"schema_version": "sol_execbench.amd_sol_bound.v3"})
    _write_json(hardware, {"schema_version": "sol_execbench.amd_hardware_model.v1"})
    bundle = root / "bundle.json"
    _write_json(
        bundle,
        {
            "schema_version": "sol_execbench.release_baseline_bundle.v1",
            "release": "gfx1200-gemm-v1",
            "scope": "authority-slice:gfx1200:gemm:1-workload",
            "baseline_artifact_ref": baseline.name,
            "baseline_artifact_sha256": sha256_file(baseline),
            "suite_manifest_ref": suite.name,
            "suite_manifest_sha256": sha256_file(suite),
            "workloads": [
                {
                    "classification": "official",
                    "definition": "gemm",
                    "workload_uuid": "w1",
                    "trace_ref": trace.name,
                    "trace_sha256": sha256_file(trace),
                    "bound_ref": bound.name,
                    "bound_sha256": sha256_file(bound),
                    "hardware_model_ref": hardware.name,
                    "hardware_model_sha256": sha256_file(hardware),
                }
            ],
        },
    )
    verification = root / "verification.json"
    _write_json(
        verification,
        {
            "schema_version": "sol_execbench.release_baseline_verification.v1",
            "release": "gfx1200-gemm-v1",
            "bundle_sha256": sha256_file(bundle),
            "rerun_trace_ref": trace.name,
            "rerun_trace_sha256": sha256_file(trace),
        },
    )
    candidate = CandidateIdentity(
        solution_ref="candidate.json",
        solution_sha256=sha256_file(solution),
        trace_relative_path="candidate.trace.jsonl",
        trace_sha256=sha256_file(trace),
        timing_relative_path="candidate.timing.json",
        timing_sha256=sha256_file(timing),
    )
    official = root / "official-score.json"
    _write_json(
        official,
        {
            "schema_version": "sol_execbench.official_score_evidence.v1",
            "scope": "authority-slice:gfx1200:gemm:1-workload",
            "score_authority": True,
            "candidate_evidence": {
                "solution_sha256": candidate.solution_sha256,
                "trace_sha256": candidate.trace_sha256,
                "timing_sha256": candidate.timing_sha256,
            },
            "scores": [
                {
                    "definition": "gemm",
                    "workload_uuid": "w1",
                    "input_refs": {
                        "trace": trace.name,
                        "sol_bound": bound.name,
                        "hardware_model": hardware.name,
                    },
                }
            ],
        },
    )
    files = {
        "candidate_solution": solution,
        "candidate_trace": trace,
        "candidate_timing": timing,
        "scoring_baseline": baseline,
        "release_baseline_bundle": bundle,
        "release_baseline_verification": verification,
        "suite_manifest": suite,
        "official_score_evidence": official,
        "amd_sol_bound:w1": bound,
        "hardware_model:gfx1200": hardware,
    }
    return EvidencePublicationManifest(
        release="gfx1200-gemm-v1",
        scope="authority-slice:gfx1200:gemm:1-workload",
        source_repository="https://github.com/example/sol-execbench-rocm",
        source_revision="a" * 40,
        container_image_digest="sha256:" + "b" * 64,
        artifact_base_uri="https://github.com/example/sol-execbench-rocm/releases/download/gfx1200-gemm-v1/",
        candidate=candidate,
        artifacts=tuple(
            PublishedArtifact(role, path.name, sha256_file(path))
            for role, path in files.items()
        ),
    )


def test_publication_manifest_is_self_checksumming_and_verifies_artifacts(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    parsed = evidence_publication_manifest_from_dict(manifest.to_dict())
    assert parsed.manifest_sha256 is not None
    parsed.verify_artifact_root(tmp_path)


def test_publication_manifest_rejects_tampering_and_missing_required_roles(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = manifest.to_dict()
    payload["scope"] = "tampered"
    with pytest.raises(ValueError, match="checksum mismatch"):
        evidence_publication_manifest_from_dict(payload)

    payload = manifest.to_dict()
    payload["artifacts"] = [
        artifact
        for artifact in payload["artifacts"]
        if artifact["role"] != "official_score_evidence"
    ]
    payload["manifest_sha256"] = None
    with pytest.raises(ValueError, match="required artifact roles"):
        evidence_publication_manifest_from_dict(payload)


def test_publication_verification_rejects_changed_download(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    (tmp_path / "candidate.trace.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum_mismatch:candidate.trace.jsonl"):
        manifest.verify_artifact_root(tmp_path)


def test_publication_verification_rejects_undeclared_file(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    (tmp_path / "legacy-bound.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="undeclared files: legacy-bound.json"):
        manifest.verify_artifact_root(tmp_path)


def test_publication_stage_creates_exact_upload_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = _manifest(source)
    (source / "local-only.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "staged"

    manifest.stage_artifact_root(source, destination)

    assert not (destination / "local-only.json").exists()
    assert {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    } == {artifact.relative_path for artifact in manifest.artifacts}
    manifest.verify_artifact_root(destination)


def test_publication_stage_removes_partial_directory_after_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = _manifest(source)
    (source / "candidate.json").unlink()
    destination = tmp_path / "staged"

    with pytest.raises(ValueError, match="cannot stage missing published artifact"):
        manifest.stage_artifact_root(source, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".staged.staging-*"))


def test_publication_verification_rejects_score_reference_mismatch(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    official_path = tmp_path / "official-score.json"
    official = json.loads(official_path.read_text(encoding="utf-8"))
    official["scores"][0]["input_refs"]["sol_bound"] = "other-bound.json"
    _write_json(official_path, official)
    artifacts = tuple(
        PublishedArtifact(
            artifact.role,
            artifact.relative_path,
            sha256_file(official_path)
            if artifact.role == "official_score_evidence"
            else artifact.sha256,
        )
        for artifact in manifest.artifacts
    )
    manifest = EvidencePublicationManifest(
        release=manifest.release,
        scope=manifest.scope,
        source_repository=manifest.source_repository,
        source_revision=manifest.source_revision,
        container_image_digest=manifest.container_image_digest,
        artifact_base_uri=manifest.artifact_base_uri,
        candidate=manifest.candidate,
        artifacts=artifacts,
    )

    with pytest.raises(ValueError, match="input reference does not match bundle"):
        manifest.verify_artifact_root(tmp_path)
