# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for the score-release packaging and verification closed loop."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_release_verifier import (
    _PROBLEM_PATH,
    _release_fixture,
    _trust_fixture,
    _write_bundle,
)

from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.scoring.release_packaging import (
    ScoreReleaseAttestation,
    collect_release_evidence,
    package_score_release,
    verify_score_release_archive,
)

_SOURCE_REVISION = "a" * 40
pytestmark = pytest.mark.requires_linux


def _verified_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    corpus, workspace = _release_fixture(tmp_path)
    _trust_fixture(monkeypatch, corpus)
    return corpus.path, _write_bundle(corpus, workspace)


def test_collect_release_evidence_excludes_unreferenced_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bundle_path = _verified_bundle(tmp_path, monkeypatch)
    workspace = bundle_path.parent

    # Files the verifier never reads: an execution plan and a trace sidecar.
    unreferenced = [
        workspace / "baseline" / "plan.json",
        workspace
        / "baseline"
        / "traces"
        / _PROBLEM_PATH
        / "trace.jsonl.profile-summary.json",
        workspace / "candidate" / "plan.json",
    ]
    for path in unreferenced:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("noise", encoding="utf-8")

    inventory = collect_release_evidence(bundle_path, bundle_root=workspace)
    paths = {item.path for item in inventory}

    for path in unreferenced:
        assert path.relative_to(workspace).as_posix() not in paths
    # The canonical bundle + corpus + statements + traces ARE included.
    assert "release-bundle.json" in paths
    assert "corpus/manifest.yaml" in paths
    assert "statements/baseline.json" in paths
    assert any(p.startswith("baseline/traces/") for p in paths)
    assert any(p.startswith("solar/manifests/") for p in paths)


def test_package_is_deterministic_and_binds_the_verified_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_manifest, bundle_path = _verified_bundle(tmp_path, monkeypatch)
    out = tmp_path / "out"
    out.mkdir()

    archive_one = out / "release1.tar.zst"
    archive_two = out / "release2.tar.zst"
    attestation_one = out / "attestation1.json"
    attestation_two = out / "attestation2.json"

    first = package_score_release(
        bundle_path=bundle_path,
        corpus_manifest_path=corpus_manifest,
        archive_output=archive_one,
        attestation_output=attestation_one,
        source_revision=_SOURCE_REVISION,
    )
    second = package_score_release(
        bundle_path=bundle_path,
        corpus_manifest_path=corpus_manifest,
        archive_output=archive_two,
        attestation_output=attestation_two,
        source_revision=_SOURCE_REVISION,
    )

    assert sha256_file(archive_one) == sha256_file(archive_two)
    assert first.release_id == second.release_id
    assert first.archive.sha256 == second.archive.sha256
    assert first.inventory_sha256 == second.inventory_sha256
    assert first.bundle_sha256 == sha256_file(bundle_path)
    assert first.source_revision == _SOURCE_REVISION
    assert first.score_authority is True
    assert first.diagnostic_only is False
    assert 0.0 <= first.official_score <= 1.0
    assert first.scored_workloads >= 1
    assert first.baseline_id == "rx9060xt-test-baseline"
    # The archive must not grow when unreferenced files are added.
    ScoreReleaseAttestation.model_validate_json(
        attestation_one.read_text(encoding="utf-8"),
    )


def test_package_archive_excludes_unreferenced_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_manifest, bundle_path = _verified_bundle(tmp_path, monkeypatch)
    workspace = bundle_path.parent
    (workspace / "baseline" / "plan.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (workspace / "baseline" / "plan.json").write_text("noise", encoding="utf-8")

    archive = tmp_path / "release.tar.zst"
    package_score_release(
        bundle_path=bundle_path,
        corpus_manifest_path=corpus_manifest,
        archive_output=archive,
        attestation_output=tmp_path / "attestation.json",
        source_revision=_SOURCE_REVISION,
    )

    import subprocess

    listing = subprocess.run(
        ["tar", "--zstd", "-tf", str(archive)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "plan.json" not in listing
    assert "release-bundle.json" in listing


def test_verify_reproduces_the_official_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_manifest, bundle_path = _verified_bundle(tmp_path, monkeypatch)
    archive = tmp_path / "release.tar.zst"
    attestation = package_score_release(
        bundle_path=bundle_path,
        corpus_manifest_path=corpus_manifest,
        archive_output=archive,
        attestation_output=tmp_path / "attestation.json",
        source_revision=_SOURCE_REVISION,
    )

    result = verify_score_release_archive(
        archive_path=archive,
        corpus_manifest_path=corpus_manifest,
        expected_sha256=attestation.archive.sha256,
    )
    assert result.suite.score == pytest.approx(attestation.official_score)
    assert result.suite.scored_workloads == attestation.scored_workloads


def test_verify_rejects_a_wrong_expected_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_manifest, bundle_path = _verified_bundle(tmp_path, monkeypatch)
    archive = tmp_path / "release.tar.zst"
    package_score_release(
        bundle_path=bundle_path,
        corpus_manifest_path=corpus_manifest,
        archive_output=archive,
        attestation_output=tmp_path / "attestation.json",
        source_revision=_SOURCE_REVISION,
    )
    with pytest.raises(ValueError, match="does not match expectation"):
        verify_score_release_archive(
            archive_path=archive,
            corpus_manifest_path=corpus_manifest,
            expected_sha256="0" * 64,
        )
