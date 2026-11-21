# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for the score-release packaging and verification closed loop."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import zstandard
from sol_execbench_type_helpers import make_hardware_validation_binding
from test_release_verifier import (
    _PROBLEM_PATH,
    _release_fixture,
    _trust_fixture,
    _write_bundle,
)

from sol_execbench.core.data.json_utils import (
    atomic_write_jsonl_values,
    load_jsonl_file,
)
from sol_execbench.core.data.trace import Evaluation, EvaluationStatus, Trace
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.scoring import release_packaging
from sol_execbench.core.scoring.formula import SOLScoreAuditError
from sol_execbench.core.scoring.release_packaging import (
    ScoreReleaseAttestation,
    collect_release_evidence,
    package_score_release,
    verify_score_release_archive,
)
from sol_execbench.core.scoring.release_verifier import verify_and_score_release

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
        hardware_validation=make_hardware_validation_binding(),
    )
    second = package_score_release(
        bundle_path=bundle_path,
        corpus_manifest_path=corpus_manifest,
        archive_output=archive_two,
        attestation_output=attestation_two,
        source_revision=_SOURCE_REVISION,
        hardware_validation=make_hardware_validation_binding(),
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
        hardware_validation=make_hardware_validation_binding(),
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
        hardware_validation=make_hardware_validation_binding(),
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
        hardware_validation=make_hardware_validation_binding(),
    )
    with pytest.raises(ValueError, match="does not match expectation"):
        verify_score_release_archive(
            archive_path=archive,
            corpus_manifest_path=corpus_manifest,
            expected_sha256="0" * 64,
        )


def _write_custom_zstd_tar(
    archive_path: Path,
    members: list[tuple[tarfile.TarInfo, bytes | None]],
) -> None:
    compressor = zstandard.ZstdCompressor(level=3)
    with (
        archive_path.open("wb") as out_file,
        compressor.stream_writer(out_file, closefd=False) as zstd_stream,
        tarfile.open(
            fileobj=zstd_stream, mode="w|", format=tarfile.PAX_FORMAT
        ) as tar,
    ):
        for info, data in members:
            if data is None:
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


@pytest.mark.parametrize(
    ("scenario", "expected_match"),
    [
        ("symlink", "unsupported release archive member"),
        ("hardlink", "unsupported release archive member"),
        ("fifo", "unsupported release archive member"),
        ("traversal", "unsafe release archive member"),
        ("absolute", "unsafe release archive member"),
        ("duplicate", "duplicate release archive member"),
    ],
)
def test_verify_rejects_malicious_archive_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    expected_match: str,
) -> None:
    corpus_manifest, _ = _verified_bundle(tmp_path, monkeypatch)
    archive = tmp_path / f"{scenario}.tar.zst"
    members: list[tuple[tarfile.TarInfo, bytes | None]] = []
    if scenario == "symlink":
        info = tarfile.TarInfo("release/escape")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        members.append((info, None))
    elif scenario == "hardlink":
        info = tarfile.TarInfo("release/escape")
        info.type = tarfile.LNKTYPE
        info.linkname = "/etc/passwd"
        members.append((info, None))
    elif scenario == "fifo":
        info = tarfile.TarInfo("release/pipe")
        info.type = tarfile.FIFOTYPE
        members.append((info, None))
    elif scenario == "traversal":
        info = tarfile.TarInfo("release/../../escape.txt")
        members.append((info, b"bad"))
    elif scenario == "absolute":
        info = tarfile.TarInfo("/etc/escape.txt")
        members.append((info, b"bad"))
    elif scenario == "duplicate":
        first = tarfile.TarInfo("release/file.json")
        second = tarfile.TarInfo("release/file.json")
        members.extend([(first, b"{}"), (second, b"{}")])

    _write_custom_zstd_tar(archive, members)
    unpack_dir = tmp_path / "unpacked"
    with pytest.raises(ValueError, match=expected_match):
        verify_score_release_archive(
            archive_path=archive,
            corpus_manifest_path=corpus_manifest,
            unpack_root=unpack_dir,
        )


def test_verify_rejects_decompression_bomb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_manifest, _ = _verified_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(release_packaging, "_MAX_EXPANDED_BYTES", 4096)
    archive = tmp_path / "bomb.tar.zst"
    info = tarfile.TarInfo("release/release-bundle.json")
    _write_custom_zstd_tar(archive, [(info, b"0" * 16_384)])

    with pytest.raises(ValueError, match="exceeds expanded limit"):
        verify_score_release_archive(
            archive_path=archive,
            corpus_manifest_path=corpus_manifest,
        )


def test_verify_rejects_unreferenced_extra_files_and_injected_symlinks(
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
        hardware_validation=make_hardware_validation_binding(),
    )

    def inject_extra_file(argv: list[str], *, cwd: Path | None = None) -> None:
        release_packaging._system_tar(argv, cwd=cwd)
        target_root = Path(argv[-1])
        (target_root / "release" / "unreferenced.txt").write_text(
            "extra", encoding="utf-8"
        )

    with pytest.raises(ValueError, match="unreferenced"):
        verify_score_release_archive(
            archive_path=archive,
            corpus_manifest_path=corpus_manifest,
            tar_runner=inject_extra_file,
        )

    def inject_symlink(argv: list[str], *, cwd: Path | None = None) -> None:
        release_packaging._system_tar(argv, cwd=cwd)
        target_root = Path(argv[-1])
        (target_root / "release" / "escape_link").symlink_to("/etc/hosts")

    with pytest.raises(ValueError, match=r"symlink|non-regular"):
        verify_score_release_archive(
            archive_path=archive,
            corpus_manifest_path=corpus_manifest,
            tar_runner=inject_symlink,
        )


def test_verifier_audits_baseline_slower_than_sol_even_when_candidate_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, workspace = _release_fixture(tmp_path)
    baseline_trace_path = (
        workspace / "baseline" / "traces" / _PROBLEM_PATH / "trace.jsonl"
    )
    baseline_traces = load_jsonl_file(Trace, baseline_trace_path)
    # Set baseline latency equal to SOL bound (1.0 ms -> violates T_b > T_SOL).
    fast_baseline = [
        trace.model_copy(
            update={
                "evaluation": trace.evaluation.model_copy(
                    update={
                        "performance": trace.evaluation.performance.model_copy(
                            update={"latency_ms": 1.0},
                        ),
                    },
                ),
            },
        )
        for trace in baseline_traces
        if trace.evaluation is not None
        and trace.evaluation.performance is not None
    ]
    atomic_write_jsonl_values(baseline_trace_path, fast_baseline)

    candidate_trace_path = (
        workspace / "candidate" / "traces" / _PROBLEM_PATH / "trace.jsonl"
    )
    candidate_traces = load_jsonl_file(Trace, candidate_trace_path)
    failed_candidate = [
        trace.model_copy(
            update={
                "evaluation": Evaluation(
                    status=EvaluationStatus.RUNTIME_ERROR,
                    environment=trace.evaluation.environment.model_copy(
                        update={"clocks_locked": None, "timing_protocol": None},
                    ),
                    timestamp="2026-07-25T00:00:00Z",
                    log="candidate failed",
                ),
            },
        )
        for trace in candidate_traces
        if trace.evaluation is not None
    ]
    atomic_write_jsonl_values(candidate_trace_path, failed_candidate)
    bundle_path = _write_bundle(corpus, workspace)
    _trust_fixture(monkeypatch, corpus)

    with pytest.raises(SOLScoreAuditError, match="strictly slower"):
        verify_and_score_release(bundle_path, corpus_manifest_path=corpus.path)
