from __future__ import annotations

import json
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.core.scoring.release_baseline import (
    ReleaseBaselineBundle,
    ReleaseBaselineVerification,
    ReleaseBaselineVerificationWorkload,
    ReleaseBaselineWorkload,
    ReleaseProvenance,
    sha256_file,
    write_release_baseline_bundle,
    write_release_baseline_verification,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts/internal/release/build_prerelease_artifact_bundle.py"
SPEC = spec_from_file_location("build_prerelease_artifact_bundle", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
build_prerelease_artifact_bundle = module_from_spec(SPEC)
sys.modules[SPEC.name] = build_prerelease_artifact_bundle
SPEC.loader.exec_module(build_prerelease_artifact_bundle)


def _load_manifest(output_dir: Path) -> dict:
    return json.loads(
        (output_dir / "prerelease_artifact_bundle.json").read_text(encoding="utf-8")
    )


def _write_release_evidence(tmp_path: Path) -> tuple[Path, Path]:
    compact_baseline_path = tmp_path / "baseline.json"
    compact_baseline_path.write_text('{"entries": []}\n', encoding="utf-8")
    bundle_path = tmp_path / "release-baseline.json"
    bundle = ReleaseBaselineBundle(
        release="v2.14-test",
        suite_manifest_ref="suite.json",
        suite_manifest_sha256="a" * 64,
        baseline_artifact_ref=str(compact_baseline_path),
        baseline_artifact_sha256=sha256_file(compact_baseline_path),
        provenance=ReleaseProvenance(solution="baseline", solution_sha256="c" * 64),
        workloads=(
            ReleaseBaselineWorkload(
                "gemm",
                "w1",
                "official",
                1.0,
                (),
                trace_ref="trace.jsonl",
                trace_sha256="f" * 64,
                bound_ref="bound.json",
                bound_sha256="d" * 64,
                hardware_model_ref="model.json",
                hardware_model_sha256="e" * 64,
            ),
            ReleaseBaselineWorkload(
                "gemm", "w2", "derived", 2.0, ("model_not_validated",)
            ),
        ),
        latency_tolerance_rel=0.05,
    )
    write_release_baseline_bundle(bundle, bundle_path)
    verification_path = tmp_path / "verification.json"
    write_release_baseline_verification(
        ReleaseBaselineVerification(
            release=bundle.release,
            bundle_ref=str(bundle_path),
            bundle_sha256=sha256_file(bundle_path),
            rerun_trace_ref="rerun.jsonl",
            rerun_trace_sha256="d" * 64,
            workloads=(
                ReleaseBaselineVerificationWorkload(
                    "gemm", "w1", "official", "official", 1.0, 1.0, 0.0, True, ()
                ),
                ReleaseBaselineVerificationWorkload(
                    "gemm",
                    "w2",
                    "derived",
                    "derived",
                    2.0,
                    2.0,
                    0.0,
                    True,
                    ("model_not_validated",),
                ),
            ),
        ),
        verification_path,
    )
    return bundle_path, verification_path


def test_bundle_copies_release_baseline_evidence_and_records_summary(
    tmp_path, monkeypatch
):
    baseline_bundle, verification = _write_release_evidence(tmp_path)

    def fake_run_command(**kwargs):
        validation_dir = Path(
            kwargs["command"][kwargs["command"].index("--output-dir") + 1]
        )
        validation_dir.mkdir(parents=True, exist_ok=True)
        (validation_dir / "release_candidate_validation.json").write_text("{}\n")
        (validation_dir / "release_candidate_validation.md").write_text(
            "# validation\n"
        )
        return build_prerelease_artifact_bundle.CommandTranscript(
            "release_candidate_validation",
            [],
            "passed",
            "diagnostic-only",
            0.0,
            "transcripts/release_candidate_validation.json",
        )

    monkeypatch.setattr(
        build_prerelease_artifact_bundle, "_run_command", fake_run_command
    )
    assert (
        build_prerelease_artifact_bundle.main(
            [
                "--version",
                "v2.14-rc1",
                "--output-dir",
                str(tmp_path / "out"),
                "--release-baseline-bundle",
                str(baseline_bundle),
                "--release-baseline-verification",
                str(verification),
                "--skip-environment-evidence",
            ]
        )
        == 0
    )
    manifest = _load_manifest(tmp_path / "out")
    artifacts = {item["id"]: item for item in manifest["artifacts"]}
    assert artifacts["release_baseline_bundle"]["required"] is True
    assert artifacts["release_baseline_verification"]["required"] is True
    assert artifacts["release_scoring_baseline"]["required"] is True
    assert (
        artifacts["release_scoring_baseline"]["sha256"]
        == json.loads(
            (tmp_path / "out/release_baseline/release_baseline_bundle.json").read_text()
        )["baseline_artifact_sha256"]
    )
    assert manifest["release_baseline_summary"] == {
        "total": 2,
        "official": 1,
        "derived": 1,
        "blocked": 0,
    }


@pytest.mark.parametrize(
    "argument", ["--release-baseline-bundle", "--release-baseline-verification"]
)
def test_bundle_requires_release_evidence_pair(tmp_path, argument):
    with pytest.raises(SystemExit, match="must be supplied together"):
        build_prerelease_artifact_bundle.main(
            ["--version", "v2.14", argument, str(tmp_path / "input.json")]
        )


def test_bundle_rejects_verification_for_different_bundle(tmp_path):
    baseline_bundle, verification = _write_release_evidence(tmp_path)
    payload = json.loads(verification.read_text())
    payload["bundle_sha256"] = "f" * 64
    verification.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="bundle checksum"):
        build_prerelease_artifact_bundle.main(
            [
                "--version",
                "v2.14",
                "--release-baseline-bundle",
                str(baseline_bundle),
                "--release-baseline-verification",
                str(verification),
            ]
        )


def test_cli_package_readiness_preserves_on_disk_release_baseline_links(
    tmp_path, monkeypatch
):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text('[{"definition": "gemm", "workload_uuid": "w1"}]\n')
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "definition": "gemm",
                "workload": {"uuid": "w1"},
                "evaluation": {"status": "PASSED", "performance": {"latency_ms": 1.0}},
            }
        )
        + "\n"
    )
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(
        json.dumps(
            [
                {
                    "definition": "gemm",
                    "workload_uuid": "w1",
                    "bound_ref": "bound.json",
                    "bound_sha256": "a" * 64,
                    "hardware_model_ref": "model.json",
                    "hardware_model_sha256": "b" * 64,
                }
            ]
        )
    )
    baseline_path, bundle_path, verification_path = (
        tmp_path / "scoring.json",
        tmp_path / "bundle.json",
        tmp_path / "verification.json",
    )
    runner = CliRunner()
    build = runner.invoke(
        cli,
        [
            "baseline",
            "release",
            "build",
            "--suite-manifest",
            str(suite_path),
            "--trace",
            str(trace_path),
            "--release",
            "v2.14",
            "--baseline-output",
            str(baseline_path),
            "--bundle-output",
            str(bundle_path),
            "--authority-json",
            str(authority_path),
            "--solution",
            "solution",
            "--solution-sha256",
            "c" * 64,
            "--environment-fingerprint",
            "env",
            "--clock-policy",
            "locked",
            "--timing-policy",
            "median",
            "--compiler-build-id",
            "rocm",
            "--scope",
            "test-authority-slice:gemm:1-workload",
            "--latency-tolerance-rel",
            "0.05",
        ],
    )
    assert build.exit_code == 0, build.output
    rerun_path = tmp_path / "rerun.jsonl"
    rerun_path.write_text(
        json.dumps(
            {
                "definition": "gemm",
                "workload": {"uuid": "w1"},
                "evaluation": {
                    "status": "PASSED",
                    "performance": {"latency_ms": 1.0},
                    "release_baseline": {
                        "bound_sha256": "a" * 64,
                        "hardware_model_sha256": "b" * 64,
                    },
                },
            }
        )
        + "\n"
    )
    verify = runner.invoke(
        cli,
        [
            "baseline",
            "release",
            "verify",
            "--bundle",
            str(bundle_path),
            "--rerun-trace",
            str(rerun_path),
            "--output",
            str(verification_path),
            "--solution-sha256",
            "c" * 64,
            "--environment-fingerprint",
            "env",
            "--clock-policy",
            "locked",
            "--timing-policy",
            "median",
            "--compiler-build-id",
            "rocm",
            "--suite-manifest-sha256",
            sha256_file(suite_path),
        ],
    )
    assert verify.exit_code == 0, verify.output

    def fake_run_command(**kwargs):
        validation_dir = Path(
            kwargs["command"][kwargs["command"].index("--output-dir") + 1]
        )
        validation_dir.mkdir(parents=True, exist_ok=True)
        (validation_dir / "release_candidate_validation.json").write_text("{}\n")
        (validation_dir / "release_candidate_validation.md").write_text(
            "# validation\n"
        )
        return build_prerelease_artifact_bundle.CommandTranscript(
            "release_candidate_validation",
            [],
            "passed",
            "diagnostic-only",
            0.0,
            "transcripts/release_candidate_validation.json",
        )

    monkeypatch.setattr(
        build_prerelease_artifact_bundle, "_run_command", fake_run_command
    )
    package_dir = tmp_path / "package"
    assert (
        build_prerelease_artifact_bundle.main(
            [
                "--version",
                "v2.14-rc1",
                "--output-dir",
                str(package_dir),
                "--release-baseline-bundle",
                str(bundle_path),
                "--release-baseline-verification",
                str(verification_path),
                "--skip-environment-evidence",
            ]
        )
        == 0
    )
    readiness_script = (
        REPO_ROOT / "scripts/internal/release/check_prerelease_readiness.py"
    )
    readiness_spec = spec_from_file_location("e2e_readiness", readiness_script)
    assert readiness_spec is not None and readiness_spec.loader is not None
    readiness = module_from_spec(readiness_spec)
    sys.modules[readiness_spec.name] = readiness
    readiness_spec.loader.exec_module(readiness)
    assert (
        readiness.main(
            [
                "--bundle-dir",
                str(package_dir),
                "--output-dir",
                str(tmp_path / "readiness"),
                "--skip-doc-claim-checks",
            ]
        )
        == 0
    )


def test_bundle_writes_manifest_transcripts_checksums_and_authority_map(
    tmp_path,
    monkeypatch,
):
    def fake_run(command, **kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, stdout="abc123\n", stderr="")
        if command[:2] == ["git", "describe"]:
            return subprocess.CompletedProcess(command, 0, stdout="v1.26\n", stderr="")
        if command[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if any(part.endswith("release_candidate_validation.py") for part in command):
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "release_candidate_validation.json").write_text(
                json.dumps({"overall_status": "passed"}) + "\n",
                encoding="utf-8",
            )
            (output_dir / "release_candidate_validation.md").write_text(
                "# Release Candidate Validation\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                command, 0, stdout="validation ok", stderr=""
            )
        if command == [
            "uv",
            "run",
            "sol-execbench",
            "--format",
            "json",
            "environment",
            "doctor",
        ]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"rocm": {"available": False}}),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(build_prerelease_artifact_bundle.subprocess, "run", fake_run)

    assert (
        build_prerelease_artifact_bundle.main(
            ["--version", "v1.26.0-rc1", "--output-dir", str(tmp_path)]
        )
        == 0
    )

    manifest = _load_manifest(tmp_path)
    assert manifest["schema_version"] == "sol_execbench.prerelease_artifact_bundle.v1"
    assert manifest["bundle_version"] == "v1.26.0-rc1"
    assert manifest["overall_status"] == "passed"
    assert manifest["git"]["commit"] == "abc123"
    assert manifest["git"]["clean_tree"] is True
    assert (tmp_path / "SHA256SUMS").exists()
    assert (tmp_path / "prerelease_artifact_bundle.md").exists()

    classes = {entry["authority_class"] for entry in manifest["authority_map"]}
    assert classes == {
        "canonical",
        "diagnostic-only",
        "provisional",
        "deferred",
        "unavailable",
    }
    artifact_by_id = {artifact["id"]: artifact for artifact in manifest["artifacts"]}
    assert artifact_by_id["release_candidate_validation_json"]["status"] == "present"
    assert artifact_by_id["transcript_release_candidate_validation"]["sha256"]
    assert (
        artifact_by_id["environment_evidence"]["authority_class"] == "diagnostic-only"
    )
    assert (
        "CDNA3/gfx942 validation infrastructure evidence was recorded on MI308X"
        in json.dumps(manifest)
    )
    assert "not full 235-problem paper validation" in (
        tmp_path / "prerelease_artifact_bundle.md"
    ).read_text(encoding="utf-8")


def test_optional_environment_evidence_unavailable_does_not_block_bundle(
    tmp_path,
    monkeypatch,
):
    def fake_run(command, **kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, stdout="abc123\n", stderr="")
        if command[:2] == ["git", "describe"]:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="no tag")
        if command[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=" M docs/CLAIMS.md\n", stderr=""
            )
        if any(part.endswith("release_candidate_validation.py") for part in command):
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "release_candidate_validation.json").write_text(
                json.dumps({"overall_status": "passed"}) + "\n",
                encoding="utf-8",
            )
            (output_dir / "release_candidate_validation.md").write_text(
                "# Release Candidate Validation\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                command, 0, stdout="validation ok", stderr=""
            )
        if command == ["missing-doctor"]:
            raise FileNotFoundError("missing-doctor")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(build_prerelease_artifact_bundle.subprocess, "run", fake_run)

    assert (
        build_prerelease_artifact_bundle.main(
            [
                "--version",
                "v1.26",
                "--output-dir",
                str(tmp_path),
                "--environment-command",
                "missing-doctor",
            ]
        )
        == 0
    )

    manifest = _load_manifest(tmp_path)
    artifact_by_id = {artifact["id"]: artifact for artifact in manifest["artifacts"]}
    assert manifest["overall_status"] == "review_needed"
    assert manifest["git"]["clean_tree"] is False
    assert artifact_by_id["environment_evidence"]["status"] == "unavailable"
    assert manifest["commands"][1]["classification"] == "unavailable"


def test_failed_release_validation_blocks_and_redacts_transcript(
    tmp_path,
    monkeypatch,
):
    def fake_run(command, **kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, stdout="abc123\n", stderr="")
        if command[:2] == ["git", "describe"]:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
        if command[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if any(part.endswith("release_candidate_validation.py") for part in command):
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(
                command,
                2,
                stdout="failed",
                stderr="HF_TOKEN=hf_secret",
            )
        if command == [
            "uv",
            "run",
            "sol-execbench",
            "--format",
            "json",
            "environment",
            "doctor",
        ]:
            return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(build_prerelease_artifact_bundle.subprocess, "run", fake_run)

    assert (
        build_prerelease_artifact_bundle.main(
            ["--version", "v1.26", "--output-dir", str(tmp_path)]
        )
        == 1
    )

    manifest = _load_manifest(tmp_path)
    transcript = json.loads(
        (tmp_path / "transcripts/release_candidate_validation.json").read_text(
            encoding="utf-8"
        )
    )
    release_artifact = {artifact["id"]: artifact for artifact in manifest["artifacts"]}[
        "release_candidate_validation_json"
    ]
    assert manifest["overall_status"] == "blocking"
    assert release_artifact["required"] is True
    assert release_artifact["status"] == "failed"
    assert "hf_secret" not in json.dumps(transcript)
    assert "<redacted>" in transcript["stderr_tail"]


def test_tail_file_redacts_single_line_without_reading_by_line(tmp_path):
    path = tmp_path / "huge.log"
    path.write_text("x" * 20000 + " HF_TOKEN=secret-tail", encoding="utf-8")

    tail = build_prerelease_artifact_bundle._tail_file(path, 200)

    assert "secret-tail" not in tail
    assert "<redacted>" in tail


def test_tail_file_redacts_secret_value_split_across_chunks(tmp_path):
    path = tmp_path / "split.log"
    token = "HF_TOKEN=secret_split_tail"
    path.write_text(
        "x" * (8192 - len("HF_TOKEN=secret")) + token,
        encoding="utf-8",
    )

    tail = build_prerelease_artifact_bundle._tail_file(path, 200)

    assert "secret_split_tail" not in tail
    assert "split_tail" not in tail
    assert "<redacted>" in tail
