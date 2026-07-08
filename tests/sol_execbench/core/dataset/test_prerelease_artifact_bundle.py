from __future__ import annotations

import json
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

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
        if command == ["uv", "run", "sol-execbench", "doctor", "--json"]:
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
        if command == ["uv", "run", "sol-execbench", "doctor", "--json"]:
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
