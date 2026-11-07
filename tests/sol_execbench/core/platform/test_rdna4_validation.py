from __future__ import annotations

from pathlib import Path

import pytest

from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.platform.rdna4_validation import (
    build_validation_manifest,
    build_validation_receipt,
    validate_environment_payload,
    verify_validation_directory,
    verify_validation_receipt,
)


def _environment() -> dict:
    return {
        "status": "available",
        "snapshot": {
            "tools": {
                "amd-smi": {
                    "parsed": {
                        "pci_vendor_ids": ["0x1002"],
                        "pci_device_ids": ["0x7590"],
                    },
                },
            },
            "gpus": [
                {
                    "index": 0,
                    "name": "AMD Radeon RX 9060 XT",
                    "gfx_target": "gfx1200",
                },
            ],
            "rocm": {"version": "7.2.0"},
            "pytorch": {
                "available": True,
                "device_count": 1,
                "device_name": "AMD Radeon RX 9060 XT",
                "gfx_target": "gfx1200",
                "torch_version": "2.11.0+rocm7.2",
                "hip_version": "7.2.26015",
            },
        },
    }


def _write_bundle(
    path: Path,
    *,
    skipped: int = 0,
    github_actions: bool = False,
) -> dict:
    path.mkdir()
    environment_path = path / "environment-doctor.json"
    junit_path = path / "pytest-rdna4.xml"
    stdout_path = path / "pytest.stdout.txt"
    stderr_path = path / "pytest.stderr.txt"
    atomic_write_json_value(environment_path, _environment())
    junit_path.write_text(
        '<testsuites><testsuite tests="2" failures="0" errors="0" '
        f'skipped="{skipped}"/></testsuites>',
        encoding="utf-8",
    )
    stdout_path.write_text("2 passed\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    manifest = build_validation_manifest(
        directory=path,
        source_revision="a" * 40,
        source_dirty=False,
        generated_at="2026-07-25T00:00:00Z",
        environment=validate_environment_payload(_environment()),
        pytest_returncode=0,
        artifact_paths=(environment_path, junit_path, stdout_path, stderr_path),
        attestation=(
            {
                "kind": "github_actions_self_hosted",
                "trusted_execution": False,
                "repository": "owner/repository",
                "workflow_ref": "owner/repository/.github/workflows/rdna4-hardware.yml@refs/heads/main",
                "workflow_name": "RDNA4 Hardware",
                "run_id": 123,
                "run_attempt": 2,
            }
            if github_actions
            else None
        ),
    )
    atomic_write_json_value(path / "manifest.json", manifest)
    return manifest


def test_verifies_content_addressed_local_rdna4_bundle(tmp_path: Path):
    directory = tmp_path / "bundle"
    manifest = _write_bundle(directory)

    verified = verify_validation_directory(
        directory,
        expected_source_revision="a" * 40,
    )

    assert verified == manifest
    assert verified["status"] == "passed"
    assert verified["release_eligible"] is False
    assert verified["target"]["rocm_version"] == "7.2.0"
    assert verified["target"]["torch_version"] == "2.11.0+rocm7.2"


def test_local_validation_bundle_cannot_become_publisher_release(
    tmp_path: Path,
):
    directory = tmp_path / "bundle"
    _write_bundle(directory)

    with pytest.raises(ValueError, match="not release eligible"):
        verify_validation_directory(directory, require_release_eligible=True)


def test_workflow_receipt_binds_exact_sha_and_evidence(tmp_path: Path) -> None:
    directory = tmp_path / "bundle"
    _write_bundle(directory, github_actions=True)
    receipt_path = directory / "receipt.json"

    receipt = build_validation_receipt(
        directory,
        receipt_path,
        source_revision="a" * 40,
        workflow_run_id=123,
        workflow_run_attempt=2,
        created_at="2026-08-15T00:00:00Z",
    )
    binding = verify_validation_receipt(
        receipt_path,
        directory,
        expected_source_revision="a" * 40,
    )

    assert receipt.evidence_sha256 == sha256_file(directory / "manifest.json")
    assert binding.receipt_sha256 == sha256_file(receipt_path)
    assert binding.workflow_run_id == 123
    assert binding.source_revision == "a" * 40


def test_workflow_receipt_rejects_identity_and_evidence_tampering(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "bundle"
    _write_bundle(directory, github_actions=True)
    receipt_path = directory / "receipt.json"
    with pytest.raises(ValueError, match="workflow identity"):
        build_validation_receipt(
            directory,
            receipt_path,
            source_revision="a" * 40,
            workflow_run_id=999,
            workflow_run_attempt=2,
            created_at="2026-08-15T00:00:00Z",
        )
    build_validation_receipt(
        directory,
        receipt_path,
        source_revision="a" * 40,
        workflow_run_id=123,
        workflow_run_attempt=2,
        created_at="2026-08-15T00:00:00Z",
    )
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="evidence checksum"):
        verify_validation_receipt(
            receipt_path,
            directory,
            expected_source_revision="a" * 40,
        )


def test_self_declared_release_fields_are_rejected(tmp_path: Path):
    directory = tmp_path / "bundle"
    manifest = _write_bundle(directory)
    manifest.pop("payload_sha256")
    manifest["release_eligible"] = True
    manifest["attestation"] = {
        "kind": "local_unsigned",
        "trusted_execution": True,
    }
    manifest["payload_sha256"] = stable_json_checksum(manifest)
    atomic_write_json_value(directory / "manifest.json", manifest)

    with pytest.raises(ValueError, match="authority contract is invalid"):
        verify_validation_directory(directory)


def test_rejects_tampered_artifact_and_source_revision(tmp_path: Path):
    directory = tmp_path / "bundle"
    _write_bundle(directory)

    (directory / "pytest.stdout.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact identity mismatch"):
        verify_validation_directory(directory)

    directory = tmp_path / "other-bundle"
    _write_bundle(directory)
    with pytest.raises(ValueError, match="source revision mismatch"):
        verify_validation_directory(
            directory,
            expected_source_revision="b" * 40,
        )


def test_rejects_skipped_hardware_tests(tmp_path: Path):
    directory = tmp_path / "bundle"
    manifest = _write_bundle(directory, skipped=1)

    assert manifest["status"] == "failed"
    with pytest.raises(ValueError, match="did not pass"):
        verify_validation_directory(directory)


def test_rejects_non_gfx1200_environment():
    payload = _environment()
    payload["snapshot"]["gpus"][0]["gfx_target"] = "gfx1150"

    with pytest.raises(ValueError, match="exactly one RX 9060 XT gfx1200"):
        validate_environment_payload(payload)


def test_rejects_other_gfx1200_pci_device():
    payload = _environment()
    payload["snapshot"]["tools"]["amd-smi"]["parsed"]["pci_device_ids"] = [
        "0x7550",
    ]

    with pytest.raises(ValueError, match="PCI identity"):
        validate_environment_payload(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("torch_version", "2.12.0+rocm7.3", "PyTorch version"),
        ("hip_version", "7.3.0", "HIP version"),
    ],
)
def test_rejects_user_space_outside_locked_scope(field, value, message):
    payload = _environment()
    payload["snapshot"]["pytorch"][field] = value

    with pytest.raises(ValueError, match=message):
        validate_environment_payload(payload)


def test_rejects_rocm_outside_locked_scope():
    payload = _environment()
    payload["snapshot"]["rocm"]["version"] = "7.3.0"

    with pytest.raises(ValueError, match="ROCm version"):
        validate_environment_payload(payload)


def test_rejects_manifest_target_that_disagrees_with_environment(
    tmp_path: Path,
):
    directory = tmp_path / "bundle"
    manifest = _write_bundle(directory)
    manifest.pop("payload_sha256")
    manifest["target"]["device_name"] = "Other RDNA 4 GPU"
    manifest["payload_sha256"] = stable_json_checksum(manifest)
    atomic_write_json_value(directory / "manifest.json", manifest)

    with pytest.raises(ValueError, match="does not match environment evidence"):
        verify_validation_directory(directory)
