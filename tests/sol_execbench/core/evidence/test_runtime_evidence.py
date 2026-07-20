from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from sol_execbench.core.evidence import runtime_evidence
from sol_execbench.core.evidence.runtime_evidence import cli
from sol_execbench.core.evidence.runtime_evidence.models import RuntimeFailureEvidence
from sol_execbench.core.platform.dependency_matrix import PytorchDependencyObservation
from sol_execbench.core.platform.docker_matrix import load_docker_target_manifest

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = REPO_ROOT / "docker" / "rocm-targets.json"
TARGET_ID = "rocm-7.2.0-ubuntu-24.04-container"


def _target():
    return load_docker_target_manifest(MANIFEST_PATH).targets_by_id[TARGET_ID]


def _matching_observation() -> PytorchDependencyObservation:
    return PytorchDependencyObservation(
        torch_distribution_version="2.11.0+rocm7.2",
        torch_version="2.11.0+rocm7.2",
        torch_local_version="rocm7.2",
        torch_rocm_target="rocm7.2",
        torch_hip_version="7.2.0",
        torch_device_available=True,
        torchvision_distribution_version="0.26.0+rocm7.2",
        triton_rocm_distribution_version="3.6.0",
        triton_rocm_status="installed",
        container_rocm_user_space_version="7.2.0",
        hipcc_version="HIP version: 7.2.0",
        toolchain_rocm_version="7.2.0",
    )


def test_runtime_entry_keeps_scopes_and_failure_artifacts_separate() -> None:
    entry = runtime_evidence.build_runtime_matrix_entry(
        target=_target(),
        dependency_observation=_matching_observation(),
        host=runtime_evidence.build_host_evidence(
            rocm_version="7.2.0",
            driver_version="6.14.0",
            dev_kfd_present=True,
            dev_kfd_accessible=True,
            dev_dri_present=True,
            dev_dri_accessible=True,
        ),
        gpu=runtime_evidence.collect_gpu_evidence(
            device_count=1,
            device_name="AMD Radeon",
            gfx_architecture="gfx1200",
            visible_device_environment={"HIP_VISIBLE_DEVICES": "0"},
        ),
        failure_evidence=[
            RuntimeFailureEvidence(
                category="dependency",
                status="recorded",
                message="diagnostic only",
            )
        ],
        container_validated=True,
    )
    payload = entry.model_dump(mode="json")

    assert payload["status"] == "container_validated"
    assert payload["claim_boundary"]["container_user_space_validated"] is True
    assert payload["claim_boundary"]["native_host_validated"] is False
    assert payload["observed"]["host"]["device_nodes"] == ["/dev/kfd", "/dev/dri"]
    assert payload["observed"]["gpu"]["gfx_architecture"] == "gfx1200"
    assert payload["artifacts"][0]["kind"] == "runtime_evidence_dependency"


def test_runtime_unavailable_overrides_dependency_classification() -> None:
    entry = runtime_evidence.build_runtime_matrix_entry(
        target=_target(),
        dependency_observation=_matching_observation(),
        runtime_unavailable_reason="/dev/kfd missing",
        container_validated=True,
    )

    assert entry.status.value == "runtime_unavailable"
    assert entry.reason_code.value == "rocm_runtime_unavailable"
    assert entry.claim_boundary.container_user_space_validated is False


def test_runtime_evidence_io_round_trips_entry_and_aggregate(tmp_path) -> None:
    entry = runtime_evidence.build_runtime_matrix_entry(
        target=_target(), dependency_observation=_matching_observation()
    )
    entry_path = tmp_path / "entry.json"
    report_path = tmp_path / "report.json"

    runtime_evidence.write_matrix_entry(entry_path, entry)
    loaded = runtime_evidence.load_matrix_entry(entry_path)
    runtime_evidence.write_aggregate_report(report_path, [loaded])

    assert loaded == entry
    assert json.loads(entry_path.read_text())["schema_version"] == (
        "sol_execbench.rocm_compatibility_matrix.v1"
    )
    assert json.loads(report_path.read_text())["status_counts"] == {"not_tested": 1}


def test_collectors_use_injected_values_without_hardware_probe() -> None:
    gpu = runtime_evidence.collect_gpu_evidence(
        device_count=2,
        device_name="Injected GPU",
        gfx_architecture="gfx1200",
        visible_device_environment={"ROCR_VISIBLE_DEVICES": "1"},
    )
    host = runtime_evidence.build_host_evidence(
        dev_kfd_present=True,
        dev_kfd_accessible=False,
        dev_dri_present=True,
        dev_dri_accessible=True,
    )

    assert gpu.device_count == 2
    assert gpu.visible_device_environment == {"ROCR_VISIBLE_DEVICES": "1"}
    assert host.device_nodes == ["/dev/dri"]


def test_dependency_collector_preserves_explicit_toolchain_overrides(
    monkeypatch,
) -> None:
    observed = PytorchDependencyObservation(torch_version="2.11.0+rocm7.2")
    collector_calls = 0

    def collect() -> PytorchDependencyObservation:
        nonlocal collector_calls
        collector_calls += 1
        return observed

    monkeypatch.setattr(
        runtime_evidence, "collect_pytorch_dependency_observation", collect
    )
    result = runtime_evidence.build_dependency_observation(
        PytorchDependencyObservation(
            container_rocm_user_space_version="7.2.0",
            hipcc_version="HIP 7.2.0",
        )
    )

    assert collector_calls == 1
    assert result.torch_version == observed.torch_version
    assert result.container_rocm_user_space_version == "7.2.0"
    assert result.hipcc_version == "HIP 7.2.0"


def test_visible_environment_arguments_merge_and_validate(monkeypatch) -> None:
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")

    assert cli.visible_env_from_args(["ROCR_VISIBLE_DEVICES=1"]) == {
        "HIP_VISIBLE_DEVICES": "0",
        "ROCR_VISIBLE_DEVICES": "1",
    }
    with pytest.raises(argparse.ArgumentTypeError, match="NAME=VALUE"):
        cli.visible_env_from_args(["BROKEN"])


def test_runtime_evidence_cli_collects_and_aggregates(tmp_path, capsys) -> None:
    entry_path = tmp_path / "entry.json"
    report_path = tmp_path / "report.json"
    collect_args = [
        "collect-target",
        "--manifest",
        str(MANIFEST_PATH),
        "--target",
        TARGET_ID,
        "--output",
        str(entry_path),
        "--torch-distribution-version",
        "2.11.0+rocm7.2",
        "--torch-version",
        "2.11.0+rocm7.2",
        "--torch-local-version",
        "rocm7.2",
        "--torch-rocm-target",
        "rocm7.2",
        "--torch-hip-version",
        "7.2.0",
        "--torch-device-available",
        "true",
        "--torchvision-distribution-version",
        "0.26.0+rocm7.2",
        "--triton-rocm-distribution-version",
        "3.6.0",
        "--triton-rocm-status",
        "installed",
        "--container-rocm-user-space-version",
        "7.2.0",
        "--hipcc-version",
        "HIP version: 7.2.0",
        "--toolchain-rocm-version",
        "7.2.0",
        "--container-validated",
        "--device-count",
        "1",
        "--device-name",
        "AMD Radeon",
        "--gfx-architecture",
        "gfx1200",
        "--visible-device-env",
        "HIP_VISIBLE_DEVICES=0",
        "--failure-category",
        "dependency",
    ]

    assert runtime_evidence.main(collect_args) == 0
    collected = json.loads(capsys.readouterr().out)
    assert collected["status"] == "container_validated"
    assert collected["artifacts"][0]["kind"] == "runtime_evidence_dependency"

    assert (
        runtime_evidence.main(
            ["aggregate", "--output", str(report_path), str(entry_path)]
        )
        == 0
    )
    aggregate = json.loads(capsys.readouterr().out)
    assert aggregate["status_counts"] == {"container_validated": 1}


def test_runtime_evidence_cli_rejects_invalid_boolean(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit):
        runtime_evidence.main(
            [
                "collect-target",
                "--manifest",
                str(MANIFEST_PATH),
                "--output",
                str(tmp_path / "entry.json"),
                "--dev-kfd-present",
                "maybe",
            ]
        )

    assert "expected boolean value" in capsys.readouterr().err
