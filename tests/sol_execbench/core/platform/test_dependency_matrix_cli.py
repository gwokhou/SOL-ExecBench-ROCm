from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sol_execbench.core.dependency_matrix import (
    PytorchDependencyObservation,
    classify_dependency_preflight,
    load_docker_target_dependency_policy,
)
from sol_execbench.core.docker_matrix import load_docker_target_manifest


REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = REPO_ROOT / "docker" / "rocm-targets.json"


def _base_command(*extra: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sol_execbench.core.dependency_matrix",
        "preflight",
        "--manifest",
        str(MANIFEST_PATH),
        "--torch-distribution-version",
        "2.10.0+rocm7.1",
        "--torch-version",
        "2.10.0+rocm7.1",
        "--torch-local-version",
        "rocm7.1",
        "--torch-rocm-target",
        "rocm7.1",
        "--torch-hip-version",
        "7.1.0",
        "--torch-cuda-version",
        "none",
        "--torch-device-available",
        "true",
        "--torchvision-distribution-version",
        "0.25.0+rocm7.1",
        "--triton-rocm-distribution-version",
        "3.6.0",
        "--triton-rocm-status",
        "installed",
        "--container-rocm-user-space-version",
        "7.1.1",
        "--toolchain-rocm-version",
        "7.1.1",
        *extra,
    ]


def test_module_main_emits_default_dependency_preflight_json() -> None:
    completed = subprocess.run(
        _base_command(),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["target_id"] == "rocm-7.1.1-ubuntu-24.04-container"
    assert payload["pytorch_rocm_target"] == "rocm7.1"
    assert payload["policy_id"] == "pytorch-2.10.0-rocm7.1-project-default"
    assert payload["wheel_availability"] == "available"
    assert payload["expected_local_version"] == "rocm7.1"
    assert payload["uv_index_name"] == "pytorch-rocm71"
    assert payload["uv_index_url"] == "https://download.pytorch.org/whl/rocm7.1"
    assert payload["lock_strategy"] == "project_default"
    assert payload["torch_version"] == "2.10.0+rocm7.1"
    assert payload["torchvision_version"] == "0.25.0+rocm7.1"
    assert payload["triton_rocm_version"] == "3.6.0"
    assert payload["triton_rocm_index_name"] == "pytorch-rocm-root"
    assert payload["triton_rocm_index_url"] == "https://download.pytorch.org/whl/"
    assert payload["status"] == "not_tested"
    assert payload["reason_code"] == "target_not_tested"
    assert payload["benchmark_allowed"] is False


def test_cli_json_matches_underlying_matrix_entry_policy_payload() -> None:
    manifest = load_docker_target_manifest(MANIFEST_PATH)
    target = manifest.targets_by_id["rocm-7.1.1-ubuntu-24.04-container"]
    policy = load_docker_target_dependency_policy(target)
    observation = PytorchDependencyObservation(
        torch_distribution_version="2.10.0+rocm7.1",
        torch_version="2.10.0+rocm7.1",
        torch_local_version="rocm7.1",
        torch_rocm_target="rocm7.1",
        torch_hip_version="7.1.0",
        torch_cuda_version=None,
        torch_device_available=True,
        torchvision_distribution_version="0.25.0+rocm7.1",
        triton_rocm_distribution_version="3.6.0",
        triton_rocm_status="installed",
        container_rocm_user_space_version="7.1.1",
        toolchain_rocm_version="7.1.1",
    )

    result = classify_dependency_preflight(
        target=target,
        policy=policy,
        observation=observation,
    )
    completed = subprocess.run(
        _base_command(),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    matrix_policy = result.entry.model_dump(mode="json")["observed"][
        "dependency_policy"
    ]

    for key in matrix_policy:
        assert payload[key] == matrix_policy[key]


def test_rocm_7_2_target_with_default_rocm_7_1_stack_reports_mixed_version() -> None:
    completed = subprocess.run(
        _base_command(
            "--target",
            "rocm-7.2.0-ubuntu-24.04-container",
            "--container-rocm-user-space-version",
            "7.2.0",
            "--toolchain-rocm-version",
            "7.2.0",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["target_id"] == "rocm-7.2.0-ubuntu-24.04-container"
    assert payload["expected_local_version"] == "rocm7.2"
    assert payload["status"] == "mixed_version"
    assert payload["reason_code"] == "target_observed_mismatch"
    assert payload["benchmark_allowed"] is False
    assert payload["score_authority"] is False
    assert payload["paper_parity_authority"] is False
    assert payload["leaderboard_authority"] is False


def test_mixed_version_debug_override_json_allows_probe_or_smoke_only() -> None:
    completed = subprocess.run(
        _base_command(
            "--torch-local-version",
            "rocm7.0",
            "--allow-mixed-version-debug",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["status"] == "mixed_version"
    assert payload["probes_allowed"] is True
    assert payload["smoke_allowed"] is True
    assert payload["benchmark_allowed"] is False
    assert payload["container_user_space_validated"] is False
    assert payload["native_host_validated"] is False
    assert payload["hardware_validated"] is False
    assert payload["score_authority"] is False
    assert payload["paper_parity_authority"] is False
    assert payload["leaderboard_authority"] is False


def test_module_main_rejects_invalid_boolean_without_traceback() -> None:
    completed = subprocess.run(
        _base_command("--torch-device-available", "maybe"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "expected boolean value" in completed.stderr
    assert "Traceback" not in completed.stderr
