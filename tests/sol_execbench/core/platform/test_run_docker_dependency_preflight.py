from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_DOCKER_SCRIPT = REPO_ROOT / "scripts" / "run_docker.sh"

pytestmark = [pytest.mark.requires_linux, pytest.mark.subprocess_uv]


def _matching_default_dependency_env() -> dict[str, str]:
    return {
        "SOL_EXECBENCH_DEPENDENCY_TORCH_DISTRIBUTION_VERSION": "2.11.0+rocm7.2",
        "SOL_EXECBENCH_DEPENDENCY_TORCH_VERSION": "2.11.0+rocm7.2",
        "SOL_EXECBENCH_DEPENDENCY_TORCH_LOCAL_VERSION": "rocm7.2",
        "SOL_EXECBENCH_DEPENDENCY_TORCH_ROCM_TARGET": "rocm7.2",
        "SOL_EXECBENCH_DEPENDENCY_TORCH_HIP_VERSION": "7.2.0",
        "SOL_EXECBENCH_DEPENDENCY_TORCH_CUDA_VERSION": "none",
        "SOL_EXECBENCH_DEPENDENCY_TORCH_DEVICE_AVAILABLE": "true",
        "SOL_EXECBENCH_DEPENDENCY_TORCHVISION_DISTRIBUTION_VERSION": ("0.26.0+rocm7.2"),
        "SOL_EXECBENCH_DEPENDENCY_TRITON_ROCM_DISTRIBUTION_VERSION": "3.6.0",
        "SOL_EXECBENCH_DEPENDENCY_TRITON_ROCM_STATUS": "installed",
        "SOL_EXECBENCH_DEPENDENCY_CONTAINER_ROCM_USER_SPACE_VERSION": "7.2.0",
        "SOL_EXECBENCH_DEPENDENCY_TOOLCHAIN_ROCM_VERSION": "7.2.0",
    }


def _run_docker_dependency_preflight(
    *args: str,
    **env_overrides: str,
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "SOL_EXECBENCH_RUN_DOCKER_DRY_RUN": "1",
        "SOL_EXECBENCH_HOST_PYTHON": sys.executable,
        **env_overrides,
    }
    return subprocess.run(
        [str(RUN_DOCKER_SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_mixed_dependency_state_blocks_before_docker_command_text() -> None:
    completed = _run_docker_dependency_preflight(
        "--preflight-only",
        "--target",
        "rocm-7.1.1-ubuntu-24.04-container",
        **{
            **_matching_default_dependency_env(),
            "SOL_EXECBENCH_DEPENDENCY_CONTAINER_ROCM_USER_SPACE_VERSION": "7.1.1",
            "SOL_EXECBENCH_DEPENDENCY_TOOLCHAIN_ROCM_VERSION": "7.1.1",
        },
    )

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "mixed_version"
    assert payload["reason_code"] == "target_observed_mismatch"
    assert payload["benchmark_allowed"] is False
    assert "docker build" not in completed.stdout
    assert "docker run" not in completed.stdout


def test_missing_required_dependency_blocks_before_docker_command_text() -> None:
    completed = _run_docker_dependency_preflight(
        "--preflight-only",
        **{
            **_matching_default_dependency_env(),
            "SOL_EXECBENCH_DEPENDENCY_TORCH_DISTRIBUTION_VERSION": "none",
            "SOL_EXECBENCH_DEPENDENCY_TORCH_IMPORT_ERROR": ("No module named 'torch'"),
        },
    )

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "pytorch_wheel_unavailable"
    assert payload["reason_code"] == "pytorch_rocm_wheel_unavailable"
    assert payload["benchmark_allowed"] is False
    assert "docker build" not in completed.stdout
    assert "docker run" not in completed.stdout


def test_matching_default_dependency_stack_stops_only_on_clean_validation_policy() -> (
    None
):
    completed = _run_docker_dependency_preflight(
        "--preflight-only",
        **_matching_default_dependency_env(),
    )

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["target_id"] == "rocm-7.2.0-ubuntu-24.04-container"
    assert payload["status"] == "not_tested"
    assert payload["reason_code"] == "target_not_tested"
    assert payload["expected_local_version"] == "rocm7.2"
    assert payload["uv_index_name"] == "pytorch-rocm72"
    assert payload["benchmark_allowed"] is False
    assert "docker build" not in completed.stdout
    assert "docker run" not in completed.stdout


def test_help_documents_dependency_override_separately_from_unknown_target() -> None:
    script = RUN_DOCKER_SCRIPT.read_text()

    assert "--allow-mixed-version-dependencies" in script
    assert "--record-container-validation" in script
    assert "SOL_EXECBENCH_ALLOW_MIXED_VERSION_DEPENDENCIES" in script
    assert "SOL_EXECBENCH_RECORD_CONTAINER_VALIDATION" in script
    assert "allow-unknown-target" in script
    assert (
        "dependency mismatch override"
        not in script.partition("--allow-unknown-target")[2].splitlines()[0]
    )


def test_unknown_target_override_does_not_allow_mixed_dependencies() -> None:
    completed = _run_docker_dependency_preflight(
        "--preflight-only",
        "--target",
        "rocm-7.1.1-ubuntu-24.04-container",
        "--allow-unknown-target",
        **{
            **_matching_default_dependency_env(),
            "SOL_EXECBENCH_DEPENDENCY_CONTAINER_ROCM_USER_SPACE_VERSION": "7.1.1",
            "SOL_EXECBENCH_DEPENDENCY_TOOLCHAIN_ROCM_VERSION": "7.1.1",
        },
    )

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "mixed_version"
    assert payload["reason_code"] == "target_observed_mismatch"
    assert payload["probes_allowed"] is False
    assert payload["smoke_allowed"] is False
    assert payload["benchmark_allowed"] is False
    assert "docker build" not in completed.stdout
    assert "docker run" not in completed.stdout


def test_dependency_override_reports_probe_smoke_only_without_authority() -> None:
    completed = _run_docker_dependency_preflight(
        "--preflight-only",
        "--target",
        "rocm-7.1.1-ubuntu-24.04-container",
        "--allow-mixed-version-dependencies",
        **{
            **_matching_default_dependency_env(),
            "SOL_EXECBENCH_DEPENDENCY_CONTAINER_ROCM_USER_SPACE_VERSION": "7.1.1",
            "SOL_EXECBENCH_DEPENDENCY_TOOLCHAIN_ROCM_VERSION": "7.1.1",
        },
    )

    assert completed.returncode != 0
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
    assert "docker build" not in completed.stdout
    assert "docker run" not in completed.stdout


def test_dependency_env_override_reports_probe_smoke_only_without_authority() -> None:
    completed = _run_docker_dependency_preflight(
        "--preflight-only",
        "--target",
        "rocm-7.1.1-ubuntu-24.04-container",
        **{
            **_matching_default_dependency_env(),
            "SOL_EXECBENCH_DEPENDENCY_CONTAINER_ROCM_USER_SPACE_VERSION": "7.1.1",
            "SOL_EXECBENCH_DEPENDENCY_TOOLCHAIN_ROCM_VERSION": "7.1.1",
            "SOL_EXECBENCH_ALLOW_MIXED_VERSION_DEPENDENCIES": "1",
        },
    )

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "mixed_version"
    assert payload["probes_allowed"] is True
    assert payload["smoke_allowed"] is True
    assert payload["benchmark_allowed"] is False
    assert payload["score_authority"] is False
    assert payload["paper_parity_authority"] is False
    assert payload["leaderboard_authority"] is False
    assert "docker run" not in completed.stdout


def test_dependency_override_allows_normal_dry_run_smoke_without_authority() -> None:
    completed = _run_docker_dependency_preflight(
        "--target",
        "rocm-7.1.1-ubuntu-24.04-container",
        "--allow-mixed-version-dependencies",
        "--",
        "sol-execbench",
        "problems/local/RX_9060_XT",
        **{
            **_matching_default_dependency_env(),
            "SOL_EXECBENCH_DEPENDENCY_CONTAINER_ROCM_USER_SPACE_VERSION": "7.1.1",
            "SOL_EXECBENCH_DEPENDENCY_TOOLCHAIN_ROCM_VERSION": "7.1.1",
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert "+ docker run" in completed.stdout
    assert "sol-execbench:rocm-7.1.1-complete" in completed.stdout
    assert "sol-execbench problems/local/RX_9060_XT" in completed.stdout


def test_missing_required_dependency_still_blocks_normal_dry_run_smoke() -> None:
    completed = _run_docker_dependency_preflight(
        "--allow-mixed-version-dependencies",
        "--",
        "sol-execbench",
        "problems/local/RX_9060_XT",
        **{
            **_matching_default_dependency_env(),
            "SOL_EXECBENCH_DEPENDENCY_TORCH_DISTRIBUTION_VERSION": "none",
            "SOL_EXECBENCH_DEPENDENCY_TORCH_IMPORT_ERROR": ("No module named 'torch'"),
        },
    )

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "pytorch_wheel_unavailable"
    assert "docker run" not in completed.stdout


def test_record_container_validation_allows_matching_not_tested_dry_run() -> None:
    completed = _run_docker_dependency_preflight(
        "--record-container-validation",
        "--",
        "sol-execbench",
        "problems/local/RX_9060_XT",
        **_matching_default_dependency_env(),
    )

    assert completed.returncode == 0, completed.stderr
    assert "+ docker run" in completed.stdout
    assert "sol-execbench:rocm-7.2-complete" in completed.stdout
    assert "sol-execbench problems/local/RX_9060_XT" in completed.stdout


def test_record_container_validation_still_blocks_mixed_dry_run() -> None:
    completed = _run_docker_dependency_preflight(
        "--record-container-validation",
        "--target",
        "rocm-7.1.1-ubuntu-24.04-container",
        "--",
        "sol-execbench",
        "problems/local/RX_9060_XT",
        **{
            **_matching_default_dependency_env(),
            "SOL_EXECBENCH_DEPENDENCY_CONTAINER_ROCM_USER_SPACE_VERSION": "7.1.1",
            "SOL_EXECBENCH_DEPENDENCY_TOOLCHAIN_ROCM_VERSION": "7.1.1",
        },
    )

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "mixed_version"
    assert "docker run" not in completed.stdout
