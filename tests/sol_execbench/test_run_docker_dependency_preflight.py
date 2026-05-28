from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DOCKER_SCRIPT = REPO_ROOT / "scripts" / "run_docker.sh"


def _matching_default_dependency_env() -> dict[str, str]:
    return {
        "SOL_EXECBENCH_DEPENDENCY_TORCH_DISTRIBUTION_VERSION": "2.10.0+rocm7.1",
        "SOL_EXECBENCH_DEPENDENCY_TORCH_VERSION": "2.10.0+rocm7.1",
        "SOL_EXECBENCH_DEPENDENCY_TORCH_LOCAL_VERSION": "rocm7.1",
        "SOL_EXECBENCH_DEPENDENCY_TORCH_ROCM_TARGET": "rocm7.1",
        "SOL_EXECBENCH_DEPENDENCY_TORCH_HIP_VERSION": "7.1.0",
        "SOL_EXECBENCH_DEPENDENCY_TORCH_CUDA_VERSION": "none",
        "SOL_EXECBENCH_DEPENDENCY_TORCH_DEVICE_AVAILABLE": "true",
        "SOL_EXECBENCH_DEPENDENCY_TORCHVISION_DISTRIBUTION_VERSION": (
            "0.25.0+rocm7.1"
        ),
        "SOL_EXECBENCH_DEPENDENCY_TRITON_ROCM_DISTRIBUTION_VERSION": "3.6.0",
        "SOL_EXECBENCH_DEPENDENCY_TRITON_ROCM_STATUS": "installed",
        "SOL_EXECBENCH_DEPENDENCY_CONTAINER_ROCM_USER_SPACE_VERSION": "7.1.1",
        "SOL_EXECBENCH_DEPENDENCY_TOOLCHAIN_ROCM_VERSION": "7.1.1",
    }


def _run_docker_dependency_preflight(
    *args: str,
    **env_overrides: str,
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "SOL_EXECBENCH_RUN_DOCKER_DRY_RUN": "1",
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
        "rocm-7.2.0-ubuntu-24.04-container",
        **{
            **_matching_default_dependency_env(),
            "SOL_EXECBENCH_DEPENDENCY_CONTAINER_ROCM_USER_SPACE_VERSION": "7.2.0",
            "SOL_EXECBENCH_DEPENDENCY_TOOLCHAIN_ROCM_VERSION": "7.2.0",
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
            "SOL_EXECBENCH_DEPENDENCY_TORCH_IMPORT_ERROR": (
                "No module named 'torch'"
            ),
        },
    )

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "pytorch_wheel_unavailable"
    assert payload["reason_code"] == "pytorch_rocm_wheel_unavailable"
    assert payload["benchmark_allowed"] is False
    assert "docker build" not in completed.stdout
    assert "docker run" not in completed.stdout


def test_matching_default_dependency_stack_stops_only_on_clean_validation_policy() -> None:
    completed = _run_docker_dependency_preflight(
        "--preflight-only",
        **_matching_default_dependency_env(),
    )

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["target_id"] == "rocm-7.1.1-ubuntu-24.04-container"
    assert payload["status"] == "not_tested"
    assert payload["reason_code"] == "target_not_tested"
    assert payload["expected_local_version"] == "rocm7.1"
    assert payload["uv_index_name"] == "pytorch-rocm71"
    assert payload["benchmark_allowed"] is False
    assert "docker build" not in completed.stdout
    assert "docker run" not in completed.stdout
