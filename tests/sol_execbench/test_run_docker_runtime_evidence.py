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
        "SOL_EXECBENCH_DEPENDENCY_HIPCC_VERSION": "HIP version: 7.1.1",
        "SOL_EXECBENCH_DEPENDENCY_TOOLCHAIN_ROCM_VERSION": "7.1.1",
    }


def _host_preflight_env() -> dict[str, str]:
    return {
        "SOL_EXECBENCH_DOCKER_CONTEXT": "default",
        "SOL_EXECBENCH_DOCKER_HOST": "unix:///var/run/docker.sock",
        "SOL_EXECBENCH_DEV_KFD_PRESENT": "true",
        "SOL_EXECBENCH_DEV_KFD_ACCESSIBLE": "true",
        "SOL_EXECBENCH_DEV_DRI_PRESENT": "true",
        "SOL_EXECBENCH_DEV_DRI_ACCESSIBLE": "true",
        "SOL_EXECBENCH_GPU_ACCESSIBLE": "true",
        "SOL_EXECBENCH_HOST_ROCM_VERSION": "7.1.1",
        "SOL_EXECBENCH_HOST_DRIVER_VERSION": "6.14.0",
        "SOL_EXECBENCH_RUNTIME_DEVICE_COUNT": "1",
        "SOL_EXECBENCH_RUNTIME_DEVICE_NAME": "AMD Radeon",
        "SOL_EXECBENCH_RUNTIME_GFX_ARCHITECTURE": "gfx1200",
        "HIP_VISIBLE_DEVICES": "0",
    }


def _run_docker(
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


def test_default_dry_run_does_not_write_runtime_sidecars(tmp_path: Path) -> None:
    entry_path = tmp_path / "entry.json"
    completed = _run_docker(
        "--preflight-only",
        **{
            **_matching_default_dependency_env(),
            **_host_preflight_env(),
        },
    )

    assert completed.returncode != 0
    assert not entry_path.exists()
    assert "runtime_evidence" not in completed.stdout


def test_explicit_entry_sidecar_records_scoped_evidence(tmp_path: Path) -> None:
    entry_path = tmp_path / "entry.json"
    completed = _run_docker(
        "--preflight-only",
        "--compatibility-entry",
        str(entry_path),
        **{
            **_matching_default_dependency_env(),
            **_host_preflight_env(),
        },
    )

    assert completed.returncode != 0
    payload = json.loads(entry_path.read_text())
    assert payload["status"] == "not_tested"
    assert payload["observed"]["host"]["rocm_version"] == "7.1.1"
    assert payload["observed"]["container"]["image_tag"] == "7.1.1-complete"
    assert payload["observed"]["python_dependency"]["torch_version"] == (
        "2.10.0+rocm7.1"
    )
    assert payload["observed"]["dependency_policy"]["expected_local_version"] == (
        "rocm7.1"
    )
    assert payload["observed"]["toolchain"]["toolchain_rocm_version"] == "7.1.1"
    assert payload["observed"]["gpu"]["gfx_architecture"] == "gfx1200"
    assert payload["observed"]["gpu"]["visible_device_environment"] == {
        "HIP_VISIBLE_DEVICES": "0"
    }
    assert "docker run" not in completed.stdout


def test_explicit_matrix_aggregates_mixed_dependency_sidecar(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    completed = _run_docker(
        "--preflight-only",
        "--target",
        "rocm-7.2.0-ubuntu-24.04-container",
        "--compatibility-matrix",
        str(matrix_path),
        **{
            **_matching_default_dependency_env(),
            "SOL_EXECBENCH_DEPENDENCY_CONTAINER_ROCM_USER_SPACE_VERSION": "7.2.0",
            "SOL_EXECBENCH_DEPENDENCY_TOOLCHAIN_ROCM_VERSION": "7.2.0",
            **_host_preflight_env(),
        },
    )

    assert completed.returncode != 0
    report = json.loads(matrix_path.read_text())
    assert report["status_counts"] == {"mixed_version": 1}
    assert report["entries"][0]["status"] == "mixed_version"
    assert report["entries"][0]["artifacts"][0]["kind"] == (
        "runtime_evidence_dependency"
    )
    assert "docker build" not in completed.stdout
    assert "docker run" not in completed.stdout


def test_runtime_unavailable_is_sidecar_status_not_benchmark_failure(
    tmp_path: Path,
) -> None:
    entry_path = tmp_path / "entry.json"
    completed = _run_docker(
        "--compatibility-entry",
        str(entry_path),
        **{
            **_host_preflight_env(),
            "SOL_EXECBENCH_DEV_KFD_PRESENT": "false",
            "SOL_EXECBENCH_DEV_KFD_ACCESSIBLE": "false",
        },
    )

    assert completed.returncode != 0
    wrapper_payload = json.loads(completed.stdout)
    sidecar_payload = json.loads(entry_path.read_text())
    assert wrapper_payload["status"] == "runtime_unavailable"
    assert sidecar_payload["status"] == "runtime_unavailable"
    assert sidecar_payload["reason_code"] == "rocm_runtime_unavailable"
    assert sidecar_payload["artifacts"][0]["kind"] == "runtime_evidence_setup_runtime"
    assert "traces.json" not in sidecar_payload


def test_script_syntax_is_valid() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(RUN_DOCKER_SCRIPT)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
