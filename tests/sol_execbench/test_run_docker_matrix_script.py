from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from sol_execbench.core.docker_matrix import load_docker_target_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE_PATH = REPO_ROOT / "docker" / "Dockerfile"
RUN_DOCKER_SCRIPT = REPO_ROOT / "scripts" / "run_docker.sh"
MANIFEST_PATH = REPO_ROOT / "docker" / "rocm-targets.json"


def _dockerfile_lines() -> list[str]:
    return DOCKERFILE_PATH.read_text().splitlines()


def test_dockerfile_declares_rocm_base_args_before_first_from() -> None:
    lines = _dockerfile_lines()
    first_from_index = next(
        index for index, line in enumerate(lines) if line.startswith("FROM ")
    )
    pre_from_lines = lines[:first_from_index]

    assert "ARG ROCM_DOCKER_IMAGE=rocm/dev-ubuntu-24.04" in pre_from_lines
    assert "ARG ROCM_DOCKER_TAG=7.1.1-complete" in pre_from_lines
    assert lines[first_from_index] == (
        "FROM ${ROCM_DOCKER_IMAGE}:${ROCM_DOCKER_TAG} AS base"
    )


def test_dockerfile_keeps_existing_runtime_setup() -> None:
    dockerfile = DOCKERFILE_PATH.read_text()

    assert "ENV ROCM_PATH=/opt/rocm" in dockerfile
    assert "HIP_PLATFORM=amd" in dockerfile
    assert "ARG HOST_UID=1000" in dockerfile
    assert "ARG HOST_GID=1000" in dockerfile
    assert "ARG HOST_USER=sol-execbench" in dockerfile
    assert "COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv" in dockerfile
    assert "COPY docker/entrypoint.sh /entrypoint.sh" in dockerfile
    assert 'ENTRYPOINT ["/entrypoint.sh"]' in dockerfile


def _run_docker_preview(*args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "SOL_EXECBENCH_RUN_DOCKER_DRY_RUN": "1",
    }
    return subprocess.run(
        [str(RUN_DOCKER_SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _run_docker_preflight(
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


def test_run_docker_help_mentions_target_flags() -> None:
    script = RUN_DOCKER_SCRIPT.read_text()

    assert "--target <id>" in script
    assert "--allow-unknown-target" in script
    assert "--preflight-only" in script


def test_run_docker_default_build_preview_uses_rocm_7_1_build_args() -> None:
    completed = _run_docker_preview("--build")

    assert completed.returncode == 0, completed.stderr
    assert '--build-arg ROCM_DOCKER_IMAGE="rocm/dev-ubuntu-24.04"' in completed.stdout
    assert '--build-arg ROCM_DOCKER_TAG="7.1.1-complete"' in completed.stdout
    assert "--device=/dev/kfd" in completed.stdout
    assert "--device=/dev/dri" in completed.stdout
    assert "--group-add video" in completed.stdout
    assert "--security-opt seccomp=unconfined" in completed.stdout
    assert "--ipc=host" in completed.stdout
    assert "--privileged" not in completed.stdout


def test_run_docker_declared_target_build_preview_uses_manifest_build_args() -> None:
    manifest = load_docker_target_manifest(MANIFEST_PATH)
    non_default = next(
        target
        for target in manifest.targets
        if target.target_id != manifest.default_target_id
    )

    completed = _run_docker_preview("--build", "--target", non_default.target_id)

    assert completed.returncode == 0, completed.stderr
    assert (
        f'--build-arg ROCM_DOCKER_IMAGE="{non_default.docker_image_repository}"'
        in completed.stdout
    )
    assert (
        f'--build-arg ROCM_DOCKER_TAG="{non_default.docker_image_tag}"'
        in completed.stdout
    )


def test_run_docker_target_flag_is_not_forwarded_to_docker_args_or_command() -> None:
    manifest = load_docker_target_manifest(MANIFEST_PATH)

    completed = _run_docker_preview(
        "--target",
        manifest.default_target_id,
        "--name",
        "matrix-test",
        "--",
        "echo",
        "inside-container",
    )

    assert completed.returncode == 0, completed.stderr
    run_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("+ docker run")
    )
    assert "--name matrix-test" in run_line
    assert "echo inside-container" in run_line
    assert "--target" not in run_line


def test_run_docker_unknown_target_rejected_before_docker_commands() -> None:
    completed = _run_docker_preview("--build", "--target", "rocm-9.9-unknown")

    assert completed.returncode != 0
    assert "Unknown Docker Target" in completed.stderr
    assert "docker build" not in completed.stdout
    assert "docker run" not in completed.stdout


def test_run_docker_preflight_only_emits_runtime_unavailable_diagnostics() -> None:
    completed = _run_docker_preflight(
        "--preflight-only",
        SOL_EXECBENCH_DOCKER_CONTEXT="desktop-linux",
        SOL_EXECBENCH_DOCKER_HOST="unix:///home/user/.docker/desktop/docker.sock",
        SOL_EXECBENCH_DEV_KFD_PRESENT="true",
        SOL_EXECBENCH_DEV_KFD_ACCESSIBLE="true",
        SOL_EXECBENCH_DEV_DRI_PRESENT="true",
        SOL_EXECBENCH_DEV_DRI_ACCESSIBLE="true",
        SOL_EXECBENCH_GPU_ACCESSIBLE="false",
    )

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "runtime_unavailable"
    assert payload["reason_code"] == "rocm_runtime_unavailable"
    assert payload["target_id"]
    assert payload["image_repository"] == "rocm/dev-ubuntu-24.04"
    assert payload["image_tag"] == "7.1.1-complete"
    assert "image_digest" in payload
    assert payload["image_digest"] is None
    assert payload["build_args"]["ROCM_DOCKER_IMAGE"] == "rocm/dev-ubuntu-24.04"
    assert payload["build_args"]["ROCM_DOCKER_TAG"] == "7.1.1-complete"
    assert payload["benchmark_allowed"] is False
    assert payload["score_authority"] is False
    assert payload["paper_parity_authority"] is False
    assert payload["leaderboard_authority"] is False
    assert "docker build" not in completed.stdout
    assert "docker run" not in completed.stdout


def test_run_docker_runtime_unavailable_skips_build_and_run() -> None:
    completed = _run_docker_preflight(
        "--build",
        SOL_EXECBENCH_DOCKER_CONTEXT="default",
        SOL_EXECBENCH_DOCKER_HOST="unix:///var/run/docker.sock",
        SOL_EXECBENCH_DEV_KFD_PRESENT="true",
        SOL_EXECBENCH_DEV_KFD_ACCESSIBLE="true",
        SOL_EXECBENCH_DEV_DRI_PRESENT="false",
        SOL_EXECBENCH_DEV_DRI_ACCESSIBLE="false",
    )

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "runtime_unavailable"
    assert payload["reason_code"] == "rocm_runtime_unavailable"
    assert "/dev/dri" in payload["reason"]
    assert "docker build" not in completed.stdout
    assert "docker run" not in completed.stdout


def test_run_docker_preflight_only_available_exits_without_build_or_run() -> None:
    completed = _run_docker_preflight(
        "--preflight-only",
        "--build",
        SOL_EXECBENCH_DOCKER_CONTEXT="default",
        SOL_EXECBENCH_DOCKER_HOST="unix:///var/run/docker.sock",
        SOL_EXECBENCH_DEV_KFD_PRESENT="true",
        SOL_EXECBENCH_DEV_KFD_ACCESSIBLE="true",
        SOL_EXECBENCH_DEV_DRI_PRESENT="true",
        SOL_EXECBENCH_DEV_DRI_ACCESSIBLE="true",
        SOL_EXECBENCH_GPU_ACCESSIBLE="true",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "not_tested"
    assert payload["benchmark_allowed"] is False
    assert payload["container_user_space_validated"] is False
    assert payload["native_host_validated"] is False
    assert "docker build" not in completed.stdout
    assert "docker run" not in completed.stdout
