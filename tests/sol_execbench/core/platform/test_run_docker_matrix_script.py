from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sol_execbench.core.platform.clock_commands import ENV_HOST_CLOCK_GPU_UUID
from sol_execbench.core.platform.docker_matrix import (
    load_docker_target_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DOCKERFILE_PATH = REPO_ROOT / "docker" / "Dockerfile"
ENTRYPOINT_PATH = REPO_ROOT / "docker" / "entrypoint.sh"
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

    assert "ARG ROCM_DOCKER_IMAGE=rocm/pytorch" in pre_from_lines
    assert (
        "ARG ROCM_DOCKER_TAG=rocm10.1.0_ubuntu26.04_py3.14_pytorch_release_2.14.0"
        in pre_from_lines
    )
    assert "ARG PYTORCH_TORCH_VERSION=2.14.0+rocm10.1.0" in pre_from_lines
    assert (
        "ARG PYTORCH_TORCHVISION_VERSION=0.29.0a0+rocm10.1.0" in pre_from_lines
    )
    assert any(
        "ARG ROCM_DOCKER_DIGEST=sha256:" in line for line in pre_from_lines
    )
    assert lines[first_from_index] == (
        "FROM ${ROCM_DOCKER_IMAGE}:${ROCM_DOCKER_TAG}@${ROCM_DOCKER_DIGEST} AS base"
    )


def test_dockerfile_preserves_official_gpu_stack_and_runtime_setup() -> None:
    dockerfile = DOCKERFILE_PATH.read_text()

    assert "ENV ROCM_PATH=/opt/rocm" in dockerfile
    assert "$(rocm-sdk path --root)" in dockerfile
    assert "HIP_PLATFORM=amd" in dockerfile
    assert "ARG HOST_UID=1000" in dockerfile
    assert "ARG HOST_GID=1000" in dockerfile
    assert "ARG HOST_USER=sol-execbench" in dockerfile
    assert "util-linux" in dockerfile
    assert (
        "uv venv --python python3.14 --system-site-packages /venv" in dockerfile
    )
    assert (
        "uv export --locked --no-dev --no-emit-project --no-hashes"
        in dockerfile
    )
    assert "uv pip install --python /venv/bin/python --no-deps" in dockerfile
    assert "uv sync" not in dockerfile
    assert (
        '"${PYTORCH_TORCH_VERSION}" "${PYTORCH_TORCHVISION_VERSION}"'
        in dockerfile
    )
    assert '"${TRITON_ROCM_VERSION}"' in dockerfile
    assert "/venv/bin/python -m pip check" in dockerfile
    assert (
        "COPY --from=ghcr.io/astral-sh/uv:0.11.31 /uv /usr/local/bin/uv"
        in dockerfile
    )
    assert "COPY docker/entrypoint.sh /entrypoint.sh" in dockerfile
    assert 'ENTRYPOINT ["/entrypoint.sh"]' in dockerfile


def test_dockerfile_installs_exact_validated_amd_smi_sudoers_rule() -> None:
    dockerfile = DOCKERFILE_PATH.read_text()

    assert 'AMD_SMI="$(command -v amd-smi)"' in dockerfile
    assert "NOPASSWD: ${AMD_SMI}" in dockerfile
    assert "${AMD_SMI} version" in dockerfile
    assert "${AMD_SMI} set -l STABLE_PEAK" in dockerfile
    assert "${AMD_SMI} set -l AUTO" in dockerfile
    assert "visudo -cf" in dockerfile
    assert "install -o root -g root -m 0440" in dockerfile
    assert "command -v rocm-smi" not in dockerfile
    assert "${AMD_SMI} set -l *" not in dockerfile


def test_entrypoint_registers_owned_lock_cleanup_before_locking() -> None:
    entrypoint = ENTRYPOINT_PATH.read_text(encoding="utf-8")

    cleanup_index = entrypoint.index("trap 'cleanup' EXIT")
    gpu_lock_index = entrypoint.index(
        '\nif [ "${SOL_EXECBENCH_GPU_LOCK_MANAGED_BY_HOST:-0}" != "1" ]',
    )
    clock_lock_index = entrypoint.index(
        '\nif [ "${SOL_EXECBENCH_CLOCKS_MANAGED_BY_HOST:-0}" = "1" ]',
    )
    assert cleanup_index < gpu_lock_index < clock_lock_index
    assert "SOL_EXECBENCH_GPU_LOCK_FD=9" in entrypoint
    assert "CLOCK_LOCK_ACQUIRED={int(clock_lock.acquired)}" in entrypoint
    assert "clock_lock.detach()" in entrypoint
    assert 'if [ "${SOL_EXECBENCH_CLOCK_LOCK_ACQUIRED}" = "1" ]' in entrypoint


def test_entrypoint_verifies_host_managed_clock_state_without_container_sudo() -> (
    None
):
    entrypoint = ENTRYPOINT_PATH.read_text(encoding="utf-8")

    assert "SOL_EXECBENCH_CLOCKS_MANAGED_BY_HOST" in entrypoint
    assert "SOL_EXECBENCH_GPU_LOCK_MANAGED_BY_HOST" in entrypoint
    assert "verify_host_managed_clocks" in entrypoint
    assert (
        "from sol_execbench.core.bench.clock_lock import verify_clocks"
        in entrypoint
    )
    assert "HOST_CLOCKS_DECLARED" in entrypoint


def test_entrypoint_rejects_false_host_clock_claim(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python"
    fake_python.write_text("#!/bin/bash\nexit 1\n", encoding="utf-8")
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FLASHINFER_TRACE_DIR": str(tmp_path),
        "SOL_EXECBENCH_CLOCKS_LOCKED": "1",
        "SOL_EXECBENCH_CLOCKS_MANAGED_BY_HOST": "1",
        "SOL_EXECBENCH_GPU_LOCK_MANAGED_BY_HOST": "1",
    }

    result = subprocess.run(
        [
            ENTRYPOINT_PATH,
            "bash",
            "-c",
            'test "${SOL_EXECBENCH_CLOCKS_LOCKED}" = "0"',
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "failed container verification" in result.stdout


@pytest.mark.parametrize(
    ("declared", "level", "expected"),
    [("1", "STABLE_PEAK", 0), ("1", "AUTO", 1), ("0", "STABLE_PEAK", 1)],
    ids=["target-verified", "target-readback-failed", "target-not-declared"],
)
def test_entrypoint_verifies_only_host_uuid_or_refuses_workload(
    tmp_path: Path, declared: str, level: str, expected: int
) -> None:
    gpu_uuid = "a3007590-0000-1000-800f-a29c1cca1511"
    tool = tmp_path / "amd-smi"
    payload = {
        "gpu_data": [{"gpu": 0, "perf_level": f"AMDSMI_DEV_PERF_LEVEL_{level}"}]
    }
    tool.write_text(
        f"#!{sys.executable}\nimport sys\n"
        f"assert sys.argv[1:] == ['metric', '-l', '--json', '--gpu', {gpu_uuid!r}]\n"
        f"print({json.dumps(payload)!r})\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    python = tmp_path / "python"
    python.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    marker = tmp_path / "workload-started"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "FLASHINFER_TRACE_DIR": str(tmp_path),
        "SOL_EXECBENCH_CLOCKS_LOCKED": declared,
        "SOL_EXECBENCH_CLOCKS_MANAGED_BY_HOST": "1",
        "SOL_EXECBENCH_GPU_LOCK_MANAGED_BY_HOST": "1",
        ENV_HOST_CLOCK_GPU_UUID: gpu_uuid,
    }
    result = subprocess.run(
        [str(ENTRYPOINT_PATH), "touch", str(marker)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == expected, result.stderr
    assert marker.exists() == (expected == 0)


def test_entrypoint_releases_owned_lock_after_workload_failure(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cleanup_log = tmp_path / "cleanup.log"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        """#!/bin/bash
if [[ ${1:-} == -c ]]; then
  printf 'cleanup\n' >>"${CLEANUP_LOG}"
  exit 0
fi
printf 'CLOCKS_LOCKED=1\nCLOCK_LOCK_ACQUIRED=1\n'
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CLEANUP_LOG": str(cleanup_log),
        "FLASHINFER_TRACE_DIR": str(tmp_path),
        "SOL_EXECBENCH_GPU_LOCK_DIR": str(tmp_path / "locks"),
    }

    result = subprocess.run(
        [ENTRYPOINT_PATH, "bash", "-c", "exit 23"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 23
    assert cleanup_log.read_text(encoding="utf-8") == "cleanup\n"


def test_entrypoint_holds_gpu_lock_until_clock_cleanup_finishes(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python"
    fake_python.write_text(
        """#!/bin/bash
if [[ ${1:-} == -c ]]; then
  touch "${CLEANUP_STARTED}"
  while [[ ! -e "${CLEANUP_RELEASE}" ]]; do sleep 0.01; done
  exit 0
fi
printf 'CLOCKS_LOCKED=1\nCLOCK_LOCK_ACQUIRED=1\n'
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    cleanup_started = tmp_path / "cleanup.started"
    cleanup_release = tmp_path / "cleanup.release"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CLEANUP_STARTED": str(cleanup_started),
        "CLEANUP_RELEASE": str(cleanup_release),
        "FLASHINFER_TRACE_DIR": str(tmp_path),
        "SOL_EXECBENCH_GPU_LOCK_DIR": str(tmp_path / "locks"),
    }
    first = subprocess.Popen(
        [ENTRYPOINT_PATH, "true"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not cleanup_started.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert cleanup_started.exists()

        second = subprocess.run(
            [ENTRYPOINT_PATH, "true"],
            env={**env, "SOL_EXECBENCH_GPU_LOCK_TIMEOUT_SECONDS": "0.05"},
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert second.returncode == 75
        assert "GPU 0 is busy" in second.stderr
    finally:
        cleanup_release.touch()
        first.communicate(timeout=5)
    assert first.returncode == 0


def test_container_validation_sidecar_uses_writable_output_transport() -> None:
    script = RUN_DOCKER_SCRIPT.read_text(encoding="utf-8")

    assert 'container_entry_path="/outputs/${transport_path##*/}"' in script
    assert 'mv "${transport_path}" "${entry_path}"' in script
    assert 'container_repo_path "${entry_path}"' not in script


def _run_docker_preview(*args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "SOL_EXECBENCH_RUN_DOCKER_DRY_RUN": "1",
        "SOL_EXECBENCH_HOST_PYTHON": sys.executable,
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


def test_run_docker_help_mentions_target_flags() -> None:
    script = RUN_DOCKER_SCRIPT.read_text()

    assert "--target <id>" in script
    assert "--allow-unknown-target" in script
    assert "--preflight-only" in script
    assert "--allow-untested-target-smoke" in script
    assert "--record-container-validation" in script


def test_run_docker_host_helpers_use_uv_managed_python() -> None:
    script = RUN_DOCKER_SCRIPT.read_text()

    assert "uv run python" in script
    assert "SOL_EXECBENCH_HOST_PYTHON" in script
    assert not re.search(
        r"^\s*python\s+-m\s+sol_execbench",
        script,
        re.MULTILINE,
    )
    assert not re.search(r"^\s*python\s+-c\b", script, re.MULTILINE)


@pytest.mark.parametrize(
    ("system", "machine"),
    [("Darwin", "x86_64"), ("Linux", "aarch64"), ("Linux", "i686")],
    ids=["macos", "linux-arm", "linux-32bit"],
)
def test_run_docker_rejects_unsupported_build_platforms(
    tmp_path: Path, system: str, machine: str
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uname = bin_dir / "uname"
    uname.write_text(
        "#!/bin/bash\n"
        f'case "$1" in -s) echo {system};; -m) echo {machine};; esac\n',
        encoding="utf-8",
    )
    uname.chmod(0o755)
    completed = subprocess.run(
        [str(RUN_DOCKER_SCRIPT), "--build"],
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 2
    assert "only on Linux x86-64" in completed.stderr
    assert "docker build" not in completed.stdout


def test_run_docker_holds_host_clock_guard_around_container() -> None:
    script = RUN_DOCKER_SCRIPT.read_text()

    assert "sol_execbench.core.bench.host_clock_guard" in script
    assert "-e SOL_EXECBENCH_CLOCKS_LOCKED" in script
    assert "-e SOL_EXECBENCH_CLOCKS_MANAGED_BY_HOST" in script
    assert "-e SOL_EXECBENCH_GPU_LOCK_MANAGED_BY_HOST" in script
    assert f"-e {ENV_HOST_CLOCK_GPU_UUID}" in script
    assert "docker image inspect --format '{{.Id}}'" in script
    assert "SOL_EXECBENCH_CONTAINER_IMAGE_ID" in script


@pytest.mark.requires_linux
@pytest.mark.subprocess_uv
def test_run_docker_default_build_preview_uses_rocm_10_1_build_args() -> None:
    completed = _run_docker_preview("--build")

    assert completed.returncode == 0, completed.stderr
    assert "docker build -t sol-execbench:rocm-10.1.0" in completed.stdout
    assert '--build-arg ROCM_DOCKER_IMAGE="rocm/pytorch"' in completed.stdout
    assert (
        '--build-arg ROCM_DOCKER_TAG="rocm10.1.0_ubuntu26.04_py3.14_pytorch_release_2.14.0"'
        in completed.stdout
    )
    assert (
        '--build-arg PYTORCH_TORCH_VERSION="2.14.0+rocm10.1.0"'
        in completed.stdout
    )
    assert (
        '--build-arg PYTORCH_TORCHVISION_VERSION="0.29.0a0+rocm10.1.0"'
        in completed.stdout
    )
    assert (
        '--build-arg TRITON_ROCM_VERSION="3.8.0+git669b31ac.rocm10.1.0"'
        in completed.stdout
    )
    assert "--device=/dev/kfd" in completed.stdout
    assert "--device=/dev/dri" in completed.stdout
    assert "--group-add video" in completed.stdout
    assert "--security-opt no-new-privileges=true" in completed.stdout
    assert "--security-opt seccomp=unconfined" not in completed.stdout
    assert "--pids-limit=512" in completed.stdout
    assert "--ipc=private" in completed.stdout
    assert "--shm-size=8g" in completed.stdout
    assert "--privileged" not in completed.stdout


@pytest.mark.requires_linux
@pytest.mark.subprocess_uv
def test_run_docker_declared_target_build_preview_uses_manifest_build_args() -> (
    None
):
    manifest = load_docker_target_manifest(MANIFEST_PATH)
    selected = manifest.targets_by_id[manifest.default_target_id]

    completed = _run_docker_preview(
        "--build",
        "--target",
        selected.target_id,
    )

    assert completed.returncode == 0, completed.stderr
    assert "docker build -t sol-execbench:rocm-10.1.0" in completed.stdout
    assert (
        f'--build-arg ROCM_DOCKER_IMAGE="{selected.docker_image_repository}"'
        in completed.stdout
    )
    assert (
        f'--build-arg ROCM_DOCKER_TAG="{selected.docker_image_tag}"'
        in completed.stdout
    )
    assert selected.pytorch_dependency_policy is not None
    assert (
        '--build-arg PYTORCH_TORCH_VERSION="'
        f'{selected.pytorch_dependency_policy["torch_version"]}"'
        in completed.stdout
    )
    assert (
        '--build-arg PYTORCH_TORCHVISION_VERSION="'
        f'{selected.pytorch_dependency_policy["torchvision_version"]}"'
        in completed.stdout
    )


@pytest.mark.requires_linux
@pytest.mark.subprocess_uv
def test_run_docker_target_flag_is_not_forwarded_to_docker_args_or_command() -> (
    None
):
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
        line
        for line in completed.stdout.splitlines()
        if line.startswith("+ docker run")
    )
    assert "--name matrix-test" in run_line
    assert "echo inside-container" in run_line
    assert "sol-execbench:rocm-10.1.0" in run_line
    assert "--target" not in run_line


@pytest.mark.requires_linux
@pytest.mark.subprocess_uv
def test_run_docker_image_tag_env_overrides_target_tag() -> None:
    completed = _run_docker_preflight(
        "--build",
        IMAGE_TAG="custom",
    )

    assert completed.returncode == 0, completed.stderr
    assert "docker build -t sol-execbench:custom" in completed.stdout
    assert "sol-execbench:custom" in completed.stdout


@pytest.mark.requires_linux
@pytest.mark.subprocess_uv
def test_run_docker_unknown_target_rejected_before_docker_commands() -> None:
    completed = _run_docker_preview("--build", "--target", "rocm-9.9-unknown")

    assert completed.returncode != 0
    assert "Unknown Docker Target" in completed.stderr
    assert "docker build" not in completed.stdout
    assert "docker run" not in completed.stdout


@pytest.mark.requires_linux
@pytest.mark.subprocess_uv
def test_run_docker_preflight_only_emits_runtime_unavailable_diagnostics() -> (
    None
):
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
    assert payload["image_repository"] == "rocm/pytorch"
    assert (
        payload["image_tag"]
        == "rocm10.1.0_ubuntu26.04_py3.14_pytorch_release_2.14.0"
    )
    assert "image_digest" in payload
    assert payload["image_digest"] is None
    assert payload["build_args"]["ROCM_DOCKER_IMAGE"] == "rocm/pytorch"
    assert (
        payload["build_args"]["ROCM_DOCKER_TAG"]
        == "rocm10.1.0_ubuntu26.04_py3.14_pytorch_release_2.14.0"
    )
    assert payload["benchmark_allowed"] is False
    assert payload["score_authority"] is False
    assert payload["paper_parity_authority"] is False
    assert payload["leaderboard_authority"] is False
    assert "docker build" not in completed.stdout
    assert "docker run" not in completed.stdout


@pytest.mark.requires_linux
@pytest.mark.subprocess_uv
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


@pytest.mark.requires_linux
@pytest.mark.subprocess_uv
def test_run_docker_preflight_only_available_exits_without_build_or_run() -> (
    None
):
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

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "not_tested"
    assert payload["benchmark_allowed"] is False
    assert payload["container_user_space_validated"] is False
    assert payload["native_host_validated"] is False
    assert "docker build" not in completed.stdout
    assert "docker run" not in completed.stdout


@pytest.mark.requires_linux
@pytest.mark.subprocess_uv
def test_run_docker_preflight_only_allows_explicit_not_tested_smoke() -> None:
    completed = _run_docker_preflight(
        "--preflight-only",
        "--allow-untested-target-smoke",
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


@pytest.mark.requires_linux
@pytest.mark.subprocess_uv
def test_run_docker_not_tested_preflight_blocks_normal_run() -> None:
    completed = _run_docker_preflight(
        "--",
        "sol-execbench",
        "tests/sol_execbench/samples/rmsnorm",
        SOL_EXECBENCH_DOCKER_CONTEXT="default",
        SOL_EXECBENCH_DOCKER_HOST="unix:///var/run/docker.sock",
        SOL_EXECBENCH_DEV_KFD_PRESENT="true",
        SOL_EXECBENCH_DEV_KFD_ACCESSIBLE="true",
        SOL_EXECBENCH_DEV_DRI_PRESENT="true",
        SOL_EXECBENCH_DEV_DRI_ACCESSIBLE="true",
        SOL_EXECBENCH_GPU_ACCESSIBLE="true",
    )

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "not_tested"
    assert payload["benchmark_allowed"] is False
    assert "docker run" not in completed.stdout


@pytest.mark.requires_linux
@pytest.mark.subprocess_uv
def test_run_docker_explicit_not_tested_smoke_reaches_dry_run_command() -> None:
    completed = _run_docker_preflight(
        "--allow-untested-target-smoke",
        "--",
        "sol-execbench",
        "tests/sol_execbench/samples/rmsnorm",
        SOL_EXECBENCH_DOCKER_CONTEXT="default",
        SOL_EXECBENCH_DOCKER_HOST="unix:///var/run/docker.sock",
        SOL_EXECBENCH_DEV_KFD_PRESENT="true",
        SOL_EXECBENCH_DEV_KFD_ACCESSIBLE="true",
        SOL_EXECBENCH_DEV_DRI_PRESENT="true",
        SOL_EXECBENCH_DEV_DRI_ACCESSIBLE="true",
        SOL_EXECBENCH_GPU_ACCESSIBLE="true",
    )

    assert completed.returncode == 0, completed.stderr
    assert "+ docker run" in completed.stdout
    assert (
        "sol-execbench tests/sol_execbench/samples/rmsnorm" in completed.stdout
    )


@pytest.mark.requires_linux
@pytest.mark.subprocess_uv
def test_run_docker_invalid_preflight_boolean_has_no_traceback() -> None:
    completed = _run_docker_preflight(
        "--preflight-only",
        SOL_EXECBENCH_DOCKER_CONTEXT="default",
        SOL_EXECBENCH_DOCKER_HOST="unix:///var/run/docker.sock",
        SOL_EXECBENCH_DEV_KFD_PRESENT="true",
        SOL_EXECBENCH_DEV_KFD_ACCESSIBLE="true",
        SOL_EXECBENCH_DEV_DRI_PRESENT="true",
        SOL_EXECBENCH_DEV_DRI_ACCESSIBLE="true",
        SOL_EXECBENCH_GPU_ACCESSIBLE="maybe",
    )

    assert completed.returncode != 0
    assert "expected boolean value" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "docker run" not in completed.stdout
