"""Dry-run-only verification of UUID file mounts and native history visibility."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from sol_execbench.core.bench.gpu_lock_paths import (
    configured_gpu_locks,
    prepare_gpu_locks,
)

SCRIPT = Path(__file__).resolve().parents[4] / "scripts/run_docker.sh"


def preview(tmp_path, *, private):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"BASH_ENV", "ENV", "SOL_EXECBENCH_CLOCK_STATE_DIR"}
    }
    env.update(
        SOL_EXECBENCH_HOST_PYTHON=sys.executable,
        SOL_EXECBENCH_RUN_DOCKER_DRY_RUN="1",
        SOL_EXECBENCH_OUTPUT_DIR=str(tmp_path / "outputs"),
    )
    if private is not None:
        env["SOL_EXECBENCH_CLOCK_STATE_DIR"] = str(private)
    return subprocess.run(
        [str(SCRIPT), "--", "true"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@pytest.mark.parametrize("private_selected", [False, True])
def test_only_two_lock_files_are_mounted_and_private_history_is_readonly(
    tmp_path, private_selected
):
    paths = configured_gpu_locks()
    prepare_gpu_locks(paths)
    private = tmp_path / "private" if private_selected else None
    result = preview(tmp_path, private=private)
    assert result.returncode == 0, result.stderr
    for path in (paths.ownership, paths.execution):
        assert (
            f"type=bind,src={path},dst=/run/lock/{path.name}" in result.stdout
        )
    assert f"src={paths.directory}," not in result.stdout
    history = private or Path("/tmp/sol-execbench-locks")
    assert f"type=bind,src={history},dst={history},readonly" in result.stdout
    assert "--entrypoint /sol-execbench/docker/entrypoint.sh" in result.stdout
    assert "PYTHONPATH=/sol-execbench/src" in result.stdout


def test_missing_execution_bind_is_rejected_without_creation(tmp_path):
    paths = configured_gpu_locks()
    prepare_gpu_locks(paths)
    paths.execution.unlink()
    result = preview(tmp_path, private=tmp_path / "private")
    assert result.returncode != 0
    assert "missing regular UUID lock bind source" in result.stderr
    assert not paths.execution.exists()
