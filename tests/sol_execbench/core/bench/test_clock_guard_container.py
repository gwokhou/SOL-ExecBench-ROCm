# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Docker daemon cleanup must precede releasing host clock policy."""

import subprocess
from pathlib import Path

import pytest

from sol_execbench.tools import docker_process

CONTAINER = "a" * 64


@pytest.mark.parametrize(
    "interrupted", [False, True], ids=["exit", "interrupt"]
)
def test_container_cleanup_after_attached_child_unwinds(
    monkeypatch, interrupted
):
    events = []

    def attached(command, **_kwargs):
        Path(command[command.index("--cidfile") + 1]).write_text(CONTAINER)
        events.append("child-cleaned")
        if interrupted:
            raise KeyboardInterrupt
        return subprocess.CompletedProcess(command, 0)

    def bounded(command, **_kwargs):
        assert command == ["docker", "rm", "--force", CONTAINER]
        events.append("container-removed")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(docker_process, "run_attached_process_group", attached)
    monkeypatch.setattr(docker_process, "run_in_process_group_bounded", bounded)
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            docker_process.run_owned_docker_process(
                ["docker", "run", "image"], {}
            )
    else:
        assert (
            docker_process.run_owned_docker_process(
                ["docker", "run", "image"], {}
            ).returncode
            == 0
        )
    assert events == ["child-cleaned", "container-removed"]


@pytest.mark.parametrize(
    "daemon_available", [False, True], ids=["unreachable", "already-removed"]
)
def test_failed_remove_requires_successful_absence_check(
    monkeypatch, tmp_path, daemon_available
):
    cidfile = tmp_path / "id"
    cidfile.write_text(CONTAINER)

    def bounded(command, **_kwargs):
        code = int(command[1] == "rm" or not daemon_available)
        return subprocess.CompletedProcess(command, code, "", "")

    monkeypatch.setattr(docker_process, "run_in_process_group_bounded", bounded)
    if daemon_available:
        docker_process._remove_owned_container("docker", cidfile)
    else:
        with pytest.raises(
            docker_process.ContainerCleanupError, match="cleanup_unverified"
        ):
            docker_process._remove_owned_container("docker", cidfile)
