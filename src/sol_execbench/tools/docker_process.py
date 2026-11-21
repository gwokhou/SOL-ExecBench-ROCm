# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Attached Docker run with daemon-side cleanup before host resource release.

The process runners are the test seam: tests emulate cidfile creation and
daemon replies without Docker or GPU access.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from sol_execbench.core.process import run_attached_process_group
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from sol_execbench.core.process.termination import defer_termination

CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
CONTAINER_CLEANUP_TIMEOUT = 30
CONTAINER_CLEANUP_OUTPUT_BYTES = 4096
CONTAINER_CLEANUP_BLOCKER_PREFIX = "container_cleanup_unverified:"
CONTAINER_ID_SUFFIX = ".container-id"
CONTAINER_NAME_PREFIX = "sol-execbench-"
_LEASE_TOKEN_PATTERN = re.compile(r"[0-9a-f]{32}")
_OWNED_CONTAINER_NAME_PATTERN = re.compile(r"sol-execbench-[0-9a-f]{32}")


class ContainerCleanupError(RuntimeError):
    """Container disappearance cannot be confirmed; retain GPU recovery gate."""

    def __init__(self, reason: str, container_id: str | None = None) -> None:
        """Retain only the owned ID, never potentially secret command arguments."""
        self.container_id = container_id
        super().__init__(f"{reason}: {container_id or 'unknown'}")


def owned_container_name(lease_id: str) -> str:
    """Return deterministic container name bound to one 32-hex clock lease."""
    if _LEASE_TOKEN_PATTERN.fullmatch(lease_id) is None:
        raise ContainerCleanupError("host_container_identity_unverified")
    return f"{CONTAINER_NAME_PREFIX}{lease_id}"


def confirm_container_absent(executable: str, identity: str) -> None:
    """Require a successful daemon query showing this exact container is gone."""
    if CONTAINER_ID_PATTERN.fullmatch(identity) is None:
        raise ContainerCleanupError("host_container_identity_unverified")
    listed = run_in_process_group_bounded(
        [
            executable,
            "ps",
            "--all",
            "--quiet",
            "--no-trunc",
            "--filter",
            f"id={identity}",
        ],
        timeout=CONTAINER_CLEANUP_TIMEOUT,
        max_capture_bytes=CONTAINER_CLEANUP_OUTPUT_BYTES,
    )
    if listed.returncode or listed.stdout.strip():
        raise ContainerCleanupError(
            "host_container_cleanup_unverified", identity
        )


def confirm_container_name_absent(executable: str, container_name: str) -> None:
    """Require a successful daemon query showing the deterministic name is gone."""
    if _OWNED_CONTAINER_NAME_PATTERN.fullmatch(container_name) is None:
        raise ContainerCleanupError("host_container_identity_unverified")
    try:
        listed = run_in_process_group_bounded(
            [
                executable,
                "ps",
                "--all",
                "--quiet",
                "--no-trunc",
                "--filter",
                f"name=^/{container_name}$",
            ],
            timeout=CONTAINER_CLEANUP_TIMEOUT,
            max_capture_bytes=CONTAINER_CLEANUP_OUTPUT_BYTES,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContainerCleanupError(
            "host_container_cleanup_unverified"
        ) from error
    if listed.returncode or listed.stdout.strip():
        found = listed.stdout.strip().splitlines()
        found_id = (
            found[0].strip()
            if found and CONTAINER_ID_PATTERN.fullmatch(found[0].strip())
            else None
        )
        raise ContainerCleanupError(
            "host_container_cleanup_unverified", found_id
        )


def _remove_by_identity(executable: str, identity: str) -> None:
    try:
        removed = run_in_process_group_bounded(
            [executable, "rm", "--force", identity],
            timeout=CONTAINER_CLEANUP_TIMEOUT,
            max_capture_bytes=CONTAINER_CLEANUP_OUTPUT_BYTES,
        )
        if removed.returncode == 0:
            return
        confirm_container_absent(executable, identity)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContainerCleanupError(
            "host_container_cleanup_unverified", identity
        ) from error


def _remove_missing_cidfile_container(
    executable: str, container_name: str | None
) -> None:
    if container_name is None:
        raise ContainerCleanupError("host_container_identity_unverified")
    try:
        listed = run_in_process_group_bounded(
            [
                executable,
                "ps",
                "--all",
                "--quiet",
                "--no-trunc",
                "--filter",
                f"name=^/{container_name}$",
            ],
            timeout=CONTAINER_CLEANUP_TIMEOUT,
            max_capture_bytes=CONTAINER_CLEANUP_OUTPUT_BYTES,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContainerCleanupError(
            "host_container_cleanup_unverified"
        ) from error
    if listed.returncode != 0:
        raise ContainerCleanupError("host_container_cleanup_unverified")
    output = listed.stdout.strip()
    if not output:
        return
    ids = [line.strip() for line in output.splitlines() if line.strip()]
    if len(ids) != 1 or CONTAINER_ID_PATTERN.fullmatch(ids[0]) is None:
        raise ContainerCleanupError("host_container_cleanup_unverified")
    _remove_by_identity(executable, ids[0])


def _remove_owned_container(
    executable: str, cidfile: Path, *, container_name: str | None = None
) -> None:
    if not cidfile.exists():
        _remove_missing_cidfile_container(executable, container_name)
        return
    identity = cidfile.read_text().strip()
    if not identity and container_name is not None:
        _remove_missing_cidfile_container(executable, container_name)
        return
    if CONTAINER_ID_PATTERN.fullmatch(identity) is None:
        raise ContainerCleanupError("host_container_identity_unverified")
    _remove_by_identity(executable, identity)


def run_owned_docker_process(
    command: Sequence[str],
    environment: Mapping[str, str],
    *,
    cidfile: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run Docker foreground client and remove only its recorded container ID."""
    if not is_docker_run(command):
        return run_attached_process_group(command, env=environment)
    if any(
        item == "--cidfile" or item.startswith("--cidfile=") for item in command
    ):
        raise ValueError(
            "host_guard_owns_cidfile: remove the user-supplied --cidfile"
        )
    if cidfile is not None:
        return _run_with_cidfile(command, environment, cidfile)
    with TemporaryDirectory(prefix="sol-clock-container-") as temporary:
        return _run_with_cidfile(
            command,
            environment,
            Path(temporary) / f"{uuid4().hex}{CONTAINER_ID_SUFFIX}",
        )


def is_docker_run(command: Sequence[str]) -> bool:
    """Recognize the wrapper's direct docker run command."""
    return (
        len(command) >= 2
        and Path(command[0]).name == "docker"
        and command[1] == "run"
    )


def _has_container_name_flag(command: Sequence[str]) -> bool:
    return any(
        item == "--name" or item.startswith("--name=") for item in command[2:]
    )


def _run_with_cidfile(
    command: Sequence[str], environment: Mapping[str, str], cidfile: Path
) -> subprocess.CompletedProcess[str]:
    if cidfile.exists():
        raise ContainerCleanupError("host_container_cidfile_already_exists")
    container_name = (
        owned_container_name(cidfile.stem)
        if not _has_container_name_flag(command)
        and _LEASE_TOKEN_PATTERN.fullmatch(cidfile.stem) is not None
        else None
    )
    injected = (
        ["--name", container_name, "--cidfile", str(cidfile)]
        if container_name is not None
        else ["--cidfile", str(cidfile)]
    )
    owned = [*command[:2], *injected, *command[2:]]
    try:
        return run_attached_process_group(owned, env=environment)
    finally:
        with defer_termination():
            _remove_owned_container(
                command[0], cidfile, container_name=container_name
            )
