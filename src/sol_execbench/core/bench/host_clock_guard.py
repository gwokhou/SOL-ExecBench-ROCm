# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Hold the host GPU/clock leases while an unprivileged container runs."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence

from sol_execbench.core.bench.clock_lock import acquire_clock_lock
from sol_execbench.core.bench.clock_ownership import (
    ENV_CLOCK_OWNER,
    clock_owner_path,
    published_clock_owner,
)
from sol_execbench.core.bench.gpu_lock import (
    acquire_gpu_lock,
    gpu_lock_directory,
)
from sol_execbench.core.process.environment import (
    ENV_SOL_EXECBENCH_CLOCKS_LOCKED,
)
from sol_execbench.core.process.termination import cleanup_on_termination
from sol_execbench.tools.docker_process import (
    CONTAINER_CLEANUP_BLOCKER_PREFIX,
    CONTAINER_ID_SUFFIX,
    ContainerCleanupError,
    is_docker_run,
    run_owned_docker_process,
)

logger = logging.getLogger(__name__)

_GPU_LOCK_TIMEOUT_ENV = "SOL_EXECBENCH_GPU_LOCK_TIMEOUT_SECONDS"
_HOST_MANAGED_ENV = {
    "SOL_EXECBENCH_CLOCKS_MANAGED_BY_HOST": "1",
    "SOL_EXECBENCH_GPU_LOCK_MANAGED_BY_HOST": "1",
}
_GPU_BUSY_EXIT_CODE = 75

AttachedRunner = Callable[
    [Sequence[str], Mapping[str, str]],
    subprocess.CompletedProcess[str],
]


def _default_runner(
    command: Sequence[str],
    env: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    owner = env.get(ENV_CLOCK_OWNER)
    cidfile = (
        clock_owner_path(gpu_lock_directory(), owner).with_suffix(
            CONTAINER_ID_SUFFIX
        )
        if owner is not None
        else None
    )
    return run_owned_docker_process(command, env, cidfile=cidfile)


def _lock_timeout_seconds(environment: Mapping[str, str]) -> float:
    raw_value = environment.get(_GPU_LOCK_TIMEOUT_ENV, "60")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{_GPU_LOCK_TIMEOUT_ENV} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{_GPU_LOCK_TIMEOUT_ENV} must be positive")
    return value


def run_with_host_clock_guard(
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    runner: AttachedRunner = _default_runner,
) -> int:
    """Run ``command`` while retaining host GPU serialization and clock state."""
    if not command:
        raise ValueError("a child command is required")
    base_environment = dict(os.environ if environment is None else environment)
    timeout_seconds = _lock_timeout_seconds(base_environment)

    try:
        with (
            cleanup_on_termination(),
            acquire_gpu_lock(timeout_seconds=timeout_seconds),
        ):
            clock_lease = acquire_clock_lock()
            with published_clock_owner(
                clock_lease,
                gpu_lock_directory(),
                container_supervised=is_docker_run(command),
            ) as owner:
                child_environment: dict[str, str] = {
                    **base_environment,
                    **_HOST_MANAGED_ENV,
                    ENV_SOL_EXECBENCH_CLOCKS_LOCKED: "1"
                    if clock_lease.locked
                    else "0",
                }
                child_environment.pop(ENV_CLOCK_OWNER, None)
                if owner is not None:
                    child_environment[ENV_CLOCK_OWNER] = owner
                if not clock_lease.locked:
                    logger.warning(
                        "Host clock locking unavailable; child will run with "
                        "SOL_EXECBENCH_CLOCKS_LOCKED=0",
                    )
                try:
                    completed = runner(command, child_environment)
                except ContainerCleanupError as error:
                    # Never change clocks under a possibly live container.
                    clock_lease.release_blocker = f"{CONTAINER_CLEANUP_BLOCKER_PREFIX}{error.container_id or 'unknown'}"
                    raise
                return completed.returncode
    except TimeoutError as exc:
        logger.error("%s", exc)
        return _GPU_BUSY_EXIT_CODE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hold the host GPU lock and verified clock lease around one child process"
        ),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by ``scripts/run_docker.sh``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        _parser().error("a child command is required after --")
    try:
        return run_with_host_clock_guard(command)
    except ValueError as exc:
        _parser().error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
