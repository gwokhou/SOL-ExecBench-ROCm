# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Hold the host GPU/clock leases while an unprivileged container runs."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from functools import partial

from sol_execbench.core.bench.clock_lock import (
    ClockLockLease,
    ClockSetupError,
    observe_clock_levels,
    set_peak_clocks,
    unlock_clocks,
)
from sol_execbench.core.bench.clock_ownership import (
    ENV_CLOCK_OWNER,
    ClockOwnershipError,
    clock_owner_path,
    observe_clock_devices,
    owned_clock_window,
)
from sol_execbench.core.bench.gpu_lock import (
    acquire_gpu_lock,
    gpu_lock_directory,
)
from sol_execbench.core.platform.clock_commands import (
    ENV_HOST_CLOCK_GPU_UUID,
    ClockLevel,
    clock_arguments,
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


@contextmanager
def _target_clock_window(
    gpu_uuid: str, *, container_supervised: bool
) -> Iterator[ClockLockLease]:
    """Own one verified UUID; require initial AUTO and successful activation."""
    gpu_uuid = gpu_uuid.strip().lower()
    clock_arguments(ClockLevel.PEAK, gpu_uuid)
    devices = tuple(
        device
        for device in observe_clock_devices()
        if device.uuid.lower() == gpu_uuid
    )
    if len(devices) != 1:
        raise ClockOwnershipError(
            "host_clock_identity_mismatch: UUID not unique"
        )
    verified_uuid = devices[0].uuid
    if observe_clock_levels(verified_uuid) != ("AMDSMI_DEV_PERF_LEVEL_AUTO",):
        raise ClockOwnershipError(
            "host_clock_initial_policy_unverified: target must report AUTO"
        )
    with owned_clock_window(
        gpu_lock_directory(),
        devices,
        activate=partial(set_peak_clocks, verified_uuid),
        restore=partial(unlock_clocks, verified_uuid),
        container_supervised=container_supervised,
    ) as lease:
        yield lease


@contextmanager
def host_clock_window(
    *, container_supervised: bool, gpu_uuid: str | None = None
) -> Iterator[ClockLockLease]:
    """Own a UUID or retain legacy all-device control, publishing before writes."""
    if gpu_uuid is not None:
        with _target_clock_window(
            gpu_uuid, container_supervised=container_supervised
        ) as lease:
            yield lease
        return
    levels = observe_clock_levels()
    if not levels or any(
        level != "AMDSMI_DEV_PERF_LEVEL_AUTO" for level in levels
    ):
        locked = bool(levels) and all(
            level == "AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK" for level in levels
        )
        with ClockLockLease(locked=locked, acquired=False) as lease:
            yield lease
        return
    devices = observe_clock_devices()
    if len(devices) != len(levels):
        raise ClockOwnershipError("clock_owner_device_count_mismatch")
    with ExitStack() as leases:
        try:
            lease = leases.enter_context(
                owned_clock_window(
                    gpu_lock_directory(),
                    devices,
                    activate=set_peak_clocks,
                    restore=unlock_clocks,
                    container_supervised=container_supervised,
                )
            )
        except ClockSetupError:
            logger.warning("STABLE_PEAK setup failed; AUTO rollback verified")
            lease = ClockLockLease(locked=False, acquired=False)
        yield lease


def run_with_host_clock_guard(
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    runner: AttachedRunner = _default_runner,
    gpu_uuid: str | None = None,
) -> int:
    """Run ``command`` while retaining host GPU serialization and clock state."""
    if not command:
        raise ValueError("a child command is required")
    base_environment = dict(os.environ if environment is None else environment)
    timeout_seconds = _lock_timeout_seconds(base_environment)
    if gpu_uuid is None:
        gpu_uuid = base_environment.get(ENV_HOST_CLOCK_GPU_UUID)

    try:
        with (
            cleanup_on_termination(),
            acquire_gpu_lock(timeout_seconds=timeout_seconds),
            host_clock_window(
                container_supervised=is_docker_run(command), gpu_uuid=gpu_uuid
            ) as clock_lease,
        ):
            child_environment: dict[str, str] = {
                **base_environment,
                **_HOST_MANAGED_ENV,
                ENV_SOL_EXECBENCH_CLOCKS_LOCKED: "1"
                if clock_lease.locked
                else "0",
            }
            child_environment.pop(ENV_CLOCK_OWNER, None)
            if gpu_uuid is not None:
                child_environment[ENV_HOST_CLOCK_GPU_UUID] = (
                    gpu_uuid.strip().lower()
                )
            if clock_lease.acquired:
                child_environment[ENV_CLOCK_OWNER] = os.environ[ENV_CLOCK_OWNER]
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
    parser.add_argument(
        "--gpu-uuid",
        help=(
            "Require owned STABLE_PEAK for one AMD-SMI UUID; defaults to "
            f"${ENV_HOST_CLOCK_GPU_UUID}, otherwise legacy all-device control"
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
        return run_with_host_clock_guard(command, gpu_uuid=args.gpu_uuid)
    except ValueError as exc:
        _parser().error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
