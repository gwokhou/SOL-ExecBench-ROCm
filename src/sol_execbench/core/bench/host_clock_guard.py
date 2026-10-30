# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Hold the host GPU/clock leases while an unprivileged container runs."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from sol_execbench.core.bench.clock_lock import (
    ClockLockLease,
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
from sol_execbench.core.bench.gpu_admission import (
    AdmissionDevice,
    AdmissionError,
    configured_admission,
)
from sol_execbench.core.bench.gpu_lock import (
    acquire_gpu_lock,
    gpu_lock_directory,
)
from sol_execbench.core.bench.gpu_lock_paths import (
    LEGACY_GPU_LOCK_ENV,
    configured_gpu_locks,
    normalize_gpu_uuid,
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
    "GPU_LOCK_HELD_BY_HOST": "1",
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
    gpu_uuid: str, *, container_supervised: bool, natural_clocks: bool = False
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
    verified_uuid = devices[0].uuid.lower()
    admission = configured_admission()
    if admission is not None:
        admission.require_ready(
            AdmissionDevice(
                pci_bdf=devices[0].bdf.lower(), smi_uuid=verified_uuid.lower()
            )
        )
    if observe_clock_levels(verified_uuid) != ("AMDSMI_DEV_PERF_LEVEL_AUTO",):
        raise ClockOwnershipError(
            "host_clock_initial_policy_unverified: target must report AUTO"
        )
    with owned_clock_window(
        gpu_lock_directory(),
        devices,
        activate=None
        if natural_clocks
        else partial(set_peak_clocks, verified_uuid),
        restore=(
            lambda: (
                observe_clock_levels(verified_uuid)
                == ("AMDSMI_DEV_PERF_LEVEL_AUTO",)
            )
        )
        if natural_clocks
        else partial(unlock_clocks, verified_uuid),
        container_supervised=container_supervised,
    ) as lease:
        yield lease


@contextmanager
def host_clock_window(
    *,
    container_supervised: bool,
    gpu_uuid: str | None = None,
    natural_clocks: bool = False,
) -> Iterator[ClockLockLease]:
    """Own only the explicitly configured UUID; never fall back to all GPUs."""
    paths = configured_gpu_locks()
    selected = normalize_gpu_uuid(paths.uuid if gpu_uuid is None else gpu_uuid)
    if selected != paths.uuid:
        raise ClockOwnershipError(
            "host clock UUID conflicts with GPU_LOCK_UUID"
        )
    with _target_clock_window(
        selected,
        container_supervised=container_supervised,
        natural_clocks=natural_clocks,
    ) as lease:
        yield lease


def run_with_host_clock_guard(
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    runner: AttachedRunner = _default_runner,
    gpu_uuid: str | None = None,
    natural_clocks: bool = False,
) -> int:
    """Run ``command`` while retaining host GPU serialization and clock state."""
    if not command:
        raise ValueError("a child command is required")
    base_environment = dict(os.environ if environment is None else environment)
    timeout_seconds = _lock_timeout_seconds(base_environment)
    paths = configured_gpu_locks()
    if any(key in base_environment for key in LEGACY_GPU_LOCK_ENV):
        raise ValueError("legacy GPU lock controls in child environment")
    if (
        "GPU_LOCK_UUID" in base_environment
        and normalize_gpu_uuid(base_environment["GPU_LOCK_UUID"]) != paths.uuid
    ):
        raise ValueError("child GPU_LOCK_UUID conflicts with native owner")
    if (
        "GPU_LOCK_DIR" in base_environment
        and Path(base_environment["GPU_LOCK_DIR"]) != paths.directory
    ):
        raise ValueError("child GPU_LOCK_DIR conflicts with native owner")
    selected = (
        gpu_uuid or base_environment.get(ENV_HOST_CLOCK_GPU_UUID) or paths.uuid
    )
    gpu_uuid = normalize_gpu_uuid(selected)
    if gpu_uuid != paths.uuid:
        raise AdmissionError("gpu_admission_device_mismatch")
    device = _admitted_device(gpu_uuid)

    try:
        with (
            cleanup_on_termination(),
            _guard_gpu_lock(timeout_seconds, device),
            host_clock_window(
                container_supervised=is_docker_run(command),
                gpu_uuid=gpu_uuid,
                natural_clocks=natural_clocks,
            ) as clock_lease,
        ):
            child_environment: dict[str, str] = {
                **base_environment,
                **_HOST_MANAGED_ENV,
                "GPU_LOCK_UUID": paths.uuid,
                "GPU_LOCK_DIR": str(paths.directory),
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


def _admitted_device(gpu_uuid: str | None) -> AdmissionDevice | None:
    if gpu_uuid is None:
        raise AdmissionError("gpu_admission_explicit_uuid_required")
    selected = tuple(
        d for d in observe_clock_devices() if d.uuid.lower() == gpu_uuid.lower()
    )
    if len(selected) != 1:
        raise AdmissionError("gpu_admission_device_identity_unverified")
    return AdmissionDevice(
        pci_bdf=selected[0].bdf.lower(), smi_uuid=selected[0].uuid.lower()
    )


@contextmanager
def _guard_gpu_lock(
    timeout: float, device: AdmissionDevice | None
) -> Generator[None]:
    if device is None:
        with acquire_gpu_lock(timeout_seconds=timeout):
            yield
    else:
        with acquire_gpu_lock(timeout_seconds=timeout, admission_device=device):
            yield


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
            f"${ENV_HOST_CLOCK_GPU_UUID}, otherwise required $GPU_LOCK_UUID"
        ),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    parser.add_argument(
        "--natural-clocks",
        action="store_true",
        help="Keep AUTO unchanged for metadata-only container checks",
    )
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
        return run_with_host_clock_guard(
            command, gpu_uuid=args.gpu_uuid, natural_clocks=args.natural_clocks
        )
    except ValueError as exc:
        _parser().error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
