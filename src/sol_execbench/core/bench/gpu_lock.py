# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""UUID L1 ownership and L2 execution locks for native GPU lifecycles."""

from __future__ import annotations

import contextlib
import fcntl
import os
import threading
import time
from collections.abc import Iterator
from contextvars import ContextVar
from pathlib import Path

from sol_execbench.core.bench.clock_ownership import (
    ENV_CLOCK_OWNER,
    ClockOwnerRecord,
    clock_owner_path,
    require_no_clock_recovery,
    verify_clock_owner,
)
from sol_execbench.core.bench.gpu_admission import (
    AdmissionDevice,
    AdmissionError,
    configured_admission,
)
from sol_execbench.core.bench.gpu_lock_paths import (
    GPULockPaths,
    configured_gpu_locks,
    open_gpu_lock,
    prepare_gpu_locks,
)

_EVALUATION_LOCK: ContextVar[tuple[int, int, str, Path, str] | None] = (
    ContextVar("evaluation_gpu_lock", default=None)
)


class GPULockVerificationError(RuntimeError):
    """An external owner, inherited FD or active execution is unverified."""


def gpu_lock_directory() -> Path:
    """Return the explicitly configured UUID lock parent directory."""
    return configured_gpu_locks().directory


def gpu_lock_path() -> Path:
    """Return the configured physical L1 path without touching devices."""
    return configured_gpu_locks().ownership


@contextlib.contextmanager
def _locked_file(path: Path, timeout_seconds: float) -> Iterator[None]:
    if timeout_seconds <= 0:
        raise ValueError("GPU lock timeout must be positive")
    with open_gpu_lock(path, create=False) as descriptor:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"GPU is busy; lock wait exceeded {timeout_seconds:g} seconds: {path}"
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)


def _external_gpu_lock_is_held(paths: GPULockPaths) -> bool:
    with open_gpu_lock(paths.ownership, create=False) as descriptor:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False


def _reuse_inherited_lock(paths: GPULockPaths) -> bool:
    descriptor = paths.inherited_fd
    if descriptor is None:
        return False
    try:
        inherited = os.fstat(descriptor)
        expected = paths.ownership.stat(follow_symlinks=False)
        if (inherited.st_dev, inherited.st_ino) != (
            expected.st_dev,
            expected.st_ino,
        ):
            raise GPULockVerificationError(
                "GPU_LOCK_FD is not the configured UUID L1 inode"
            )
        if not _external_gpu_lock_is_held(paths):
            raise GPULockVerificationError("GPU_LOCK_FD does not hold L1")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, ValueError) as error:
        raise GPULockVerificationError(
            "GPU_LOCK_FD is not a live inherited L1 lock"
        ) from error
    return True


def verify_host_gpu_owner(paths: GPULockPaths) -> None:
    """Verify host L1, UUID and the live project-native owner receipt."""
    if not _external_gpu_lock_is_held(paths):
        raise GPULockVerificationError("host-managed GPU L1 lock is not held")
    token = os.environ.get(ENV_CLOCK_OWNER)
    if token is None:
        raise GPULockVerificationError(
            "host-managed GPU requires native owner evidence"
        )
    record = ClockOwnerRecord.model_validate_json(
        clock_owner_path(paths.directory, token).read_text()
    )
    if len(record.devices) != 1 or record.devices[0].uuid.lower() != paths.uuid:
        raise GPULockVerificationError("host-managed GPU owner UUID mismatch")
    verify_clock_owner(paths.directory, token, record.devices)


@contextlib.contextmanager
def acquire_gpu_lock(
    *,
    timeout_seconds: float = 60.0,
    admission_device: AdmissionDevice | None = None,
) -> Iterator[None]:
    """Hold L1 through native owner publication, cleanup and restoration."""
    paths = configured_gpu_locks()
    prepare_gpu_locks(paths)
    admission = configured_admission()
    if admission_device is not None and admission_device.smi_uuid != paths.uuid:
        raise AdmissionError("gpu_admission_device_mismatch")
    if paths.host_managed:
        raise GPULockVerificationError(
            "host-managed worker cannot acquire ownership"
        )
    if admission is not None and (
        admission_device is None or paths.inherited_fd is not None
    ):
        raise AdmissionError("gpu_admission_requires_native_owned_entry")
    if _reuse_inherited_lock(paths):
        verify_host_gpu_owner(paths)
        yield
        return
    with _locked_file(paths.ownership, timeout_seconds):
        if admission is not None and admission_device is not None:
            admission.require_ready(admission_device)
        require_no_clock_recovery(paths.directory, gpu_uuid=paths.uuid)
        yield


@contextlib.contextmanager
def acquire_gpu_recovery_lock(
    *, timeout_seconds: float = 60.0
) -> Iterator[None]:
    """Hold the same L1 for original-owner recovery; skip ready-only admission."""
    paths = configured_gpu_locks()
    if paths.host_managed or paths.inherited_fd is not None:
        raise GPULockVerificationError("recovery requires owned physical L1")
    prepare_gpu_locks(paths)
    with _locked_file(paths.ownership, timeout_seconds):
        yield


def _identity(paths: GPULockPaths) -> tuple[int, int, str, Path, str]:
    return (
        os.getpid(),
        threading.get_ident(),
        paths.uuid,
        paths.execution.resolve(),
        "L2",
    )


@contextlib.contextmanager
def acquire_evaluation_gpu_lock(
    *,
    timeout_seconds: float = 60.0,
    admission_device: AdmissionDevice | None = None,
) -> Iterator[None]:
    """Own L1 then L2, or verify host L1 before competing for L2."""
    paths = configured_gpu_locks()
    identity = _identity(paths)
    if _EVALUATION_LOCK.get() == identity:
        yield
        return
    prepare_gpu_locks(paths)
    if paths.host_managed:
        verify_host_gpu_owner(paths)
    ownership = (
        contextlib.nullcontext()
        if paths.host_managed
        else acquire_gpu_lock(
            timeout_seconds=timeout_seconds, admission_device=admission_device
        )
    )
    with ownership, _locked_file(paths.execution, timeout_seconds):
        if paths.host_managed:
            verify_host_gpu_owner(paths)
        token = _EVALUATION_LOCK.set(identity)
        try:
            yield
        finally:
            _EVALUATION_LOCK.reset(token)


def require_evaluation_gpu_lock() -> None:
    """Require this process/thread's active UUID L2 boundary for internal work."""
    if _EVALUATION_LOCK.get() != _identity(configured_gpu_locks()):
        raise GPULockVerificationError(
            "GPU execution requires an active native evaluation boundary"
        )


def require_supervised_gpu_execution() -> None:
    """Validate a native scope or its supervised, already serialized child.

    Evaluation subprocesses do not inherit ContextVars. Their original native
    owner receipt and both held physical locks remain mandatory; this check
    never acquires, changes or releases the supervisor's locks.
    """
    paths = configured_gpu_locks()
    verify_host_gpu_owner(paths)
    with open_gpu_lock(paths.execution, create=False) as descriptor:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    raise GPULockVerificationError("supervised GPU execution requires held L2")


__all__ = [
    "GPULockVerificationError",
    "acquire_evaluation_gpu_lock",
    "acquire_gpu_lock",
    "acquire_gpu_recovery_lock",
    "gpu_lock_directory",
    "gpu_lock_path",
    "require_evaluation_gpu_lock",
    "require_supervised_gpu_execution",
    "verify_host_gpu_owner",
]
