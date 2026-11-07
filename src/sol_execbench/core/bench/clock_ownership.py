# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Live, cooperative clock ownership across host/container boundaries.

The shared lock directory is a trusted orchestration channel, not a security
boundary against processes allowed to modify that directory. A token alone or
a leftover file never proves that an owner is alive.
"""

from __future__ import annotations

import fcntl
import os
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from sol_execbench.core.bench.clock_lock import (
    ClockLockLease,
    ClockRestoreError,
)
from sol_execbench.core.bench.gpu_admission import (
    AdmissionDevice,
    AdmissionError,
    clock_state_directory,
    configured_admission,
    durable_json_write,
    ensure_durable_directory,
)
from sol_execbench.core.bench.gpu_lock_paths import (
    LEGACY_CLOCK_STATE_DIRECTORY,
    normalize_gpu_uuid,
)
from sol_execbench.core.control_plane_schema_versions import (
    ExecutionControlSchema,
)
from sol_execbench.core.data.base_model import CurrentFrozenSchemaModel
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.platform.amd_smi import AMDSMIListPayload
from sol_execbench.core.platform.runtime import resolve_rocm_tool_command
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from sol_execbench.core.timestamps import utc_timestamp

ENV_CLOCK_OWNER = "SOL_EXECBENCH_CLOCK_OWNER"
LEASE_DIRECTORY = "clock-leases"
EXECUTION_CLEANUP_BLOCKER = "execution_cleanup_unverified"
OWNER_TOKEN_PATTERN = re.compile(r"[0-9a-f]{32}")
IDENTITY_TIMEOUT_SECONDS = 10
IDENTITY_OUTPUT_BYTES = 16384


class ClockOwnership(StrEnum):
    """Restoration responsibility of the current scope."""

    OWNED = "owned"
    BORROWED = "borrowed"
    NONE = "none"


class ClockOwnerState(StrEnum):
    """Durable owner outcome, distinct from a borrower's exit."""

    ACTIVE = "active"
    RESTORED = "restored"
    RESTORE_FAILED = "restore_failed"


class ClockDevice(BaseModel):
    """Stable physical identity covered by an owner."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    bdf: str = Field(min_length=1)
    uuid: str = Field(min_length=1)


class ClockOwnerRecord(CurrentFrozenSchemaModel):
    """Persistent receipt plus live file lock used for cooperative borrowing."""

    current_schema_version = ExecutionControlSchema.CLOCK_OWNER
    schema_version: Literal[ExecutionControlSchema.CLOCK_OWNER] = (
        ExecutionControlSchema.CLOCK_OWNER
    )
    lease_id: str = Field(pattern=OWNER_TOKEN_PATTERN.pattern)
    devices: tuple[ClockDevice, ...] = Field(min_length=1)
    state: ClockOwnerState = ClockOwnerState.ACTIVE
    owner_pid: int = Field(gt=0)
    started_at: str = Field(default_factory=utc_timestamp)
    ended_at: str | None = None
    recovered_at: str | None = None
    release_blocker: str | None = None
    recovery_observations: tuple[str, ...] = ()
    container_supervised: bool = False
    restore_to: Literal["AUTO"] = "AUTO"


class ClockOwnershipError(RuntimeError):
    """Owner proof or restoration is missing; never downgrade to best-effort."""


def observe_clock_devices() -> tuple[ClockDevice, ...]:
    """Observe the physical identities of the legacy all-GPU host lease."""
    result = run_in_process_group_bounded(
        [resolve_rocm_tool_command("amd-smi"), "list", "--json"],
        timeout=IDENTITY_TIMEOUT_SECONDS,
        max_capture_bytes=IDENTITY_OUTPUT_BYTES,
    )
    if result.returncode:
        raise ClockOwnershipError("clock_owner_identity_unavailable")
    devices = AMDSMIListPayload.model_validate_json(result.stdout).root
    if not devices or any(
        not device.bdf or not device.uuid for device in devices
    ):
        raise ClockOwnershipError("clock_owner_identity_incomplete")
    return tuple(ClockDevice(bdf=str(d.bdf), uuid=str(d.uuid)) for d in devices)


def clock_owner_path(directory: Path, token: str) -> Path:
    """Resolve only a canonical token under the shared ownership directory."""
    if OWNER_TOKEN_PATTERN.fullmatch(token) is None:
        raise ClockOwnershipError("clock_owner_token_invalid")
    return clock_state_directory(directory) / LEASE_DIRECTORY / f"{token}.json"


def write_clock_record(path: Path, record: ClockOwnerRecord) -> None:
    """Persist native responsibility before any admitted device side effect."""
    value = record.model_dump(mode="json")
    if configured_admission() is None:
        atomic_write_json_value(path, value)
    else:
        durable_json_write(path, value)


def admission_device(record: ClockOwnerRecord) -> AdmissionDevice:
    """Require one explicit canonical device in the admitted native owner."""
    if len(record.devices) != 1:
        raise AdmissionError("gpu_admission_requires_single_device")
    device = record.devices[0]
    return AdmissionDevice(
        pci_bdf=device.bdf.lower(), smi_uuid=device.uuid.lower()
    )


def verify_clock_owner(
    directory: Path, token: str, devices: Sequence[ClockDevice]
) -> ClockOwnerRecord:
    """Require active receipt, matching devices and a currently held owner lock."""
    path = clock_owner_path(directory, token)
    try:
        record = ClockOwnerRecord.model_validate_json(path.read_text())
        descriptor = os.open(
            path.with_suffix(".lock"), os.O_RDONLY | os.O_NOFOLLOW
        )
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise ClockOwnershipError("clock_owner_not_live")
        finally:
            os.close(descriptor)
    except (OSError, ValueError) as error:
        raise ClockOwnershipError("clock_owner_record_unavailable") from error
    if record.lease_id != token or record.state != ClockOwnerState.ACTIVE:
        raise ClockOwnershipError("clock_owner_not_active")
    if not devices or not set(devices).issubset(record.devices):
        raise ClockOwnershipError("clock_owner_device_mismatch")
    return record


def require_no_clock_recovery(directory: Path, *, gpu_uuid: str) -> None:
    """Require no unresolved lease covering the held canonical GPU UUID."""
    directories = {
        clock_state_directory(directory),
        LEGACY_CLOCK_STATE_DIRECTORY,
    }
    paths = {
        path
        for root in directories
        for path in (root / LEASE_DIRECTORY).glob("*.json")
    }
    for path in sorted(paths):
        try:
            record = ClockOwnerRecord.model_validate_json(path.read_text())
            if record.state is ClockOwnerState.RESTORED:
                continue
            devices = {
                normalize_gpu_uuid(device.uuid) for device in record.devices
            }
        except (OSError, ValueError) as error:
            raise ClockOwnershipError(
                f"clock_recovery_record_invalid: {path}"
            ) from error
        if gpu_uuid in devices:
            raise ClockOwnershipError(
                f"clock_recovery_required: inspect {path}"
            )


@contextmanager
def published_clock_owner(
    lease: ClockLockLease,
    directory: Path,
    devices: tuple[ClockDevice, ...] | None = None,
    *,
    container_supervised: bool = False,
) -> Iterator[str | None]:
    """Own lease teardown, publishing proof until AUTO restoration is recorded."""
    if not lease.acquired:
        with lease:
            yield None
        return
    descriptor: int | None = None
    admission = configured_admission()
    try:
        record = ClockOwnerRecord(
            lease_id=uuid4().hex,
            devices=devices or observe_clock_devices(),
            owner_pid=os.getpid(),
            container_supervised=container_supervised and lease.locked,
        )
        path = clock_owner_path(directory, record.lease_id)
        if admission is not None:
            admission.hold(admission_device(record), record.lease_id, path)
            ensure_durable_directory(path.parent)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            path.with_suffix(".lock"), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_clock_record(path, record)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        with lease:
            raise
    previous = os.environ.get(ENV_CLOCK_OWNER)
    os.environ[ENV_CLOCK_OWNER] = record.lease_id
    try:
        with lease:
            yield record.lease_id
    finally:
        if previous is None:
            os.environ.pop(ENV_CLOCK_OWNER, None)
        else:
            os.environ[ENV_CLOCK_OWNER] = previous
        try:
            # Activation may have persisted container supervision even when
            # natural clocks require no setter. Preserve that recovery duty.
            if container_supervised:
                record = ClockOwnerRecord.model_validate_json(path.read_text())
            record = record.model_copy(
                update={
                    "state": ClockOwnerState.RESTORED
                    if lease.released
                    else ClockOwnerState.RESTORE_FAILED,
                    "ended_at": utc_timestamp(),
                    "release_blocker": lease.release_blocker,
                }
            )
            write_clock_record(path, record)
            if (
                admission is not None
                and record.state == ClockOwnerState.RESTORED
            ):
                admission.release(
                    admission_device(record), record.lease_id, path
                )
        finally:
            os.close(descriptor)


def _record_container_supervision(directory: Path, token: str) -> None:
    """Persist possible container ownership before allowing workload startup."""
    path = clock_owner_path(directory, token)
    record = ClockOwnerRecord.model_validate_json(path.read_text())
    record = record.model_copy(update={"container_supervised": True})
    write_clock_record(path, record)


@contextmanager
def owned_clock_window(
    directory: Path,
    devices: tuple[ClockDevice, ...],
    *,
    activate: Callable[[], None] | None,
    restore: Callable[[], bool],
    container_supervised: bool = False,
) -> Iterator[ClockLockLease]:
    """Publish responsibility before activation, then require verified rollback.

    Callers hold GPU exclusion and verify identity/initial AUTO first. Hardware
    callbacks are the test seam shared by host setup and target-scoped PMC setup.
    A failure to publish ownership must not invoke either mutating callback.
    """
    activation_started = False

    def rollback() -> bool:
        return not activation_started or restore()

    lease = ClockLockLease(locked=False, acquired=True, restore=rollback)
    try:
        with published_clock_owner(
            lease, directory, devices, container_supervised=container_supervised
        ) as token:
            activation_started = True
            if activate is not None:
                activate()
                lease.locked = True
            if container_supervised:
                if token is None:
                    raise ClockOwnershipError("clock_owner_missing")
                _record_container_supervision(directory, token)
            yield lease
    except BaseException as error:
        if not lease.released and lease.release_blocker is None:
            raise ClockRestoreError(
                "clock_restore_unverified: failed to reset and verify owned "
                "GPU policy at AUTO; inspect the clock owner record"
            ) from error
        raise
