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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from sol_execbench.core.bench.clock_lock import ClockLockLease
from sol_execbench.core.data.base_model import CurrentFrozenSchemaModel
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity.schema_versions import SchemaVersion
from sol_execbench.core.platform.amd_smi import AMDSMIListPayload
from sol_execbench.core.platform.runtime import resolve_rocm_tool_command
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from sol_execbench.core.timestamps import utc_timestamp

ENV_CLOCK_OWNER = "SOL_EXECBENCH_CLOCK_OWNER"
LEASE_DIRECTORY = "clock-leases"
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

    current_schema_version = SchemaVersion.CLOCK_OWNER
    schema_version: Literal[SchemaVersion.CLOCK_OWNER] = (
        SchemaVersion.CLOCK_OWNER
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
    return directory / LEASE_DIRECTORY / f"{token}.json"


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


def require_no_clock_recovery(directory: Path) -> None:
    """Fail closed after a crashed owner or failed restore, once GPU lock is held."""
    for path in sorted((directory / LEASE_DIRECTORY).glob("*.json")):
        try:
            record = ClockOwnerRecord.model_validate_json(path.read_text())
        except (OSError, ValueError) as error:
            raise ClockOwnershipError(
                f"clock_recovery_record_invalid: {path}"
            ) from error
        if record.state != ClockOwnerState.RESTORED:
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
    try:
        record = ClockOwnerRecord(
            lease_id=uuid4().hex,
            devices=devices or observe_clock_devices(),
            owner_pid=os.getpid(),
            container_supervised=container_supervised,
        )
        path = clock_owner_path(directory, record.lease_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            path.with_suffix(".lock"), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_write_json_value(path, record.model_dump(mode="json"))
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
            record = record.model_copy(
                update={
                    "state": ClockOwnerState.RESTORED
                    if lease.released
                    else ClockOwnerState.RESTORE_FAILED,
                    "ended_at": utc_timestamp(),
                    "release_blocker": lease.release_blocker,
                }
            )
            atomic_write_json_value(path, record.model_dump(mode="json"))
        finally:
            os.close(descriptor)
