# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Target-machine PMC policy, using the existing owned clock lease teardown.

The evaluation boundary already owns the GPU lock. This window surrounds only
counter replay, never canonical timing. No privilege escalation of Python is
needed: only the two exact amd-smi set commands use sudo -n.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field

from sol_execbench.core.bench.clock_lock import ClockLockLease
from sol_execbench.core.bench.clock_ownership import (
    ENV_CLOCK_OWNER,
    ClockDevice,
    ClockOwnership,
    ClockOwnershipError,
    published_clock_owner,
    verify_clock_owner,
)
from sol_execbench.core.bench.gpu_lock import gpu_lock_directory
from sol_execbench.core.data.base_model import CurrentFrozenSchemaModel
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity.schema_versions import SchemaVersion
from sol_execbench.core.platform.amd_smi import (
    AMDSMIListPayload,
    parse_performance_levels,
)
from sol_execbench.core.platform.clock_commands import (
    ClockLevel,
    clock_arguments,
    clock_executable,
)
from sol_execbench.core.process.environment import ENV_ROCR_VISIBLE_DEVICES
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from sol_execbench.core.process.termination import cleanup_on_termination
from sol_execbench.core.timestamps import utc_timestamp

TARGET_ROCR_UUID = "GPU-a30fa29c1cca1511"
TARGET_BDF = "0000:05:00.0"
TARGET_SMI_UUID = "a3007590-0000-1000-800f-a29c1cca1511"
CONTROL_TIMEOUT_SECONDS = 20
CONTROL_OUTPUT_BYTES = 8192
ARCHIVE_SUFFIX = ".pmc-clock"
ENV_PMC_CLOCK_TARGET = "SOL_EXECBENCH_PMC_CLOCK_TARGET"
ENV_PMC_CLOCK_MODE = "SOL_EXECBENCH_PMC_CLOCK_MODE"
ALL_GPU_TARGET = "all"


class PMCClockMode(StrEnum):
    """Whether setup failure may continue as diagnostic-only collection."""

    REQUIRED = "required"
    BEST_EFFORT = "best-effort"


@dataclass(frozen=True)
class PMCClockOptions:
    """Explicit GPU selector (BDF/UUID/index or all) and setup failure policy."""

    target: str = TARGET_BDF
    mode: PMCClockMode = PMCClockMode.REQUIRED

    def __post_init__(self) -> None:
        """Normalize the policy and reject empty device selectors."""
        object.__setattr__(self, "mode", PMCClockMode(self.mode))
        if not self.target.strip() or self.target.startswith("-"):
            raise ValueError(
                "pmc_clock_target_invalid: use a GPU BDF, UUID, index or all"
            )


def configured_pmc_clock_options() -> PMCClockOptions | None:
    """Auto-enable only the qualified target; explicit options work elsewhere."""
    target = os.environ.get(ENV_PMC_CLOCK_TARGET)
    if target is None:
        if os.environ.get(ENV_ROCR_VISIBLE_DEVICES) != TARGET_ROCR_UUID:
            return None
        target = TARGET_BDF
    return PMCClockOptions(
        target=target,
        mode=PMCClockMode(
            os.environ.get(ENV_PMC_CLOCK_MODE, PMCClockMode.REQUIRED)
        ),
    )


class PMCClockReceipt(CurrentFrozenSchemaModel):
    """Independent control evidence; not benchmark score authority."""

    current_schema_version = SchemaVersion.PMC_CLOCK_WINDOW
    schema_version: Literal[SchemaVersion.PMC_CLOCK_WINDOW] = (
        SchemaVersion.PMC_CLOCK_WINDOW
    )
    diagnostic_only: Literal[True] = True
    score_authority: Literal[False] = False
    target: str
    mode: PMCClockMode
    started_at: str = Field(default_factory=utc_timestamp)
    ended_at: str | None = None
    peak_verified: bool = False
    auto_verified: bool = False
    collection_completed: bool = False
    ownership: ClockOwnership = ClockOwnership.NONE
    owner_lease_id: str | None = None
    borrow_ended: bool = False
    observations: list[dict[str, str | int]] = Field(default_factory=list)


class PMCClockControl:
    """Bounded target-only commands with incrementally persisted evidence."""

    def __init__(self, archive: Path, options: PMCClockOptions) -> None:
        """Bind one collection to its selector, policy and independent archive."""
        self.options = options
        self.device_count = 0
        self.devices: tuple[ClockDevice, ...] = ()
        self.path = archive / f"{uuid4().hex}.json"
        self.receipt = PMCClockReceipt(target=options.target, mode=options.mode)
        self.executable = clock_executable()
        self.save()

    def save(self) -> None:
        """Persist before changing policy and after each observation."""
        atomic_write_json_value(self.path, self.receipt.model_dump(mode="json"))

    def run(self, *arguments: str, mutate: bool = False) -> str:
        """Run one command; retain bounded tool output without environment."""
        command = [self.executable, *arguments]
        if mutate:
            command = ["sudo", "-n", *command]
        self.receipt.observations.append(
            {"started_arguments": " ".join(arguments)}
        )
        self.save()
        result = run_in_process_group_bounded(
            command,
            timeout=CONTROL_TIMEOUT_SECONDS,
            max_capture_bytes=CONTROL_OUTPUT_BYTES,
        )
        self.receipt.observations.append(
            {
                "arguments": " ".join(arguments),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        self.save()
        if result.returncode:
            raise RuntimeError(
                f"pmc_clock_command_failed: see {self.path}; authenticate the "
                "targeted amd-smi commands on the host before retrying",
            )
        return result.stdout

    def verify(self, level: ClockLevel) -> None:
        """Require exactly one selected device reporting the requested policy."""
        raw = self.run(
            "metric", "--gpu", self.options.target, "--perf-level", "--json"
        )
        if (
            parse_performance_levels(raw)
            != (f"AMDSMI_DEV_PERF_LEVEL_{level}",) * self.device_count
        ):
            raise RuntimeError(f"pmc_clock_readback_failed: expected {level}")

    def set_level(self, level: ClockLevel) -> None:
        """Use verified UUIDs and the exact sudoers argument order for both levels."""
        if not self.devices:
            raise ClockOwnershipError("pmc_clock_identity_unverified")
        failures: list[Exception] = []
        for device in self.devices:
            try:
                self.run(*clock_arguments(level, device.uuid), mutate=True)
            except Exception as error:
                if level != ClockLevel.AUTO:
                    raise
                failures.append(error)
        if failures:
            raise failures[0]

    def preflight(self) -> None:
        """Refuse wrong identity or an externally owned performance policy."""
        identities = AMDSMIListPayload.model_validate_json(
            self.run("list", "--json"),
        ).root
        selected = [
            item
            for item in identities
            if self.options.target
            in (
                item.bdf,
                item.uuid,
                str(item.gpu),
                ALL_GPU_TARGET,
            )
        ]
        self.device_count = len(selected)
        if not selected or (
            self.options.target != ALL_GPU_TARGET and len(selected) != 1
        ):
            raise RuntimeError("pmc_clock_identity_mismatch: refusing mutation")
        if (
            self.options.target == TARGET_BDF
            and selected[0].uuid != TARGET_SMI_UUID
        ):
            raise RuntimeError(
                "pmc_clock_identity_mismatch: target UUID changed"
            )
        if any(not item.bdf or not item.uuid for item in selected):
            raise ClockOwnershipError("pmc_clock_identity_incomplete")
        self.devices = tuple(
            ClockDevice(bdf=str(item.bdf), uuid=str(item.uuid))
            for item in selected
        )

    def restore(self) -> bool:
        """Attempt AUTO and readback independently, including failed writes."""
        return self._restore_auto()

    def _restore_auto(self) -> bool:
        """Require readback even when a set command reports an error."""
        with suppress(Exception):
            self.set_level(ClockLevel.AUTO)
        try:
            self.verify(ClockLevel.AUTO)
        except Exception:  # noqa: BLE001 -- ClockLockLease propagates failure
            return False
        self.receipt = self.receipt.model_copy(update={"auto_verified": True})
        self.save()
        return True


@contextmanager
def _lease_window(control: PMCClockControl) -> Iterator[ClockLockLease]:
    """Borrow a live verified owner or publish our own restoration obligation."""
    control.preflight()
    owner = os.environ.get(ENV_CLOCK_OWNER)
    if owner is not None:
        verify_clock_owner(gpu_lock_directory(), owner, control.devices)
        control.verify(ClockLevel.PEAK)
        control.receipt = control.receipt.model_copy(
            update={
                "ownership": ClockOwnership.BORROWED,
                "owner_lease_id": owner,
                "peak_verified": True,
            }
        )
        control.save()
        try:
            with ClockLockLease(locked=True, acquired=False) as lease:
                yield lease
        finally:
            control.receipt = control.receipt.model_copy(
                update={"borrow_ended": True}
            )
            verify_clock_owner(gpu_lock_directory(), owner, control.devices)
            control.verify(ClockLevel.PEAK)
        return
    control.verify(ClockLevel.AUTO)
    lease = ClockLockLease(locked=False, acquired=True, restore=control.restore)
    control.receipt = control.receipt.model_copy(
        update={"ownership": ClockOwnership.OWNED}
    )
    with published_clock_owner(
        lease, gpu_lock_directory(), control.devices
    ) as owner:
        control.receipt = control.receipt.model_copy(
            update={
                "ownership": ClockOwnership.OWNED,
                "owner_lease_id": owner,
            }
        )
        control.save()
        control.set_level(ClockLevel.PEAK)
        control.verify(ClockLevel.PEAK)
        lease.locked = True
        control.receipt = control.receipt.model_copy(
            update={"peak_verified": True}
        )
        control.save()
        yield lease


@contextmanager
def pmc_clock_window(
    output_directory: Path,
    options: PMCClockOptions | None = None,
) -> Iterator[ClockLockLease]:
    """Reuse owned or borrowed lease teardown while the GPU lock is held.

    Borrowers never set AUTO. Required mode refuses setup failure; best-effort
    can proceed only when no ownership proof is disputed and rollback succeeded.
    """
    options = options or configured_pmc_clock_options()
    if options is None:
        yield ClockLockLease(locked=False, acquired=False)
        return
    archive = output_directory.with_name(output_directory.name + ARCHIVE_SUFFIX)
    control = PMCClockControl(archive, options)
    with cleanup_on_termination():
        try:
            with ExitStack() as leases:
                try:
                    lease = leases.enter_context(_lease_window(control))
                except ClockOwnershipError:
                    raise
                except Exception:
                    owned = control.receipt.ownership == ClockOwnership.OWNED
                    if (
                        options.mode == PMCClockMode.REQUIRED
                        or os.environ.get(ENV_CLOCK_OWNER) is not None
                        or (owned and not control.receipt.auto_verified)
                    ):
                        raise
                    lease = ClockLockLease(locked=False, acquired=False)
                yield lease
                if lease.locked:
                    control.verify(ClockLevel.PEAK)
                control.receipt = control.receipt.model_copy(
                    update={"collection_completed": True}
                )
        finally:
            control.receipt = control.receipt.model_copy(
                update={"ended_at": utc_timestamp()}
            )
            control.save()
