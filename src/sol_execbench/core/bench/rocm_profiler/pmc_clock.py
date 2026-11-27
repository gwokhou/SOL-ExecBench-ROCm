# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Target-machine PMC policy, using the existing owned clock lease teardown.

The evaluation boundary already owns the GPU lock. This window surrounds only
counter replay, never canonical timing. No privilege escalation of Python is
needed: only the two exact amd-smi set commands use sudo -n.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field

from sol_execbench.core.bench.clock_lock import (
    ClockLockLease,
    ClockRestoreError,
)
from sol_execbench.core.bench.clock_ownership import (
    ENV_CLOCK_OWNER,
    ClockDevice,
    ClockOwnership,
    ClockOwnershipError,
    owned_clock_window,
    verify_clock_owner,
)
from sol_execbench.core.bench.gpu_lock import gpu_lock_directory
from sol_execbench.core.bench.gpu_lock_paths import (
    GPU_LOCK_HOST_ENV,
    GPU_LOCK_UUID_ENV,
    normalize_gpu_uuid,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_models import (
    ROCR_UUID_PATTERN,
)
from sol_execbench.core.bench.rocm_profiler.schema_versions import (
    ProfilerArtifactSchema,
)
from sol_execbench.core.data.base_model import CurrentFrozenSchemaModel
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.platform.amd_smi import (
    AMDSMIListPayload,
    parse_performance_levels,
)
from sol_execbench.core.platform.clock_commands import (
    ClockLevel,
    clock_arguments,
    clock_executable,
    host_clock_executable,
)
from sol_execbench.core.platform.kfd_topology import (
    DEFAULT_KFD_TOPOLOGY_NODES_ROOT,
    DEFAULT_PCI_DEVICES_ROOT,
    effective_kfd_nodes_root,
    read_kfd_gpu_nodes,
)
from sol_execbench.core.process.environment import (
    ENV_ROCR_VISIBLE_DEVICES,
    ENV_SOL_EXECBENCH_CLOCKS_LOCKED,
)
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from sol_execbench.core.process.termination import cleanup_on_termination
from sol_execbench.core.timestamps import utc_timestamp

PCI_DEVICES_ROOT = DEFAULT_PCI_DEVICES_ROOT
KFD_TOPOLOGY_NODES_ROOT = DEFAULT_KFD_TOPOLOGY_NODES_ROOT
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


def required_visible_rocr_uuid() -> str:
    """Return the single physical ROCR_VISIBLE_DEVICES UUID or fail."""
    visible = os.environ.get(ENV_ROCR_VISIBLE_DEVICES, "").strip()
    if re.fullmatch(ROCR_UUID_PATTERN, visible) is None:
        raise RuntimeError(
            "pmc_clock_device_uuid_required: "
            "set single-GPU ROCR_VISIBLE_DEVICES"
        )
    return visible


def _resolve_visible_rocr_bdf(visible: str) -> str | None:
    """Resolve one ROCr GPU-<unique_id> selector to its unique target."""
    if re.fullmatch(ROCR_UUID_PATTERN, visible) is None:
        return None
    want = visible.removeprefix("GPU-").lower()
    matches: list[str] = []
    for uid_path in sorted(PCI_DEVICES_ROOT.glob("*/unique_id")):
        with suppress(OSError):
            if uid_path.read_text(encoding="utf-8").strip().lower() == want:
                matches.append(uid_path.parent.name.lower())
    if len(matches) == 1:
        return matches[0]
    kfd_nodes = read_kfd_gpu_nodes(
        effective_kfd_nodes_root(PCI_DEVICES_ROOT, KFD_TOPOLOGY_NODES_ROOT)
    )
    kfd_matches = [bdf for bdf, uid in kfd_nodes if uid == want]
    if len(kfd_matches) != 1:
        return None
    matched_bdf = kfd_matches[0]
    same_bdf_count = sum(bdf == matched_bdf for bdf, _ in kfd_nodes)
    lock_uuid = os.environ.get(GPU_LOCK_UUID_ENV, "").strip()
    if same_bdf_count > 1 and lock_uuid:
        return normalize_gpu_uuid(lock_uuid)
    return matched_bdf


def _default_pmc_clock_target() -> str:
    """Resolve an omitted target from env or the single visible ROCr GPU."""
    explicit = os.environ.get(ENV_PMC_CLOCK_TARGET)
    if explicit is not None:
        return explicit
    visible = os.environ.get(ENV_ROCR_VISIBLE_DEVICES, "").strip()
    return _resolve_visible_rocr_bdf(visible) or ""


@dataclass(frozen=True, slots=True, kw_only=True)
class PMCClockOptions:
    """Explicit GPU selector (BDF/UUID/index or all) and setup failure policy."""

    target: str = field(default_factory=_default_pmc_clock_target)
    mode: PMCClockMode = PMCClockMode.REQUIRED

    def __post_init__(self) -> None:
        """Normalize the policy and reject empty device selectors."""
        normalized_target = self.target.strip().lower()
        object.__setattr__(self, "target", normalized_target)
        object.__setattr__(self, "mode", PMCClockMode(self.mode))
        if not normalized_target or normalized_target.startswith("-"):
            raise ValueError(
                "pmc_clock_target_invalid: use a GPU BDF, UUID, index or all"
            )


def configured_pmc_clock_options() -> PMCClockOptions | None:
    """Auto-enable when ROCR_VISIBLE_DEVICES maps to one local GPU."""
    target = os.environ.get(ENV_PMC_CLOCK_TARGET)
    if target is None:
        visible = os.environ.get(ENV_ROCR_VISIBLE_DEVICES, "").strip()
        target = _resolve_visible_rocr_bdf(visible)
        if target is None:
            return None
    return PMCClockOptions(
        target=target,
        mode=PMCClockMode(
            os.environ.get(ENV_PMC_CLOCK_MODE, PMCClockMode.REQUIRED)
        ),
    )


class PMCClockReceipt(CurrentFrozenSchemaModel):
    """Independent control evidence; not benchmark score authority."""

    current_schema_version = ProfilerArtifactSchema.PMC_CLOCK_WINDOW
    schema_version: Literal[ProfilerArtifactSchema.PMC_CLOCK_WINDOW] = (
        ProfilerArtifactSchema.PMC_CLOCK_WINDOW
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
        command = (
            ["sudo", "-n", host_clock_executable(), *arguments]
            if mutate
            else [self.executable, *arguments]
        )
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

    def observe_levels(self) -> tuple[str, ...]:
        """Read normalized performance levels for the selected target."""
        raw = self.run(
            "metric", "--gpu", self.options.target, "--perf-level", "--json"
        )
        return parse_performance_levels(raw)

    def verify(self, level: ClockLevel) -> None:
        """Require every selected device reporting the requested policy."""
        if (
            self.observe_levels()
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
        lock_uuid = os.environ.get(GPU_LOCK_UUID_ENV, "").strip()
        if self.options.target == ALL_GPU_TARGET and lock_uuid:
            raise RuntimeError(
                "pmc_clock_identity_mismatch: "
                "all-GPU target disallowed when GPU_LOCK_UUID is set"
            )
        identities = AMDSMIListPayload.model_validate_json(
            self.run("list", "--json"),
        ).root
        selected = [
            item
            for item in identities
            if self.options.target
            in (
                (item.bdf or "").strip().lower(),
                (item.uuid or "").strip().lower(),
                str(item.gpu).strip().lower(),
                ALL_GPU_TARGET,
            )
        ]
        self.device_count = len(selected)
        if not selected or (
            self.options.target != ALL_GPU_TARGET and len(selected) != 1
        ):
            raise RuntimeError("pmc_clock_identity_mismatch: refusing mutation")
        if any(not item.bdf or not item.uuid for item in selected):
            raise ClockOwnershipError("pmc_clock_identity_incomplete")
        if (
            self.options.target != ALL_GPU_TARGET
            and lock_uuid
            and (selected[0].uuid or "").strip().lower()
            != normalize_gpu_uuid(lock_uuid)
        ):
            raise RuntimeError(
                "pmc_clock_identity_mismatch: "
                "target UUID disagrees with GPU_LOCK_UUID"
            )
        self.devices = tuple(
            ClockDevice(
                bdf=str(item.bdf).strip().lower(),
                uuid=str(item.uuid).strip().lower(),
            )
            for item in selected
        )

    def restore(self) -> bool:
        """Attempt AUTO and readback independently, including failed writes."""
        return self._restore_auto()

    def activate(self) -> None:
        """Bind the published owner before setting and verifying STABLE_PEAK."""
        self.receipt = self.receipt.model_copy(
            update={"owner_lease_id": os.environ[ENV_CLOCK_OWNER]}
        )
        self.save()
        self.set_level(ClockLevel.PEAK)
        self.verify(ClockLevel.PEAK)
        self.receipt = self.receipt.model_copy(update={"peak_verified": True})
        self.save()

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
def _elevate_natural_owner_window(
    control: PMCClockControl, owner: str
) -> Iterator[ClockLockLease]:
    """Temporarily lock STABLE_PEAK under an active natural AUTO owner."""
    control.receipt = control.receipt.model_copy(
        update={"ownership": ClockOwnership.OWNED, "owner_lease_id": owner}
    )
    control.save()
    activation_started = False

    def rollback() -> bool:
        return not activation_started or control.restore()

    lease = ClockLockLease(locked=False, acquired=True, restore=rollback)
    try:
        with lease:
            activation_started = True
            control.set_level(ClockLevel.PEAK)
            control.verify(ClockLevel.PEAK)
            lease.locked = True
            control.receipt = control.receipt.model_copy(
                update={"peak_verified": True}
            )
            control.save()
            lease.publish_environment()
            try:
                yield lease
            finally:
                lease.restore_environment()
                verify_clock_owner(gpu_lock_directory(), owner, control.devices)
    except BaseException as error:
        if not lease.released and lease.release_blocker is None:
            raise ClockRestoreError(
                "clock_restore_unverified: failed to reset and verify owned "
                "GPU policy at AUTO; inspect the clock owner record"
            ) from error
        raise


@contextmanager
def _lease_window(control: PMCClockControl) -> Iterator[ClockLockLease]:
    """Borrow a live verified owner or publish our own restoration obligation."""
    control.preflight()
    owner = os.environ.get(ENV_CLOCK_OWNER)
    if owner is not None:
        verify_clock_owner(gpu_lock_directory(), owner, control.devices)
        levels = control.observe_levels()
        expected_peak = (
            f"AMDSMI_DEV_PERF_LEVEL_{ClockLevel.PEAK}",
        ) * control.device_count
        expected_auto = (
            f"AMDSMI_DEV_PERF_LEVEL_{ClockLevel.AUTO}",
        ) * control.device_count
        if levels == expected_peak:
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
        if (
            levels == expected_auto
            and os.environ.get(ENV_SOL_EXECBENCH_CLOCKS_LOCKED) != "1"
            and os.environ.get(GPU_LOCK_HOST_ENV) != "1"
        ):
            with _elevate_natural_owner_window(control, owner) as lease:
                yield lease
            return
        raise RuntimeError(
            f"pmc_clock_readback_failed: expected {ClockLevel.PEAK}"
        )
    control.verify(ClockLevel.AUTO)
    control.receipt = control.receipt.model_copy(
        update={"ownership": ClockOwnership.OWNED}
    )
    with owned_clock_window(
        gpu_lock_directory(),
        control.devices,
        activate=control.activate,
        restore=control.restore,
    ) as lease:
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
                except (ClockOwnershipError, ClockRestoreError):
                    raise
                except Exception:
                    owned = control.receipt.ownership == ClockOwnership.OWNED
                    if (
                        options.mode == PMCClockMode.REQUIRED
                        or not owned
                        or not control.receipt.auto_verified
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
