"""Native GPU entry boundary, retaining existing owner/recovery responsibilities."""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path

from sol_execbench.core.bench.clock_lock import (
    ClockLockLease,
    observe_clock_levels,
    verify_clocks,
)
from sol_execbench.core.bench.clock_ownership import (
    EXECUTION_CLEANUP_BLOCKER,
    ClockDevice,
    ClockOwnershipError,
    observe_clock_devices,
    published_clock_owner,
)
from sol_execbench.core.bench.gpu_admission import AdmissionDevice
from sol_execbench.core.bench.gpu_lock import (
    GPULockVerificationError,
    acquire_evaluation_gpu_lock,
    require_evaluation_gpu_lock,
    require_supervised_gpu_execution,
)
from sol_execbench.core.bench.gpu_lock_paths import configured_gpu_locks
from sol_execbench.core.process.termination import (
    cleanup_on_termination,
    defer_termination,
)

PCI_DEVICES_ROOT = Path("/sys/bus/pci/devices")


def resolve_gpu_device() -> ClockDevice:
    """Resolve the explicit lock UUID and verify the single visible ROCr target."""
    paths = configured_gpu_locks()
    requested = os.environ.get("SOL_EXECBENCH_HOST_CLOCK_GPU_UUID", paths.uuid)
    if requested.strip().lower() != paths.uuid:
        raise ClockOwnershipError("GPU target and GPU_LOCK_UUID disagree")
    selected = [
        device
        for device in observe_clock_devices()
        if device.uuid.lower() == paths.uuid
    ]
    if len(selected) != 1:
        raise ClockOwnershipError("GPU_LOCK_UUID has no unique native device")
    device = ClockDevice(
        bdf=selected[0].bdf.lower(), uuid=selected[0].uuid.lower()
    )
    # Read the actual ROCr identity from this BDF; never derive a lock UUID
    # from a visible ordinal or convert a truncated ROCr UUID into an SMI UUID.
    unique = (
        (PCI_DEVICES_ROOT / device.bdf / "unique_id")
        .read_text()
        .strip()
        .lower()
    )
    visible = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip().lower()
    if not unique or visible != f"gpu-{unique}":
        raise ClockOwnershipError(
            "ROCR_VISIBLE_DEVICES must select the resolved UUID's unique_id"
        )
    if os.environ.get("HIP_VISIBLE_DEVICES") not in (None, "0"):
        raise ClockOwnershipError(
            "HIP_VISIBLE_DEVICES must preserve the single resolved GPU"
        )
    return device


def synchronize_initialized_device() -> None:
    """Synchronize existing work without initializing a device on a CPU path."""
    if "torch" not in sys.modules:
        return
    import torch

    if torch.cuda.is_initialized():
        with defer_termination():
            torch.cuda.synchronize()


@contextmanager
def _natural_clock_owner(device: ClockDevice) -> Iterator[ClockLockLease]:
    """Record a natural-clock GPU owner without issuing a performance setter."""
    paths = configured_gpu_locks()
    if observe_clock_levels(device.uuid) != ("AMDSMI_DEV_PERF_LEVEL_AUTO",):
        raise ClockOwnershipError(
            "natural GPU execution requires verified initial AUTO"
        )
    lease = ClockLockLease(
        locked=False,
        acquired=True,
        restore=lambda: (
            observe_clock_levels(device.uuid) == ("AMDSMI_DEV_PERF_LEVEL_AUTO",)
        ),
    )
    with published_clock_owner(lease, paths.directory, (device,)):
        yield lease


@contextmanager
def native_gpu_execution(
    device: str = "cuda:0",
    *,
    timeout_seconds: float = 60.0,
    manage_clocks: bool = False,
) -> Iterator[None]:
    """Cover identity, L1/L2, native owner, synchronization and verified teardown."""
    if device.split(":", 1)[0] == "cpu":
        yield
        return
    if device not in ("cuda", "cuda:0", "hip", "hip:0"):
        raise ValueError(
            "GPU entry requires device 0 of the explicitly isolated UUID"
        )
    try:
        require_evaluation_gpu_lock()
    except GPULockVerificationError:
        pass
    else:
        require_supervised_gpu_execution()
        if manage_clocks and not verify_clocks(configured_gpu_locks().uuid):
            raise ClockOwnershipError(
                "nested performance work requires the existing owner's STABLE_PEAK"
            )
        yield
        return
    physical = resolve_gpu_device()
    paths = configured_gpu_locks()
    identity = AdmissionDevice(
        pci_bdf=physical.bdf.lower(), smi_uuid=physical.uuid.lower()
    )
    with (
        cleanup_on_termination()
        if threading.current_thread() is threading.main_thread()
        else nullcontext(),
        acquire_evaluation_gpu_lock(
            timeout_seconds=timeout_seconds, admission_device=identity
        ),
    ):
        if resolve_gpu_device() != physical:
            raise ClockOwnershipError(
                "GPU identity changed after ownership acquisition"
            )
        if paths.host_managed or paths.inherited_fd is not None:
            owner = nullcontext(None)
        elif manage_clocks:
            from sol_execbench.core.bench.host_clock_guard import (
                host_clock_window,
            )

            owner = host_clock_window(
                container_supervised=False, gpu_uuid=paths.uuid
            )
        else:
            owner = _natural_clock_owner(physical)
        with owner as lease:
            if (
                lease is None
                and manage_clocks
                and not verify_clocks(paths.uuid)
            ):
                raise ClockOwnershipError(
                    "borrowed performance work requires the owner's STABLE_PEAK"
                )
            if lease is not None:
                lease.publish_environment()
            try:
                yield
            finally:
                try:
                    synchronize_initialized_device()
                except BaseException:
                    if lease is not None:
                        lease.release_blocker = EXECUTION_CLEANUP_BLOCKER
                    raise
                finally:
                    if lease is not None:
                        lease.restore_environment()


def acquire_clock_lock() -> ClockLockLease:
    """Borrow verified target clocks inside an active native execution boundary.

    Performance setup/restoration belongs to the native owner entry. Direct
    calls cannot acquire a second lease or mutate every visible GPU.
    """
    require_supervised_gpu_execution()
    paths = configured_gpu_locks()
    return ClockLockLease(locked=verify_clocks(paths.uuid), acquired=False)


def lock_clocks() -> bool:
    """Compatibility wrapper returning whether GPUs are at STABLE_PEAK."""
    lease = acquire_clock_lock()
    if lease.acquired:
        lease.detach()
    return lease.locked
