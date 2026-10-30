"""Public native boundary verified with explicit CPU hardware observations."""

from __future__ import annotations

import json

import pytest

from sol_execbench.core.bench import gpu_execution
from sol_execbench.core.bench.clock_ownership import (
    ClockDevice,
    ClockOwnershipError,
)
from sol_execbench.core.bench.gpu_lock import acquire_gpu_lock

UUID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def hardware(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setenv("GPU_LOCK_UUID", UUID)
    monkeypatch.setenv(
        "SOL_EXECBENCH_CLOCK_STATE_DIR", str(tmp_path / "private")
    )
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-0123456789abcdef")
    for name in [
        "SOL_EXECBENCH_GPU_LOCK_DIR",
        "SOL_EXECBENCH_ADMISSION_RECORD",
        "GPU_LOCK_HELD_BY_HOST",
        "GPU_LOCK_FD",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "sol_execbench.core.bench.clock_ownership.LEGACY_CLOCK_STATE_DIRECTORY",
        tmp_path / "legacy",
    )
    pci = tmp_path / "pci" / "0000:05:00.0"
    pci.mkdir(parents=True)
    (pci / "unique_id").write_text("0123456789abcdef")
    monkeypatch.setattr(gpu_execution, "PCI_DEVICES_ROOT", pci.parent)
    monkeypatch.setattr(
        gpu_execution,
        "observe_clock_devices",
        lambda: (ClockDevice(uuid=UUID, bdf="0000:05:00.0"),),
    )
    monkeypatch.setattr(
        gpu_execution,
        "observe_clock_levels",
        lambda _: ("AMDSMI_DEV_PERF_LEVEL_AUTO",),
    )
    monkeypatch.setattr(
        gpu_execution, "synchronize_initialized_device", lambda: None
    )
    return tmp_path


def test_cpu_entry_does_not_resolve_config_or_device(monkeypatch):
    monkeypatch.delenv("GPU_LOCK_DIR", raising=False)
    monkeypatch.delenv("GPU_LOCK_UUID", raising=False)
    monkeypatch.setattr(
        gpu_execution,
        "resolve_gpu_device",
        lambda: pytest.fail("CPU queried device"),
    )
    with gpu_execution.native_gpu_execution("cpu"):
        pass


def test_natural_execution_publishes_and_restores_without_setter(
    hardware, monkeypatch
):
    monkeypatch.setattr(
        "sol_execbench.core.bench.host_clock_guard.set_peak_clocks",
        lambda *_: pytest.fail("natural entry set clocks"),
    )
    with gpu_execution.native_gpu_execution():
        records = list((hardware / "private/clock-leases").glob("*.json"))
        assert len(records) == 1
        assert json.loads(records[0].read_text())["state"] == "active"
        with (
            pytest.raises(TimeoutError),
            acquire_gpu_lock(timeout_seconds=0.01),
        ):
            pass
        with gpu_execution.native_gpu_execution():
            pass
    assert json.loads(records[0].read_text())["state"] == "restored"


def test_synchronize_failure_keeps_native_blocker(hardware, monkeypatch):
    def failure():
        raise RuntimeError("sync failed")

    monkeypatch.setattr(
        gpu_execution, "synchronize_initialized_device", failure
    )
    with (
        pytest.raises(RuntimeError, match="sync failed"),
        gpu_execution.native_gpu_execution(),
    ):
        pass
    records = list((hardware / "private/clock-leases").glob("*.json"))
    record = json.loads(records[0].read_text())
    assert record["state"] == "restore_failed"
    assert record["release_blocker"] == gpu_execution.EXECUTION_CLEANUP_BLOCKER
    monkeypatch.setattr(
        gpu_execution, "synchronize_initialized_device", lambda: None
    )
    with (
        pytest.raises(ClockOwnershipError, match="clock_recovery_required"),
        gpu_execution.native_gpu_execution(),
    ):
        pytest.fail("pending cleanup admitted new work")


@pytest.mark.parametrize(
    "variable,value",
    [
        ("GPU_LOCK_UUID", "11111111-2222-3333-4444-666666666666"),
        (
            "SOL_EXECBENCH_HOST_CLOCK_GPU_UUID",
            "11111111-2222-3333-4444-666666666666",
        ),
        ("ROCR_VISIBLE_DEVICES", "0"),
        ("HIP_VISIBLE_DEVICES", "1"),
    ],
)
def test_target_mismatch_prevents_gpu_work(
    hardware, monkeypatch, variable, value
):
    monkeypatch.setenv(variable, value)
    with (
        pytest.raises(ClockOwnershipError),
        gpu_execution.native_gpu_execution(),
    ):
        pytest.fail("mismatched device admitted")
    assert not (hardware / "private").exists()


def test_raw_lock_does_not_replace_native_owner(hardware):
    from sol_execbench.core.bench.gpu_lock import (
        GPULockVerificationError,
        acquire_evaluation_gpu_lock,
    )

    with (
        acquire_evaluation_gpu_lock(),
        pytest.raises(GPULockVerificationError, match="native owner"),
        gpu_execution.native_gpu_execution(),
    ):
        pytest.fail("unowned raw lock became a public execution context")


def test_nested_performance_request_cannot_silently_borrow_auto(
    hardware, monkeypatch
):
    monkeypatch.setattr(gpu_execution, "verify_clocks", lambda _: False)
    with (
        gpu_execution.native_gpu_execution(),
        pytest.raises(
            ClockOwnershipError, match="existing owner's STABLE_PEAK"
        ),
        gpu_execution.native_gpu_execution(manage_clocks=True),
    ):
        pytest.fail("nested scope silently accepted AUTO")
