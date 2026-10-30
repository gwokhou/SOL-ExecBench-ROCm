# SPDX-License-Identifier: Apache-2.0
"""Acquisition faults must not lose the host's restoration responsibility."""

import os
import signal
import subprocess
from multiprocessing import get_context
from typing import TypedDict

import pytest

from sol_execbench.core.bench import (
    clock_lock,
    clock_ownership,
    host_clock_guard,
)
from sol_execbench.core.bench.clock_ownership import (
    ENV_CLOCK_OWNER,
    ClockDevice,
    ClockOwnerRecord,
    ClockOwnershipError,
    ClockOwnerState,
    verify_clock_owner,
)
from sol_execbench.core.bench.gpu_lock import acquire_gpu_lock

DEVICE = ClockDevice(
    bdf="0000:05:00.0", uuid="11111111-2222-3333-4444-555555555555"
)


class HardwareState(TypedDict):
    level: str
    writes: list[str]
    runner: list[str]
    fault: BaseException | None
    restore: bool


@pytest.fixture
def hardware(monkeypatch, tmp_path):
    return mock_hardware(monkeypatch, tmp_path)


def mock_hardware(monkeypatch, tmp_path):
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("SOL_EXECBENCH_CLOCK_STATE_DIR", str(tmp_path))
    monkeypatch.delenv(ENV_CLOCK_OWNER, raising=False)
    state: HardwareState = {
        "level": "AUTO",
        "writes": [],
        "runner": [],
        "fault": None,
        "restore": True,
    }

    def levels(*_):
        return (f"AMDSMI_DEV_PERF_LEVEL_{state['level']}",)

    def write(command):
        level = command[command.index("-l") + 1]
        # The durable record and live owner lock must predate even the first write.
        token = os.environ[ENV_CLOCK_OWNER]
        assert (
            verify_clock_owner(tmp_path, token, (DEVICE,)).state
            == ClockOwnerState.ACTIVE
        )
        state["writes"].append(level)
        if level == "AUTO" and not state["restore"]:
            raise subprocess.CalledProcessError(1, command)
        state["level"] = level
        if level == "STABLE_PEAK" and state["fault"] is not None:
            raise state["fault"]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        host_clock_guard, "observe_clock_devices", lambda: (DEVICE,)
    )
    monkeypatch.setattr(host_clock_guard, "observe_clock_levels", levels)
    monkeypatch.setattr(clock_lock, "observe_clock_levels", levels)
    monkeypatch.setattr(clock_lock, "_run_checked_amd_smi", write)
    monkeypatch.setattr(clock_lock.time, "sleep", lambda _: None)
    return state


def run_guard(hardware, *, container=False):
    def runner(command, environment):
        hardware["runner"].append(environment["SOL_EXECBENCH_CLOCKS_LOCKED"])
        return subprocess.CompletedProcess(command, 0)

    command = ("docker", "run") if container else ("test-workload",)
    return host_clock_guard.run_with_host_clock_guard(command, runner=runner)


def owner_record(tmp_path):
    paths = list((tmp_path / clock_ownership.LEASE_DIRECTORY).glob("*.json"))
    assert len(paths) == 1
    return ClockOwnerRecord.model_validate_json(paths[0].read_text())


def test_owner_precedes_first_mutation_and_survives_until_restore(
    hardware, tmp_path
):
    assert run_guard(hardware, container=True) == 0
    assert hardware["writes"] == ["STABLE_PEAK", "AUTO"]
    assert hardware["runner"] == ["1"]
    assert owner_record(tmp_path).state == ClockOwnerState.RESTORED
    assert owner_record(tmp_path).container_supervised


@pytest.mark.parametrize(
    "restore", [True, False], ids=["verified-rollback", "failed-rollback"]
)
@pytest.mark.parametrize("failure", ["command", "readback"])
def test_partial_setup_failure_can_continue_only_after_verified_rollback(
    hardware, tmp_path, monkeypatch, restore, failure
):
    if failure == "command":
        hardware["fault"] = subprocess.CalledProcessError(1, "amd-smi")
    else:
        monkeypatch.setattr(clock_lock, "verify_clocks", lambda *_: False)
    hardware["restore"] = restore
    if restore:
        with pytest.raises(
            clock_lock.ClockSetupError, match="clock_setup_failed"
        ):
            run_guard(hardware, container=True)
        assert hardware["runner"] == []
        assert owner_record(tmp_path).state == ClockOwnerState.RESTORED
        with acquire_gpu_lock(timeout_seconds=0.01):
            pass
    else:
        with pytest.raises(
            clock_lock.ClockRestoreError, match="clock_restore_unverified"
        ):
            run_guard(hardware, container=True)
        assert hardware["runner"] == []
        assert owner_record(tmp_path).state == ClockOwnerState.RESTORE_FAILED
        with (
            pytest.raises(ClockOwnershipError, match="clock_recovery_required"),
            acquire_gpu_lock(timeout_seconds=0.01),
        ):
            pytest.fail("failed rollback must block subsequent work")
    assert not owner_record(tmp_path).container_supervised


@pytest.mark.parametrize(
    "fault",
    [KeyboardInterrupt(), SystemExit(143)],
    ids=["interrupt", "termination"],
)
def test_interruption_during_acquisition_restores_without_running(
    hardware, tmp_path, fault
):
    hardware["fault"] = fault
    with pytest.raises(type(fault)):
        run_guard(hardware)
    assert hardware["writes"] == ["STABLE_PEAK", "AUTO"]
    assert hardware["runner"] == []
    assert owner_record(tmp_path).state == ClockOwnerState.RESTORED


def test_publication_failure_never_touches_hardware(hardware, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("cannot persist owner")

    monkeypatch.setattr(clock_ownership, "atomic_write_json_value", unavailable)
    with pytest.raises(OSError, match="cannot persist owner"):
        run_guard(hardware)
    assert hardware["writes"] == []
    assert hardware["runner"] == []


def test_failed_final_receipt_never_downgrades_setup_error(
    hardware, monkeypatch
):
    hardware["fault"] = subprocess.CalledProcessError(1, "amd-smi")
    write = clock_ownership.atomic_write_json_value

    def fail_final(path, record):
        if record["state"] == "restored":
            raise OSError("cannot persist restored state")
        write(path, record)

    monkeypatch.setattr(clock_ownership, "atomic_write_json_value", fail_final)
    with pytest.raises(OSError, match="cannot persist restored state"):
        run_guard(hardware)
    assert hardware["writes"] == ["STABLE_PEAK", "AUTO"]
    assert hardware["runner"] == []


def test_sigterm_during_stabilization_restores(hardware, monkeypatch, tmp_path):
    monkeypatch.setattr(
        clock_lock.time, "sleep", lambda _: os.kill(os.getpid(), signal.SIGTERM)
    )
    with pytest.raises(SystemExit):
        run_guard(hardware)
    assert hardware["writes"] == ["STABLE_PEAK", "AUTO"]
    assert hardware["runner"] == []
    assert owner_record(tmp_path).state == ClockOwnerState.RESTORED


def crash_during_stabilization(directory):
    """Spawn target with simulated hardware; never execute AMD-SMI."""

    def stop(_):
        os.kill(os.getpid(), signal.SIGKILL)

    with pytest.MonkeyPatch.context() as monkeypatch:
        state = mock_hardware(monkeypatch, directory)
        monkeypatch.setattr(clock_lock.time, "sleep", stop)
        run_guard(state, container=True)


def test_sigkill_during_stabilization_leaves_recovery_gate(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("SOL_EXECBENCH_CLOCK_STATE_DIR", str(tmp_path))
    child = get_context("spawn").Process(
        target=crash_during_stabilization, args=(tmp_path,)
    )
    child.start()
    try:
        child.join(timeout=10)
        assert child.exitcode == -signal.SIGKILL
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5)
        child.close()
    record = owner_record(tmp_path)
    assert record.state == ClockOwnerState.ACTIVE
    assert record.ended_at is None
    assert not record.container_supervised
    with (
        pytest.raises(ClockOwnershipError, match="clock_recovery_required"),
        acquire_gpu_lock(timeout_seconds=0.01),
    ):
        pytest.fail("crashed acquisition must block subsequent work")


def test_direct_clock_acquirer_requires_native_execution(monkeypatch):
    from sol_execbench.core.bench.gpu_execution import acquire_clock_lock
    from sol_execbench.core.bench.gpu_lock import GPULockVerificationError

    monkeypatch.setattr(
        clock_lock, "set_peak_clocks", lambda *_: pytest.fail("unowned setter")
    )
    with pytest.raises((GPULockVerificationError, FileNotFoundError)):
        acquire_clock_lock()
