# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Ownership proof, stale records and release ordering without GPU mutation."""

import os
import signal

import pytest

from sol_execbench.core.bench.clock_lock import ClockLockLease
from sol_execbench.core.bench.clock_ownership import (
    ENV_CLOCK_OWNER,
    LEASE_DIRECTORY,
    ClockDevice,
    ClockOwnerRecord,
    ClockOwnershipError,
    ClockOwnerState,
    published_clock_owner,
    verify_clock_owner,
)
from sol_execbench.core.bench.gpu_lock import acquire_gpu_lock
from sol_execbench.core.process.termination import cleanup_on_termination

DEVICE = ClockDevice(
    bdf="0000:05:00.0", uuid="11111111-2222-3333-4444-555555555555"
)


@pytest.fixture(autouse=True)
def isolation(monkeypatch, tmp_path):
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("SOL_EXECBENCH_CLOCK_STATE_DIR", str(tmp_path))
    monkeypatch.delenv(ENV_CLOCK_OWNER, raising=False)


@pytest.mark.parametrize("exit_kind", ["success", "exception", "sigterm"])
def test_owner_records_restore_before_gpu_lock_release(tmp_path, exit_kind):
    events = []

    def restore():
        with (
            pytest.raises(TimeoutError),
            acquire_gpu_lock(timeout_seconds=0.01),
        ):
            pytest.fail("GPU lock released before restoration")
        events.append("restore")
        return True

    lease = ClockLockLease(locked=True, acquired=True, restore=restore)

    def run():
        with (
            cleanup_on_termination(),
            acquire_gpu_lock(timeout_seconds=0.1),
            published_clock_owner(lease, tmp_path, (DEVICE,)) as token,
        ):
            assert token is not None
            assert token == os.environ[ENV_CLOCK_OWNER]
            assert (
                verify_clock_owner(tmp_path, token, (DEVICE,)).state
                == ClockOwnerState.ACTIVE
            )
            if exit_kind == "exception":
                raise ValueError("body")
            if exit_kind == "sigterm":
                os.kill(os.getpid(), signal.SIGTERM)

    if exit_kind == "success":
        run()
    else:
        with pytest.raises(
            ValueError if exit_kind == "exception" else SystemExit
        ):
            run()
    assert events == ["restore"]
    record = ClockOwnerRecord.model_validate_json(
        next((tmp_path / LEASE_DIRECTORY).glob("*.json")).read_text()
    )
    assert record.state == ClockOwnerState.RESTORED
    assert ENV_CLOCK_OWNER not in os.environ
    with acquire_gpu_lock(timeout_seconds=0.1):
        pass
    assert lease.release()
    assert events == ["restore"]


def test_failed_restore_quarantines_next_gpu_acquisition(tmp_path):
    lease = ClockLockLease(locked=True, acquired=True, restore=lambda: False)
    with (
        pytest.raises(RuntimeError, match="failed to reset"),
        acquire_gpu_lock(timeout_seconds=0.1),
        published_clock_owner(lease, tmp_path, (DEVICE,)),
    ):
        pass
    with (
        pytest.raises(ClockOwnershipError, match="clock_recovery_required"),
        acquire_gpu_lock(timeout_seconds=0.1),
    ):
        pytest.fail("unverified state must block future GPU work")
    assert not lease.released
    lease.detach()


def test_token_without_live_lock_is_not_ownership(tmp_path):
    lease = ClockLockLease(locked=True, acquired=True, restore=lambda: True)
    with published_clock_owner(lease, tmp_path, (DEVICE,)) as token:
        assert token is not None
        record = verify_clock_owner(tmp_path, token, (DEVICE,))
    # Simulate a crash leaving an active-looking record, but no held lock.
    path = tmp_path / LEASE_DIRECTORY / f"{token}.json"
    path.write_text(record.model_dump_json())
    with pytest.raises(ClockOwnershipError, match="clock_owner_not_live"):
        verify_clock_owner(tmp_path, token, (DEVICE,))


def test_live_owner_must_cover_requested_identity(tmp_path):
    lease = ClockLockLease(locked=True, acquired=True, restore=lambda: True)
    with published_clock_owner(lease, tmp_path, (DEVICE,)) as token:
        assert token is not None
        with pytest.raises(ClockOwnershipError, match="device_mismatch"):
            verify_clock_owner(
                tmp_path, token, (ClockDevice(bdf=DEVICE.bdf, uuid="wrong"),)
            )


@pytest.mark.parametrize(
    "auto", [False, True], ids=["still-peak", "restored-auto"]
)
def test_recovery_requires_live_auto_before_clearing_gate(
    tmp_path, monkeypatch, auto
):
    from sol_execbench.core.bench import clock_recovery

    lease = ClockLockLease(locked=True, acquired=True, restore=lambda: False)
    with (
        pytest.raises(RuntimeError, match="failed to reset"),
        published_clock_owner(lease, tmp_path, (DEVICE,)) as token,
    ):
        pass
    lease.detach()
    assert token is not None
    monkeypatch.setattr(
        clock_recovery, "observe_clock_devices", lambda: (DEVICE,)
    )

    def read_device(command, _bdf, *_arguments):
        if command == "process":
            return '[{"gpu": 0, "process_list": []}]'
        level = "AUTO" if auto else "STABLE_PEAK"
        return (
            '{"gpu_data": [{"gpu":0,"perf_level":"AMDSMI_DEV_PERF_LEVEL_'
            + level
            + '"}]}'
        )

    monkeypatch.setattr(clock_recovery, "_read_device", read_device)
    if auto:
        clock_recovery.confirm_clock_recovery(tmp_path, token)
        with acquire_gpu_lock(timeout_seconds=0.1):
            pass
        assert (
            tmp_path / LEASE_DIRECTORY / "recovery-history" / f"{token}.json"
        ).exists()
    else:
        with pytest.raises(ClockOwnershipError, match="auto_unverified"):
            clock_recovery.confirm_clock_recovery(tmp_path, token)
        with (
            pytest.raises(ClockOwnershipError, match="clock_recovery_required"),
            acquire_gpu_lock(timeout_seconds=0.1),
        ):
            pytest.fail("failed verification must preserve the gate")


def test_recovery_cannot_clear_live_owner(tmp_path):
    from sol_execbench.core.bench.clock_recovery import confirm_clock_recovery

    lease = ClockLockLease(locked=True, acquired=True, restore=lambda: True)
    with published_clock_owner(lease, tmp_path, (DEVICE,)) as token:
        assert token is not None
        with pytest.raises(RuntimeError, match="already held"):
            confirm_clock_recovery(tmp_path, token)
