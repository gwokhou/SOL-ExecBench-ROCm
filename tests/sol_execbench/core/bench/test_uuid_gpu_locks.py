"""CPU fixtures for the shared UUID contract; no hardware identity is claimed."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from sol_execbench.core.bench import gpu_lock
from sol_execbench.core.bench.clock_lock import ClockLockLease
from sol_execbench.core.bench.clock_ownership import (
    ENV_CLOCK_OWNER,
    ClockDevice,
    ClockOwnerRecord,
    ClockOwnershipError,
    ClockOwnerState,
    clock_owner_path,
    published_clock_owner,
    verify_clock_owner,
    write_clock_record,
)
from sol_execbench.core.bench.gpu_lock_paths import (
    LEGACY_GPU_LOCK_ENV,
    configured_gpu_locks,
    open_gpu_lock,
    prepare_gpu_locks,
)
from sol_execbench.core.process.locks import exclusive_file_lock

UUID = "11111111-2222-3333-4444-555555555555"
OTHER_UUID = "11111111-2222-3333-4444-666666666666"


@pytest.fixture
def paths(tmp_path, monkeypatch):
    for name in (*LEGACY_GPU_LOCK_ENV, "GPU_LOCK_HELD_BY_HOST", "GPU_LOCK_FD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("SOL_EXECBENCH_ADMISSION_RECORD", raising=False)
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setenv("GPU_LOCK_UUID", UUID)
    monkeypatch.setenv(
        "SOL_EXECBENCH_CLOCK_STATE_DIR", str(tmp_path / "private")
    )
    monkeypatch.setattr(
        "sol_execbench.core.bench.clock_ownership.LEGACY_CLOCK_STATE_DIRECTORY",
        tmp_path / "old-private",
    )
    return configured_gpu_locks()


def test_uuid_names_and_normalization(paths, monkeypatch):
    monkeypatch.setenv("GPU_LOCK_UUID", f" {UUID.upper()} ")
    parsed = configured_gpu_locks()
    assert parsed.uuid == UUID
    assert parsed.ownership.name == f"gpu-{UUID}.L1-ownership.lock"
    assert parsed.execution.name == f"gpu-{UUID}.L2-execution.lock"
    assert not paths.directory.exists()


@pytest.mark.parametrize(
    "name,value",
    [
        ("GPU_LOCK_DIR", "relative"),
        ("GPU_LOCK_DIR", ""),
        ("GPU_LOCK_UUID", "GPU-a30fa29c1cca1511"),
        ("GPU_LOCK_UUID", ""),
        ("GPU_LOCK_HELD_BY_HOST", "0"),
        ("GPU_LOCK_FD", "-1"),
        *[(name, "ignored") for name in LEGACY_GPU_LOCK_ENV],
    ],
)
def test_invalid_or_legacy_configuration_stops(paths, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="GPU_LOCK|legacy GPU"):
        configured_gpu_locks()
    assert not paths.directory.exists()


def test_same_uuid_contends_and_different_uuid_is_independent(
    paths, monkeypatch
):
    with gpu_lock.acquire_gpu_lock():
        with (
            pytest.raises(TimeoutError),
            gpu_lock.acquire_gpu_lock(timeout_seconds=0.01),
        ):
            pass
        monkeypatch.setenv("GPU_LOCK_UUID", OTHER_UUID)
        with gpu_lock.acquire_gpu_lock(timeout_seconds=0.01):
            pass


def test_two_uuid_owners_can_remain_active_together(paths, monkeypatch):
    first = ClockDevice(bdf="0000:05:00.0", uuid=UUID)
    second = ClockDevice(bdf="0000:14:00.0", uuid=OTHER_UUID)
    lease_a = ClockLockLease(locked=True, acquired=True, restore=lambda: True)
    with (
        gpu_lock.acquire_gpu_lock(timeout_seconds=0.1),
        published_clock_owner(lease_a, paths.directory, (first,)) as token_a,
    ):
        monkeypatch.setenv("GPU_LOCK_UUID", OTHER_UUID)
        with gpu_lock.acquire_gpu_lock(timeout_seconds=0.1):
            lease_b = ClockLockLease(
                locked=True, acquired=True, restore=lambda: True
            )
            with published_clock_owner(
                lease_b, paths.directory, (second,)
            ) as token_b:
                assert token_a is not None and token_b is not None
                for token, device in ((token_a, first), (token_b, second)):
                    assert (
                        verify_clock_owner(
                            paths.directory, token, (device,)
                        ).state
                        is ClockOwnerState.ACTIVE
                    )
    assert lease_a.released and lease_b.released


@pytest.mark.parametrize("location", ["private", "old-private"])
@pytest.mark.parametrize("state", list(ClockOwnerState))
@pytest.mark.parametrize("coverage", ["same", "other", "both"])
def test_recovery_gate_is_scoped_to_recorded_physical_devices(
    paths, tmp_path, location, state, coverage
):
    devices = {
        "same": (ClockDevice(bdf="0000:05:00.0", uuid=UUID),),
        "other": (ClockDevice(bdf="0000:14:00.0", uuid=OTHER_UUID),),
    }
    devices["both"] = devices["other"] + devices["same"]
    record = ClockOwnerRecord(
        lease_id="c" * 32,
        devices=devices[coverage],
        owner_pid=os.getpid(),
        state=state,
    )
    root = tmp_path / location / "clock-leases"
    root.mkdir(parents=True)
    (root / f"{record.lease_id}.json").write_text(record.model_dump_json())
    if coverage != "other" and state is not ClockOwnerState.RESTORED:
        with (
            pytest.raises(ClockOwnershipError, match="clock_recovery_required"),
            gpu_lock.acquire_gpu_lock(timeout_seconds=0.1),
        ):
            pytest.fail("same-device recovery must remain blocking")
    else:
        with gpu_lock.acquire_gpu_lock(timeout_seconds=0.1):
            pass


@pytest.mark.parametrize("location", ["private", "old-private"])
@pytest.mark.parametrize("damage", ["json", "schema", "devices", "uuid"])
def test_unidentifiable_recovery_record_still_blocks(
    paths, tmp_path, location, damage
):
    record = ClockOwnerRecord(
        lease_id="d" * 32,
        devices=(ClockDevice(bdf="0000:14:00.0", uuid=OTHER_UUID),),
        owner_pid=os.getpid(),
    ).model_dump(mode="json")
    if damage == "schema":
        record["schema_version"] = "unknown"
    elif damage == "devices":
        record["devices"] = []
    elif damage == "uuid":
        record["devices"].append({"bdf": "0000:05:00.0", "uuid": "unknown"})
    root = tmp_path / location / "clock-leases"
    root.mkdir(parents=True)
    (root / "damaged.json").write_text(
        "incomplete" if damage == "json" else json.dumps(record)
    )
    with (
        pytest.raises(ClockOwnershipError, match="clock_recovery_"),
        gpu_lock.acquire_gpu_lock(timeout_seconds=0.1),
    ):
        pytest.fail("an unknown device cannot be proven unrelated")


def test_recovery_matches_normalized_uuid_from_receipt(
    paths, tmp_path, monkeypatch
):
    uuid = "abcd1111-2222-3333-4444-555555555555"
    monkeypatch.setenv("GPU_LOCK_UUID", uuid)
    record = ClockOwnerRecord(
        lease_id="e" * 32,
        devices=(ClockDevice(bdf="0000:05:00.0", uuid=uuid.upper()),),
        owner_pid=os.getpid(),
    )
    root = tmp_path / "private" / "clock-leases"
    root.mkdir(parents=True)
    (root / f"{record.lease_id}.json").write_text(record.model_dump_json())
    with (
        pytest.raises(ClockOwnershipError, match="clock_recovery_required"),
        gpu_lock.acquire_gpu_lock(timeout_seconds=0.1),
    ):
        pytest.fail("record spelling must not hide the current device")


def test_symlink_rejected_without_changing_target(paths, tmp_path):
    target = tmp_path / "target"
    target.write_text("untouched")
    paths.directory.mkdir()
    paths.ownership.symlink_to(target)
    with pytest.raises(OSError, match="symbolic links"):
        prepare_gpu_locks(paths)
    assert target.read_text() == "untouched"


def test_host_missing_files_never_create_private_inode(paths, monkeypatch):
    monkeypatch.setenv("GPU_LOCK_HELD_BY_HOST", "1")
    with (
        pytest.raises(FileNotFoundError),
        gpu_lock.acquire_evaluation_gpu_lock(),
    ):
        pass
    assert not paths.directory.exists()


def test_real_inherited_fd_keeps_owner_and_wrong_inode_rejected(
    paths, monkeypatch
):
    prepare_gpu_locks(paths)
    with (
        _live_owner(paths, monkeypatch),
        open_gpu_lock(paths.ownership, create=False) as descriptor,
    ):
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setenv("GPU_LOCK_FD", str(descriptor))
        with (
            gpu_lock.acquire_evaluation_gpu_lock(timeout_seconds=0.01),
            gpu_lock.acquire_evaluation_gpu_lock(timeout_seconds=0.01),
        ):
            pass
        with (
            open_gpu_lock(paths.ownership, create=False) as competitor,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setenv("GPU_LOCK_FD", "999999")
        with (
            pytest.raises(gpu_lock.GPULockVerificationError),
            gpu_lock.acquire_gpu_lock(),
        ):
            pass
        with open_gpu_lock(paths.execution, create=False) as wrong:
            monkeypatch.setenv("GPU_LOCK_FD", str(wrong))
            with (
                pytest.raises(gpu_lock.GPULockVerificationError),
                gpu_lock.acquire_gpu_lock(),
            ):
                pass


def test_host_workers_compete_and_nested_thread_reuses(paths, monkeypatch):
    record = ClockOwnerRecord(
        lease_id="a" * 32,
        devices=(ClockDevice(bdf="0000:05:00.0", uuid=UUID),),
        owner_pid=os.getpid(),
    )
    path = clock_owner_path(paths.directory, record.lease_id)
    write_clock_record(
        path, record.model_copy(update={"state": ClockOwnerState.RESTORED})
    )
    with (
        gpu_lock.acquire_gpu_lock(),
        exclusive_file_lock(path.with_suffix(".lock")),
    ):
        write_clock_record(path, record)
        monkeypatch.setenv("GPU_LOCK_HELD_BY_HOST", "1")
        monkeypatch.setenv(ENV_CLOCK_OWNER, record.lease_id)
        with gpu_lock.acquire_evaluation_gpu_lock():
            with gpu_lock.acquire_evaluation_gpu_lock():
                pass

            def compete():
                with gpu_lock.acquire_evaluation_gpu_lock(timeout_seconds=0.01):
                    pytest.fail("sibling entered L2")

            with (
                ThreadPoolExecutor(max_workers=1) as executor,
                pytest.raises(TimeoutError),
            ):
                executor.submit(compete).result(timeout=3)
        with gpu_lock.acquire_evaluation_gpu_lock():
            pass


def test_old_private_pending_record_still_blocks(paths, monkeypatch, tmp_path):
    legacy = tmp_path / "old-private" / "clock-leases"
    legacy.mkdir(parents=True)
    (legacy / "bad.json").write_text("incomplete")
    with (
        pytest.raises(RuntimeError, match="clock_recovery_record_invalid"),
        gpu_lock.acquire_gpu_lock(),
    ):
        pass


@contextmanager
def _live_owner(paths, monkeypatch):
    record = ClockOwnerRecord(
        lease_id="b" * 32,
        devices=(ClockDevice(bdf="0000:05:00.0", uuid=UUID),),
        owner_pid=os.getpid(),
    )
    path = clock_owner_path(paths.directory, record.lease_id)
    write_clock_record(path, record)
    with exclusive_file_lock(path.with_suffix(".lock")):
        monkeypatch.setenv(ENV_CLOCK_OWNER, record.lease_id)
        yield
    write_clock_record(
        path, record.model_copy(update={"state": ClockOwnerState.RESTORED})
    )


def test_actual_child_inherits_l1_without_unlocking_owner(paths, monkeypatch):
    prepare_gpu_locks(paths)
    child = """
from sol_execbench.core.bench.gpu_lock import acquire_evaluation_gpu_lock, require_supervised_gpu_execution
with acquire_evaluation_gpu_lock(timeout_seconds=0.1):
    require_supervised_gpu_execution()
"""
    with (
        _live_owner(paths, monkeypatch),
        open_gpu_lock(paths.ownership, create=False) as descriptor,
    ):
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setenv("GPU_LOCK_FD", str(descriptor))
        result = subprocess.run(
            [sys.executable, "-B", "-c", child],
            pass_fds=(descriptor,),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        with (
            open_gpu_lock(paths.ownership, create=False) as competitor,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
