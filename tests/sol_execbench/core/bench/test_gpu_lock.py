from __future__ import annotations

import contextlib
import fcntl
import importlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from inspect import unwrap
from types import SimpleNamespace

import pytest

from sol_execbench.core.bench.gpu_lock import (
    GPULockVerificationError,
    acquire_evaluation_gpu_lock,
    acquire_gpu_lock,
    gpu_lock_directory,
)


def test_gpu_lock_uses_configured_shared_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))

    with acquire_gpu_lock(timeout_seconds=0.1):
        assert gpu_lock_directory() == tmp_path
        assert (
            tmp_path
            / "gpu-11111111-2222-3333-4444-555555555555.L1-ownership.lock"
        ).is_file()


def test_gpu_lock_rejects_concurrent_evaluation(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))

    with (
        acquire_gpu_lock(timeout_seconds=0.1),
        pytest.raises(TimeoutError, match="GPU is busy"),
        acquire_gpu_lock(timeout_seconds=0.01),
    ):
        pass


def test_gpu_lock_reuses_verified_entrypoint_file_descriptor(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))
    lock_path = (
        tmp_path / "gpu-11111111-2222-3333-4444-555555555555.L1-ownership.lock"
    )
    from sol_execbench.core.bench.gpu_lock_paths import (
        configured_gpu_locks,
        prepare_gpu_locks,
    )

    prepare_gpu_locks(configured_gpu_locks())
    descriptor = os.open(lock_path, os.O_CREAT | os.O_APPEND | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setenv("GPU_LOCK_FD", str(descriptor))

        with (
            pytest.raises(
                GPULockVerificationError, match="requires native owner"
            ),
            acquire_gpu_lock(timeout_seconds=0.01),
        ):
            pytest.fail("bare inherited FD is not native ownership")
    finally:
        os.close(descriptor)


def test_evaluation_rejects_host_lock_without_native_owner(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("GPU_LOCK_HELD_BY_HOST", "1")

    monkeypatch.delenv("GPU_LOCK_HELD_BY_HOST")
    with acquire_gpu_lock(timeout_seconds=0.1):
        monkeypatch.setenv("GPU_LOCK_HELD_BY_HOST", "1")
        with (
            pytest.raises(
                GPULockVerificationError, match="native owner evidence"
            ),
            acquire_evaluation_gpu_lock(timeout_seconds=0.01),
        ):
            pass


def test_evaluation_rejects_unheld_host_managed_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("GPU_LOCK_HELD_BY_HOST", "1")

    from sol_execbench.core.bench.gpu_lock_paths import (
        configured_gpu_locks,
        prepare_gpu_locks,
    )

    monkeypatch.delenv("GPU_LOCK_HELD_BY_HOST")
    prepare_gpu_locks(configured_gpu_locks())
    monkeypatch.setenv("GPU_LOCK_HELD_BY_HOST", "1")
    with (
        pytest.raises(GPULockVerificationError, match="is not held"),
        acquire_evaluation_gpu_lock(timeout_seconds=0.01),
    ):
        pass


def test_nested_evaluation_reuses_only_its_thread_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.delenv("GPU_LOCK_HELD_BY_HOST", raising=False)
    with acquire_evaluation_gpu_lock(timeout_seconds=0.1):
        with (
            acquire_evaluation_gpu_lock(timeout_seconds=0.01),
            (
                tmp_path
                / "gpu-11111111-2222-3333-4444-555555555555.L1-ownership.lock"
            ).open("a+") as handle,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=1) as executor:

            def compete():
                with acquire_evaluation_gpu_lock(timeout_seconds=0.01):
                    pytest.fail("another thread borrowed the active lock")

            with pytest.raises(TimeoutError):
                executor.submit(compete).result(timeout=2)
    with acquire_gpu_lock(timeout_seconds=0.01):
        pass


def test_gpu_pytest_fixture_uses_native_lock(tmp_path, monkeypatch):
    conftest = importlib.import_module("conftest")

    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.delenv("GPU_LOCK_HELD_BY_HOST", raising=False)
    with conftest._real_gpu_execution_lock():
        with (
            (
                tmp_path
                / "gpu-11111111-2222-3333-4444-555555555555.L1-ownership.lock"
            ).open("a+") as handle,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with acquire_evaluation_gpu_lock(timeout_seconds=0.01):
            pass


def test_independent_process_contends_on_same_native_inode(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))
    command = [
        sys.executable,
        "-c",
        """
import fcntl, sys
with open(sys.argv[1], 'a+') as handle:
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(42)
""",
        str(
            tmp_path
            / "gpu-11111111-2222-3333-4444-555555555555.L1-ownership.lock"
        ),
    ]
    with acquire_evaluation_gpu_lock(timeout_seconds=0.1):
        assert subprocess.run(command, timeout=5, check=False).returncode == 42
    assert subprocess.run(command, timeout=5, check=False).returncode == 0


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_gpu_fixture_synchronizes_exception_before_unlock(
    tmp_path,
    monkeypatch,
    cleanup_fails,
):
    import torch

    conftest = importlib.import_module("conftest")
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(conftest, "_has_real_gpu_marker", lambda _: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    cleaned = []

    def synchronize():
        with (
            (
                tmp_path
                / "gpu-11111111-2222-3333-4444-555555555555.L1-ownership.lock"
            ).open("a+") as handle,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cleaned.append(True)
        if cleanup_fails:
            raise RuntimeError("synchronize failed")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    fixture = unwrap(conftest._serialize_real_gpu_test)(
        SimpleNamespace(node=None)
    )
    next(fixture)
    with pytest.raises(
        RuntimeError,
        match="synchronize failed" if cleanup_fails else "test failed",
    ):
        fixture.throw(RuntimeError("test failed"))
    assert cleaned == [True]
    from sol_execbench.core.bench.clock_ownership import ClockOwnershipError

    expected = (
        pytest.raises(ClockOwnershipError, match="clock_recovery_required")
        if cleanup_fails
        else contextlib.nullcontext()
    )
    with expected, acquire_gpu_lock(timeout_seconds=0.01):
        pass


pytestmark = pytest.mark.usefixtures("native_gpu_cpu_observation")
