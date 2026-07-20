from __future__ import annotations

import fcntl
import os

import pytest

from sol_execbench.core.bench.gpu_lock import acquire_gpu_lock, gpu_lock_directory


def test_gpu_lock_uses_configured_shared_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("SOL_EXECBENCH_GPU_LOCK_DIR", str(tmp_path))

    with acquire_gpu_lock(2, timeout_seconds=0.1):
        assert gpu_lock_directory() == tmp_path
        assert (tmp_path / "gpu-2.lock").read_text().startswith("pid=")


def test_gpu_lock_rejects_concurrent_evaluation(tmp_path, monkeypatch):
    monkeypatch.setenv("SOL_EXECBENCH_GPU_LOCK_DIR", str(tmp_path))

    with acquire_gpu_lock(timeout_seconds=0.1):
        with pytest.raises(TimeoutError, match="GPU 0 is busy"):
            with acquire_gpu_lock(timeout_seconds=0.01):
                pass


def test_gpu_lock_reuses_verified_entrypoint_file_descriptor(tmp_path, monkeypatch):
    monkeypatch.setenv("SOL_EXECBENCH_GPU_LOCK_DIR", str(tmp_path))
    lock_path = tmp_path / "gpu-0.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_APPEND | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setenv("SOL_EXECBENCH_GPU_LOCK_FD", str(descriptor))

        with acquire_gpu_lock(timeout_seconds=0.01):
            assert lock_path.exists()
    finally:
        os.close(descriptor)
