# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Iterator

import pytest

from sol_execbench.core.bench import host_clock_guard
from sol_execbench.core.bench.clock_lock import ClockLockLease


@contextlib.contextmanager
def _gpu_lock() -> Iterator[None]:
    yield


def test_guard_publishes_verified_host_state_and_releases_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = ClockLockLease(locked=True, acquired=True)
    release_calls: list[bool] = []
    monkeypatch.setattr(
        host_clock_guard, "acquire_gpu_lock", lambda **kwargs: _gpu_lock()
    )
    monkeypatch.setattr(host_clock_guard, "acquire_clock_lock", lambda: lease)

    def unlock() -> bool:
        release_calls.append(True)
        return True

    monkeypatch.setattr(
        "sol_execbench.core.bench.clock_lock.unlock_clocks",
        unlock,
    )
    observed: dict[str, str] = {}

    def runner(command, environment):
        assert command == ("docker", "run")
        observed.update(environment)
        return subprocess.CompletedProcess(command, 23)

    status = host_clock_guard.run_with_host_clock_guard(
        ("docker", "run"),
        environment={"PATH": "/bin"},
        runner=runner,
    )

    assert status == 23
    assert observed["SOL_EXECBENCH_CLOCKS_LOCKED"] == "1"
    assert observed["SOL_EXECBENCH_CLOCKS_MANAGED_BY_HOST"] == "1"
    assert observed["SOL_EXECBENCH_GPU_LOCK_MANAGED_BY_HOST"] == "1"
    assert release_calls == [True]


def test_guard_runs_unlocked_without_false_clock_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        host_clock_guard, "acquire_gpu_lock", lambda **kwargs: _gpu_lock()
    )
    monkeypatch.setattr(
        host_clock_guard,
        "acquire_clock_lock",
        lambda: ClockLockLease(locked=False, acquired=False),
    )
    observed: dict[str, str] = {}

    def runner(command, environment):
        observed.update(environment)
        return subprocess.CompletedProcess(command, 0)

    assert (
        host_clock_guard.run_with_host_clock_guard(
            ("true",),
            environment={},
            runner=runner,
        )
        == 0
    )
    assert observed["SOL_EXECBENCH_CLOCKS_LOCKED"] == "0"


def test_guard_reports_gpu_lock_timeout_as_temporary_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextlib.contextmanager
    def busy_lock(**kwargs) -> Iterator[None]:
        raise TimeoutError("GPU 0 is busy")
        yield

    monkeypatch.setattr(host_clock_guard, "acquire_gpu_lock", busy_lock)

    assert (
        host_clock_guard.run_with_host_clock_guard(
            ("true",),
            environment={},
        )
        == 75
    )


def test_guard_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        host_clock_guard.run_with_host_clock_guard(
            ("true",),
            environment={"SOL_EXECBENCH_GPU_LOCK_TIMEOUT_SECONDS": "0"},
        )
