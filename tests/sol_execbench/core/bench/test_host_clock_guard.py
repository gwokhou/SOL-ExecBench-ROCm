# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from sol_execbench.core.bench import clock_ownership, host_clock_guard


@pytest.fixture(autouse=True)
def owner_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("SOL_EXECBENCH_CLOCK_STATE_DIR", str(tmp_path))
    monkeypatch.delenv(clock_ownership.ENV_CLOCK_OWNER, raising=False)
    monkeypatch.setattr(
        host_clock_guard,
        "observe_clock_devices",
        lambda: (
            clock_ownership.ClockDevice(
                bdf="0000:05:00.0", uuid="11111111-2222-3333-4444-555555555555"
            ),
        ),
    )
    monkeypatch.setattr(
        host_clock_guard,
        "observe_clock_levels",
        lambda *_: ("AMDSMI_DEV_PERF_LEVEL_AUTO",),
    )
    monkeypatch.setattr(host_clock_guard, "set_peak_clocks", lambda *_: None)
    monkeypatch.setattr(host_clock_guard, "unlock_clocks", lambda *_: True)


@contextlib.contextmanager
def _gpu_lock() -> Iterator[None]:
    yield


@pytest.mark.parametrize(
    "suffix", ["", "/", "/."], ids=["plain", "slash", "dot"]
)
@pytest.mark.parametrize(
    "explicit", [False, True], ids=["inherited", "explicit"]
)
def test_guard_accepts_equivalent_lock_directory_spelling(
    tmp_path, monkeypatch, suffix, explicit
):
    directory = f"{tmp_path}{suffix}"
    monkeypatch.setenv("GPU_LOCK_DIR", directory)
    observed = {}

    def runner(command, environment):
        observed.update(environment)
        return subprocess.CompletedProcess(command, 0)

    assert (
        host_clock_guard.run_with_host_clock_guard(
            ("unused-cpu-test-command",),
            environment={"GPU_LOCK_DIR": directory} if explicit else None,
            runner=runner,
        )
        == 0
    )
    assert observed["GPU_LOCK_DIR"] == str(Path(directory))


@pytest.mark.parametrize("key", ["GPU_LOCK_DIR", "GPU_LOCK_UUID"])
def test_guard_rejects_different_child_lock_identity_before_hardware(
    tmp_path, monkeypatch, key
):
    def forbidden():
        pytest.fail(
            "conflicting configuration must fail before hardware access"
        )

    monkeypatch.setattr(host_clock_guard, "observe_clock_devices", forbidden)
    value = (
        str(tmp_path / "different")
        if key == "GPU_LOCK_DIR"
        else "11111111-2222-3333-4444-666666666666"
    )
    with pytest.raises(ValueError, match=f"child {key} conflicts"):
        host_clock_guard.run_with_host_clock_guard(
            ("unused-cpu-test-command",), environment={key: value}
        )


def test_guard_publishes_verified_host_state_and_releases_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_calls: list[bool] = []
    monkeypatch.setattr(
        host_clock_guard,
        "acquire_gpu_lock",
        lambda **kwargs: _gpu_lock(),
    )

    def unlock(*_) -> bool:
        release_calls.append(True)
        return True

    monkeypatch.setattr(
        host_clock_guard,
        "unlock_clocks",
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
    assert observed["GPU_LOCK_HELD_BY_HOST"] == "1"
    assert release_calls == [True]


def test_guard_runs_unlocked_without_false_clock_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        host_clock_guard,
        "acquire_gpu_lock",
        lambda **kwargs: _gpu_lock(),
    )
    monkeypatch.setattr(
        host_clock_guard,
        "observe_clock_levels",
        lambda *_: None,
    )
    observed: dict[str, str] = {}

    def runner(command, environment):
        observed.update(environment)
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(
        clock_ownership.ClockOwnershipError, match="initial_policy_unverified"
    ):
        host_clock_guard.run_with_host_clock_guard(
            ("true",), environment={}, runner=runner
        )
    assert observed == {}


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


def test_container_cleanup_failure_keeps_clock_policy_and_recovery_gate(
    monkeypatch, tmp_path
):
    restores = []
    monkeypatch.setattr(
        host_clock_guard,
        "unlock_clocks",
        lambda *_: restores.append(True) or True,
    )

    def runner(_command, _environment):
        raise host_clock_guard.ContainerCleanupError("container remains")

    with pytest.raises(
        host_clock_guard.ContainerCleanupError, match="container remains"
    ):
        host_clock_guard.run_with_host_clock_guard(
            ("docker", "run"), runner=runner
        )
    assert restores == []
    with pytest.raises(
        clock_ownership.ClockOwnershipError, match="clock_recovery_required"
    ):
        clock_ownership.require_no_clock_recovery(
            tmp_path, gpu_uuid="11111111-2222-3333-4444-555555555555"
        )
