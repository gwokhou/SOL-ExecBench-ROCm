# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for sol_execbench.core.bench.clock_lock."""

import gc
from unittest.mock import MagicMock, patch

import pytest

from sol_execbench.core.bench import clock_lock as clock_lock_module
from sol_execbench.core.bench.clock_lock import (
    ClockLockLease,
    are_clocks_locked,
    unlock_clocks,
    verify_clocks,
)
from sol_execbench.core.bench.gpu_execution import (
    acquire_clock_lock,
    lock_clocks,
)

_MODULE = "sol_execbench.core.bench.clock_lock"


@pytest.fixture(autouse=True)
def _mock_tool_paths(monkeypatch):
    monkeypatch.setattr(
        clock_lock_module,
        "clock_executable",
        lambda: "amd-smi",
    )


class TestLockClocks:
    def test_direct_call_cannot_set_clocks_without_native_boundary(self):
        from sol_execbench.core.bench.gpu_lock import GPULockVerificationError

        with (
            patch(f"{_MODULE}.subprocess.run") as run,
            pytest.raises((FileNotFoundError, GPULockVerificationError)),
        ):
            acquire_clock_lock()
        run.assert_not_called()

    @pytest.mark.usefixtures("native_gpu_cpu_observation")
    @pytest.mark.parametrize("locked", [False, True])
    def test_borrows_target_without_setting_or_restoring(self, locked):
        from sol_execbench.core.bench import gpu_execution

        with (
            patch.object(
                gpu_execution, "verify_clocks", return_value=locked
            ) as verify,
            patch(f"{_MODULE}.subprocess.run") as run,
            gpu_execution.native_gpu_execution(),
        ):
            with acquire_clock_lock() as lease:
                assert lease.locked is locked
                assert not lease.acquired
            assert lock_clocks() is locked
        assert verify.call_args.args == (
            "11111111-2222-3333-4444-555555555555",
        )
        run.assert_not_called()


class TestClockLockLease:
    def test_exposes_owned_lease_state(self):
        lease = ClockLockLease(locked=True, acquired=True)

        assert lease.active is True
        assert lease.released is False
        assert lease.detached is False
        lease.detach()
        assert lease.active is False
        assert lease.released is False
        assert lease.detached is True

    def test_context_manager_releases_only_owned_lock(self):
        owned_lease = ClockLockLease(locked=True, acquired=True)
        external_lease = ClockLockLease(locked=True, acquired=False)
        with patch(f"{_MODULE}.unlock_clocks", return_value=True) as unlock:
            with owned_lease:
                pass
            with external_lease:
                pass

        unlock.assert_called_once_with()
        assert owned_lease.released is True
        assert external_lease.released is True

    def test_failed_release_can_be_retried(self):
        lease = ClockLockLease(locked=True, acquired=True)
        with patch(
            f"{_MODULE}.unlock_clocks",
            side_effect=[False, True],
        ) as unlock:
            assert lease.release() is False
            assert lease.active is True
            assert lease.released is False
            assert lease.release() is True
            assert lease.active is False
            assert lease.released is True
            assert lease.release() is True

        assert unlock.call_count == 2

    def test_cleanup_failure_after_success_is_reported(self):
        lease = ClockLockLease(locked=True, acquired=True)
        with (
            patch(f"{_MODULE}.unlock_clocks", return_value=False),
            pytest.raises(RuntimeError, match="failed to reset"),
            lease,
        ):
            pass
        lease.detach()

    def test_cleanup_failure_does_not_mask_body_exception(self):
        lease = ClockLockLease(locked=True, acquired=True)
        with (
            patch(f"{_MODULE}.unlock_clocks", return_value=False),
            pytest.raises(ValueError, match="body failed") as caught,
            lease,
        ):
            raise ValueError("body failed")

        assert caught.value.__notes__ == [
            "failed to reset and verify owned GPU policy at AUTO",
        ]
        lease.detach()

    def test_detached_lease_rejects_release(self):
        lease = ClockLockLease(locked=True, acquired=True)
        lease.detach()

        with pytest.raises(RuntimeError, match="detached"):
            lease.release()

    def test_garbage_collected_owned_lease_emits_stable_error(self, caplog):
        lease = ClockLockLease(locked=True, acquired=True)
        with patch(f"{_MODULE}.unlock_clocks", return_value=False):
            assert lease.release() is False

        del lease
        gc.collect()

        assert "gpu_clock_lease_leaked" in caplog.text

    @pytest.mark.parametrize("resolution", ("release", "detach"))
    def test_resolved_owned_lease_does_not_emit_leak_error(
        self,
        resolution,
        caplog,
    ):
        lease = ClockLockLease(locked=True, acquired=True)
        if resolution == "release":
            with patch(f"{_MODULE}.unlock_clocks", return_value=True):
                assert lease.release() is True
        else:
            lease.detach()

        del lease
        gc.collect()

        assert "gpu_clock_lease_leaked" not in caplog.text

    @pytest.mark.parametrize(
        "lease",
        (
            ClockLockLease(locked=True, acquired=False),
            ClockLockLease(locked=False, acquired=False),
        ),
    )
    def test_non_owned_lease_does_not_emit_leak_error(self, lease, caplog):
        del lease
        gc.collect()

        assert "gpu_clock_lease_leaked" not in caplog.text


class TestVerifyClocks:
    def _make_smi_result(self, stdout: str, returncode: int = 0):
        return MagicMock(returncode=returncode, stdout=stdout, stderr="")

    def test_stable_peak_detected(self):
        result = self._make_smi_result(
            '{"gpu_data": [{"gpu": 0, "perf_level": '
            '"AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK"}]}',
        )
        with patch(f"{_MODULE}.subprocess.run", return_value=result) as run:
            assert verify_clocks() is True
        run.assert_called_once_with(
            ["amd-smi", "metric", "-l", "--json"],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )

    def test_auto_detected_as_not_locked(self):
        result = self._make_smi_result(
            '{"gpu_data": [{"gpu": 0, "perf_level": "AMDSMI_DEV_PERF_LEVEL_AUTO"}]}',
        )
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks() is False

    def test_mixed_gpu_levels_are_not_locked(self):
        result = self._make_smi_result(
            '{"gpu_data": ['
            '{"gpu": 0, "perf_level": "AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK"},'
            '{"gpu": 1, "perf_level": "AMDSMI_DEV_PERF_LEVEL_AUTO"}]}',
        )
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks() is False

    def test_all_gpu_levels_must_be_stable_peak(self):
        result = self._make_smi_result(
            '{"gpu_data": ['
            '{"gpu": 0, "perf_level": "AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK"},'
            '{"gpu": 1, "perf_level": "AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK"}]}',
        )
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks() is True

    def test_amd_smi_not_found(self):
        with patch(f"{_MODULE}.subprocess.run", side_effect=FileNotFoundError):
            assert verify_clocks() is False

    def test_amd_smi_nonzero_exit(self):
        import subprocess as sp

        with patch(
            f"{_MODULE}.subprocess.run",
            side_effect=sp.CalledProcessError(1, "amd-smi"),
        ):
            assert verify_clocks() is False

    def test_empty_gpu_data(self):
        result = self._make_smi_result('{"gpu_data": []}')
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks() is False

    def test_invalid_json(self):
        result = self._make_smi_result("not-json")
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks() is False


class TestUnlockClocks:
    def test_unlocks_and_verifies_all_gpus_with_amd_smi(self):
        reset = MagicMock(returncode=0, stdout="", stderr="")
        verify = MagicMock(
            returncode=0,
            stdout=(
                '{"gpu_data": ['
                '{"gpu": 0, "perf_level": "AMDSMI_DEV_PERF_LEVEL_AUTO"},'
                '{"gpu": 1, "perf_level": "AMDSMI_DEV_PERF_LEVEL_AUTO"}]}'
            ),
            stderr="",
        )
        with patch(
            f"{_MODULE}.subprocess.run",
            side_effect=[reset, verify],
        ) as mock_run:
            assert unlock_clocks() is True

        assert mock_run.call_args_list[0].args[0] == [
            "sudo",
            "-n",
            "amd-smi",
            "set",
            "-l",
            "AUTO",
        ]

    def test_returns_false_when_auto_cannot_be_verified(self):
        reset = MagicMock(returncode=0, stdout="", stderr="")
        verify = MagicMock(
            returncode=0,
            stdout=(
                '{"gpu_data": [{"gpu": 0, "perf_level": '
                '"AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK"}]}'
            ),
            stderr="",
        )
        with patch(f"{_MODULE}.subprocess.run", side_effect=[reset, verify]):
            assert unlock_clocks() is False

    def test_does_not_raise_on_failure(self):
        with patch(
            f"{_MODULE}.subprocess.run",
            side_effect=Exception("no sudo"),
        ):
            assert unlock_clocks() is False


class TestAreClocksLocked:
    def test_returns_true_when_env_set(self, monkeypatch):
        monkeypatch.setenv("SOL_EXECBENCH_CLOCKS_LOCKED", "1")
        assert are_clocks_locked() is True

    def test_returns_false_when_env_zero(self, monkeypatch):
        monkeypatch.setenv("SOL_EXECBENCH_CLOCKS_LOCKED", "0")
        assert are_clocks_locked() is False

    def test_returns_false_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("SOL_EXECBENCH_CLOCKS_LOCKED", raising=False)
        assert are_clocks_locked() is False
