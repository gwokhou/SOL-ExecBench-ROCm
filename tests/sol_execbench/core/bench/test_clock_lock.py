# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for sol_execbench.core.bench.clock_lock."""

from unittest.mock import MagicMock, patch

import pytest

from sol_execbench.core.bench import clock_lock as clock_lock_module
from sol_execbench.core.bench.clock_lock import (
    are_clocks_locked,
    lock_clocks,
    probe_clock_lock_available,
    unlock_clocks,
    verify_clocks,
)

_MODULE = "sol_execbench.core.bench.clock_lock"


@pytest.fixture(autouse=True)
def _mock_tool_paths(monkeypatch):
    monkeypatch.setattr(
        clock_lock_module,
        "resolve_rocm_tool_command",
        lambda _tool: "amd-smi",
    )


class TestProbeClockLockAvailable:
    def test_returns_true_when_all_lifecycle_commands_are_allowed(self):
        probe_result = MagicMock(returncode=0)
        with patch(f"{_MODULE}.subprocess.run", return_value=probe_result) as mock_run:
            result = probe_clock_lock_available()

        assert result is True
        assert [call.args[0] for call in mock_run.call_args_list] == [
            ["sudo", "-n", "-l", "--", "amd-smi", "version"],
            [
                "sudo",
                "-n",
                "-l",
                "--",
                "amd-smi",
                "set",
                "-l",
                "STABLE_PEAK",
            ],
            ["sudo", "-n", "-l", "--", "amd-smi", "set", "-l", "AUTO"],
        ]
        assert all(call.kwargs["timeout"] == 10 for call in mock_run.call_args_list)

    def test_returns_false_when_sudo_fails(self):
        probe_result = MagicMock(returncode=1)
        with patch(f"{_MODULE}.subprocess.run", return_value=probe_result):
            assert probe_clock_lock_available() is False

    def test_returns_false_when_amd_smi_not_found(self):
        with patch(f"{_MODULE}.subprocess.run", side_effect=FileNotFoundError):
            assert probe_clock_lock_available() is False


class TestLockClocks:
    def _patch_verify_and_sleep(self):
        return (
            patch(f"{_MODULE}.verify_clocks", return_value=True),
            patch(f"{_MODULE}.time.sleep"),
        )

    def test_locks_with_stable_peak(self):
        p_verify, p_sleep = self._patch_verify_and_sleep()
        with patch(f"{_MODULE}.subprocess.run") as mock_run, p_verify, p_sleep:
            result = lock_clocks()

        assert result is True
        assert mock_run.call_count == 1
        mock_run.assert_called_once_with(
            ["sudo", "-n", "amd-smi", "set", "-l", "STABLE_PEAK"],
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        )

    def test_returns_false_when_stable_peak_reports_failure_with_zero_exit(self):
        failed = MagicMock(
            returncode=0,
            stdout="ERROR: GPU[0]\t: Unable to set performance level\n",
            stderr="",
        )

        with patch(f"{_MODULE}.subprocess.run", return_value=failed) as mock_run:
            result = lock_clocks()

        assert result is False
        mock_run.assert_called_once_with(
            ["sudo", "-n", "amd-smi", "set", "-l", "STABLE_PEAK"],
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        )

    def test_returns_false_when_amd_smi_fails(self):
        import subprocess as sp

        with patch(
            f"{_MODULE}.subprocess.run",
            side_effect=sp.CalledProcessError(1, "amd-smi"),
        ) as mock_run:
            result = lock_clocks()

        assert result is False
        assert mock_run.call_count == 1

    def test_returns_false_and_unlocks_when_verification_fails(self):
        p_sleep = patch(f"{_MODULE}.time.sleep")
        p_verify = patch(f"{_MODULE}.verify_clocks", return_value=False)
        p_unlock = patch(f"{_MODULE}.unlock_clocks", return_value=True)
        with patch(f"{_MODULE}.subprocess.run"), p_verify, p_sleep, p_unlock as unlock:
            result = lock_clocks()

        assert result is False
        unlock.assert_called_once_with()

    def test_sleeps_before_verification(self):
        from sol_execbench.core.bench.clock_lock import VERIFY_DELAY_S

        p_verify = patch(f"{_MODULE}.verify_clocks", return_value=True)
        p_sleep = patch(f"{_MODULE}.time.sleep")
        with patch(f"{_MODULE}.subprocess.run"), p_verify, p_sleep as mock_sleep:
            lock_clocks()

        mock_sleep.assert_called_once_with(VERIFY_DELAY_S)


class TestVerifyClocks:
    def _make_smi_result(self, stdout: str, returncode: int = 0):
        return MagicMock(returncode=returncode, stdout=stdout, stderr="")

    def test_stable_peak_detected(self):
        result = self._make_smi_result(
            '{"gpu_data": [{"gpu": 0, "perf_level": '
            '"AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK"}]}'
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
            '{"gpu_data": [{"gpu": 0, "perf_level": "AMDSMI_DEV_PERF_LEVEL_AUTO"}]}'
        )
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks() is False

    def test_mixed_gpu_levels_are_not_locked(self):
        result = self._make_smi_result(
            '{"gpu_data": ['
            '{"gpu": 0, "perf_level": "AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK"},'
            '{"gpu": 1, "perf_level": "AMDSMI_DEV_PERF_LEVEL_AUTO"}]}'
        )
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks() is False

    def test_all_gpu_levels_must_be_stable_peak(self):
        result = self._make_smi_result(
            '{"gpu_data": ['
            '{"gpu": 0, "perf_level": "AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK"},'
            '{"gpu": 1, "perf_level": "AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK"}]}'
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
            f"{_MODULE}.subprocess.run", side_effect=[reset, verify]
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
        with patch(f"{_MODULE}.subprocess.run", side_effect=Exception("no sudo")):
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
