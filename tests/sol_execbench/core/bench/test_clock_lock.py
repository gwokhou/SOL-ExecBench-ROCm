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
    def _which(tool):
        return {"rocm-smi": "rocm-smi", "amd-smi": "amd-smi"}.get(tool)

    monkeypatch.setattr(clock_lock_module.shutil, "which", _which)


class TestProbeClockLockAvailable:
    def test_returns_true_when_amd_smi_version_succeeds(self):
        probe_result = MagicMock(returncode=0)
        with patch(f"{_MODULE}.subprocess.run", return_value=probe_result) as mock_run:
            result = probe_clock_lock_available()

        assert result is True
        mock_run.assert_called_once_with(
            ["sudo", "-n", "amd-smi", "version"], capture_output=True
        )

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
        with patch(f"{_MODULE}.subprocess.run") as mock_run, p_verify, p_sleep:
            result = lock_clocks()

        assert result is False
        # 1 lock call + 1 unlock call (amd-smi AUTO)
        assert mock_run.call_count == 2

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
        result = self._make_smi_result("GPU[0]\t\t: Performance Level: stable_peak\n")
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks() is True

    def test_auto_detected_as_not_locked(self):
        result = self._make_smi_result("GPU[0]\t\t: Performance Level: auto\n")
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks() is False

    def test_high_detected_as_not_locked(self):
        result = self._make_smi_result("GPU[0]\t\t: Performance Level: high\n")
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks() is False

    def test_case_insensitive_match(self):
        result = self._make_smi_result("GPU[0]\t\t: Performance Level: STABLE_PEAK\n")
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks() is True

    def test_rocm_smi_not_found(self):
        with patch(f"{_MODULE}.subprocess.run", side_effect=FileNotFoundError):
            assert verify_clocks() is False

    def test_rocm_smi_nonzero_exit(self):
        result = self._make_smi_result("", returncode=1)
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks() is False

    def test_empty_output(self):
        result = self._make_smi_result("")
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks() is False


class TestUnlockClocks:
    def test_unlocks_with_amd_smi(self):
        with patch(f"{_MODULE}.subprocess.run") as mock_run:
            unlock_clocks()

        mock_run.assert_called_once_with(
            ["sudo", "-n", "amd-smi", "set", "-l", "AUTO"],
            capture_output=True,
        )

    def test_does_not_raise_on_failure(self):
        with patch(f"{_MODULE}.subprocess.run", side_effect=Exception("no sudo")):
            unlock_clocks()


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
