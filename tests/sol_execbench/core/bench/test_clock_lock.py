# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for sol_execbench.core.bench.clock_lock."""

from unittest.mock import MagicMock, call, patch

import pytest

from sol_execbench.core.bench import clock_lock as clock_lock_module
from sol_execbench.core.bench.clock_lock import (
    are_clocks_locked,
    lock_clocks,
    probe_clock_lock_available,
    unlock_clocks,
    verify_clocks,
)
from sol_execbench.core.bench.config.device_config import (
    CLOCK_LOCK_PRESETS,
    ClockPreset,
    get_clock_preset,
)

_MODULE = "sol_execbench.core.bench.clock_lock"


@pytest.fixture(autouse=True)
def _mock_rocm_smi_path(monkeypatch):
    monkeypatch.setattr(clock_lock_module.shutil, "which", lambda _tool: "rocm-smi")


class TestProbeClockLockAvailable:
    def test_returns_true_when_sudo_rocm_smi_succeeds(self):
        probe_result = MagicMock(returncode=0)
        with patch(f"{_MODULE}.subprocess.run", return_value=probe_result) as mock_run:
            result = probe_clock_lock_available()

        assert result is True
        mock_run.assert_called_once_with(
            ["sudo", "-n", "rocm-smi", "--showclocks"], capture_output=True
        )

    def test_returns_false_when_sudo_fails(self):
        probe_result = MagicMock(returncode=1)
        with patch(f"{_MODULE}.subprocess.run", return_value=probe_result):
            assert probe_clock_lock_available() is False

    def test_returns_false_when_rocm_smi_not_found(self):
        with patch(f"{_MODULE}.subprocess.run", side_effect=FileNotFoundError):
            assert probe_clock_lock_available() is False

    def test_uses_resolved_rocm_smi_path(self, monkeypatch):
        monkeypatch.setattr(
            clock_lock_module.shutil,
            "which",
            lambda _tool: "/opt/rocm/bin/rocm-smi",
        )
        probe_result = MagicMock(returncode=0)
        with patch(f"{_MODULE}.subprocess.run", return_value=probe_result) as mock_run:
            assert probe_clock_lock_available() is True

        mock_run.assert_called_once_with(
            ["sudo", "-n", "/opt/rocm/bin/rocm-smi", "--showclocks"],
            capture_output=True,
        )


class TestLockClocks:
    @staticmethod
    @pytest.fixture(autouse=True)
    def _clean_env(monkeypatch):
        monkeypatch.delenv("SOL_EXECBENCH_SCLK_LEVEL", raising=False)
        monkeypatch.delenv("SOL_EXECBENCH_MCLK_LEVEL", raising=False)

    def _patch_verify_and_sleep(self):
        return (
            patch(f"{_MODULE}.verify_clocks", return_value=True),
            patch(f"{_MODULE}.time.sleep"),
        )

    def test_locks_sclk_and_mclk_for_known_device(self):
        p_verify, p_sleep = self._patch_verify_and_sleep()
        with patch(f"{_MODULE}.subprocess.run") as mock_run, p_verify, p_sleep:
            result = lock_clocks("AMD Radeon gfx1200")

        assert result is True
        assert mock_run.call_args_list[:3] == [
            call(
                ["sudo", "-n", "rocm-smi", "--setperflevel", "manual"],
                check=True,
                capture_output=True,
            ),
            call(
                ["sudo", "-n", "rocm-smi", "--setsclk", "1"],
                check=True,
                capture_output=True,
            ),
            call(
                ["sudo", "-n", "rocm-smi", "--setmclk", "1"],
                check=True,
                capture_output=True,
            ),
        ]

    def test_returns_false_for_unknown_device(self):
        p_verify, p_sleep = self._patch_verify_and_sleep()
        with patch(f"{_MODULE}.subprocess.run") as mock_run, p_verify, p_sleep:
            result = lock_clocks("Unknown GPU")

        assert result is False
        mock_run.assert_not_called()

    def test_env_var_overrides_preset(self, monkeypatch):
        monkeypatch.setenv("SOL_EXECBENCH_SCLK_LEVEL", "2")
        monkeypatch.setenv("SOL_EXECBENCH_MCLK_LEVEL", "3")

        p_verify, p_sleep = self._patch_verify_and_sleep()
        with patch(f"{_MODULE}.subprocess.run") as mock_run, p_verify, p_sleep:
            result = lock_clocks("Unknown GPU")

        assert result is True
        assert mock_run.call_args_list[1] == call(
            ["sudo", "-n", "rocm-smi", "--setsclk", "2"],
            check=True,
            capture_output=True,
        )
        assert mock_run.call_args_list[2] == call(
            ["sudo", "-n", "rocm-smi", "--setmclk", "3"],
            check=True,
            capture_output=True,
        )

    def test_returns_false_when_sclk_lock_fails(self):
        import subprocess as sp

        with patch(
            f"{_MODULE}.subprocess.run",
            side_effect=sp.CalledProcessError(1, "rocm-smi"),
        ) as mock_run:
            result = lock_clocks("AMD Radeon gfx1200")

        assert result is False
        assert mock_run.call_count == 1

    def test_returns_false_and_unlocks_when_mclk_fails(self):
        import subprocess as sp

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 3:
                raise sp.CalledProcessError(1, "rocm-smi")
            return MagicMock()

        with patch(f"{_MODULE}.subprocess.run", side_effect=side_effect) as mock_run:
            result = lock_clocks("AMD Radeon gfx1200")

        assert result is False
        assert mock_run.call_count == 5

    def test_returns_false_when_verification_fails(self):
        p_sleep = patch(f"{_MODULE}.time.sleep")
        p_verify = patch(f"{_MODULE}.verify_clocks", return_value=False)
        with patch(f"{_MODULE}.subprocess.run") as mock_run, p_verify, p_sleep:
            result = lock_clocks("AMD Radeon gfx1200")

        assert result is False
        assert mock_run.call_count == 5

    def test_sleeps_before_verification(self):
        from sol_execbench.core.bench.clock_lock import VERIFY_DELAY_S

        p_verify = patch(f"{_MODULE}.verify_clocks", return_value=True)
        p_sleep = patch(f"{_MODULE}.time.sleep")
        with patch(f"{_MODULE}.subprocess.run"), p_verify, p_sleep as mock_sleep:
            lock_clocks("AMD Radeon gfx1200")

        mock_sleep.assert_called_once_with(VERIFY_DELAY_S)


class TestVerifyClocks:
    def _make_smi_result(self, stdout: str, returncode: int = 0):
        return MagicMock(returncode=returncode, stdout=stdout, stderr="")

    def test_active_sclk_and_mclk_levels_match(self):
        result = self._make_smi_result(
            "sclk clock level:\n0: 500Mhz\n1: 2500Mhz *\n"
            "mclk clock level:\n0: 96Mhz\n1: 1200Mhz *\n"
        )
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks(1, 1) is True

    def test_same_line_clock_levels_match(self):
        result = self._make_smi_result(
            "GPU[0]\t\t: sclk clock level: 1: (0Mhz)\n"
            "GPU[0]\t\t: mclk clock level: 0: (96Mhz)\n"
        )
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks(1, 0) is True

    def test_sclk_mismatch_fails(self):
        result = self._make_smi_result(
            "sclk clock level:\n0: 500Mhz *\n1: 2500Mhz\n"
            "mclk clock level:\n0: 96Mhz\n1: 1200Mhz *\n"
        )
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks(1, 1) is False

    def test_mclk_mismatch_fails(self):
        result = self._make_smi_result(
            "sclk clock level:\n0: 500Mhz\n1: 2500Mhz *\n"
            "mclk clock level:\n0: 96Mhz *\n1: 1200Mhz\n"
        )
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks(1, 1) is False

    def test_supported_mclk_level_passes_when_low_power_keeps_active_level_low(self):
        showclocks = self._make_smi_result(
            "WARNING: AMD GPU device(s) is/are in a low-power state.\n"
            "GPU[0]\t\t: sclk clock level: 1: (0Mhz)\n"
            "GPU[0]\t\t: mclk clock level: 0: (96Mhz)\n"
        )
        supported = self._make_smi_result(
            "GPU[0]\t\t: Supported mclk frequencies on GPU0\n"
            "GPU[0]\t\t: 0: 96Mhz *\n"
            "GPU[0]\t\t: 1: 456Mhz\n"
            "GPU[0]\t\t: 2: 772Mhz\n"
        )
        with patch(
            f"{_MODULE}.subprocess.run", side_effect=[showclocks, supported]
        ) as mock_run:
            assert verify_clocks(1, 1) is True

        assert mock_run.call_args_list == [
            call(["rocm-smi", "--showclocks"], capture_output=True, text=True),
            call(["rocm-smi", "-s"], capture_output=True, text=True),
        ]

    def test_low_power_mclk_fallback_rejects_unsupported_level(self):
        showclocks = self._make_smi_result(
            "WARNING: AMD GPU device(s) is/are in a low-power state.\n"
            "GPU[0]\t\t: sclk clock level: 1: (0Mhz)\n"
            "GPU[0]\t\t: mclk clock level: 0: (96Mhz)\n"
        )
        supported = self._make_smi_result(
            "GPU[0]\t\t: Supported mclk frequencies on GPU0\n"
            "GPU[0]\t\t: 0: 96Mhz *\n"
            "GPU[0]\t\t: 1: 456Mhz\n"
        )
        with patch(f"{_MODULE}.subprocess.run", side_effect=[showclocks, supported]):
            assert verify_clocks(1, 5) is False

    def test_rocm_smi_not_found(self):
        with patch(f"{_MODULE}.subprocess.run", side_effect=FileNotFoundError):
            assert verify_clocks(1, 1) is False

    def test_rocm_smi_nonzero_exit(self):
        result = self._make_smi_result("", returncode=1)
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks(1, 1) is False

    def test_empty_output(self):
        result = self._make_smi_result("")
        with patch(f"{_MODULE}.subprocess.run", return_value=result):
            assert verify_clocks(1, 1) is False


class TestUnlockClocks:
    def test_resets_clocks_and_perf_level(self):
        with patch(f"{_MODULE}.subprocess.run") as mock_run:
            unlock_clocks()

        assert mock_run.call_args_list == [
            call(["sudo", "-n", "rocm-smi", "--resetclocks"], capture_output=True),
            call(
                ["sudo", "-n", "rocm-smi", "--setperflevel", "auto"],
                capture_output=True,
            ),
        ]

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


class TestGetClockPreset:
    def test_gfx1200(self):
        preset = get_clock_preset("AMD Radeon gfx1200")
        assert preset == ClockPreset(sclk_level=1, mclk_level=1)

    def test_instinct(self):
        preset = get_clock_preset("AMD Instinct MI300X")
        assert preset == ClockPreset(sclk_level=1, mclk_level=1)

    def test_unknown_device_returns_none(self):
        assert get_clock_preset("NVIDIA H100") is None

    def test_empty_string_returns_none(self):
        assert get_clock_preset("") is None

    def test_presets_cover_all_known_devices(self):
        for name, preset in CLOCK_LOCK_PRESETS.items():
            assert get_clock_preset(name) == preset
