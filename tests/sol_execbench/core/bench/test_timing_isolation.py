# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for timing isolation audit module."""

from __future__ import annotations

import json
import logging
import os
import subprocess


from sol_execbench.core.bench.timing_isolation import (
    _detect_gpu_count,
    clear_gpu_cache_between_subprocesses,
    collect_timing_environment_snapshot,
    detect_concurrent_gpu_processes,
    validate_gpu_device_isolation,
    verify_clock_state_with_warning,
)
from sol_execbench.core.platform.environment import EnvironmentEvidenceStatus


class TestDetectConcurrentGpuProcesses:
    """Test concurrent GPU process detection."""

    def test_no_processes_running(self, caplog):
        """Test that empty list is returned when no GPU processes are running."""
        import unittest.mock

        with unittest.mock.patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                ["amd-smi", "process", "--json"],
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "gpu": 0,
                            "process_list": [
                                {"process_info": "No running processes detected"}
                            ],
                        }
                    ]
                ),
                stderr="",
            ),
        ):
            result = detect_concurrent_gpu_processes()
            assert result == []

    def test_processes_detected(self, caplog):
        """Test that process list is returned when GPU processes are present."""
        import unittest.mock

        mock_output = json.dumps(
            [
                {
                    "gpu": 0,
                    "process_list": [
                        {"pid": 12345, "name": "python"},
                        {"pid": 67890, "name": "rocprofv3"},
                    ],
                }
            ]
        )
        with unittest.mock.patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                ["amd-smi", "process", "--json"],
                returncode=0,
                stdout=mock_output,
                stderr="",
            ),
        ):
            result = detect_concurrent_gpu_processes()
            assert len(result) == 2
            assert any(p.get("pid") == 12345 for p in result)
            assert any(p.get("pid") == 67890 for p in result)

    def test_timeout_handling(self, caplog):
        """Test that empty list is returned on subprocess timeout."""
        import unittest.mock

        with unittest.mock.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(["amd-smi", "process"], 5),
        ):
            result = detect_concurrent_gpu_processes()
            assert result == []
            # Check for timeout in log (case-insensitive)
            assert (
                "timeout" in caplog.text.lower() or "timed out" in caplog.text.lower()
            )

    def test_file_not_found_handling(self, caplog):
        """Test that empty list is returned when amd-smi is not found."""
        import unittest.mock

        with unittest.mock.patch(
            "subprocess.run",
            side_effect=FileNotFoundError("amd-smi not found"),
        ):
            result = detect_concurrent_gpu_processes()
            assert result == []
            assert "not found" in caplog.text.lower()


class TestVerifyClockStateWithWarning:
    """Test clock state verification with context-aware logging."""

    def test_clocks_locked_stable_peak(self, caplog):
        """Test that True is returned when clocks are in STABLE_PEAK mode."""
        import unittest.mock

        caplog.set_level(logging.INFO)

        with unittest.mock.patch(
            "sol_execbench.core.bench.clock_lock.verify_clocks",
            return_value=True,
        ):
            result = verify_clock_state_with_warning(context="test_context")
            assert result is True
            assert "STABLE_PEAK mode confirmed" in caplog.text
            assert "test_context" in caplog.text

    def test_clocks_not_locked(self, caplog):
        """Test that False is returned and warning is logged when clocks not locked."""
        import unittest.mock
        import logging

        caplog.set_level(logging.WARNING)

        with unittest.mock.patch(
            "sol_execbench.core.bench.clock_lock.verify_clocks",
            return_value=False,
        ):
            result = verify_clock_state_with_warning(context="batch_start")
            assert result is False
            assert "not in STABLE_PEAK mode" in caplog.text
            assert "Timing measurements may be unstable" in caplog.text

    def test_context_parameter_in_log(self, caplog):
        """Test that context parameter appears in log messages."""
        import unittest.mock
        import logging

        caplog.set_level(logging.INFO)

        with unittest.mock.patch(
            "sol_execbench.core.bench.clock_lock.verify_clocks",
            return_value=True,
        ):
            verify_clock_state_with_warning(context="problem_42")
            assert "problem_42" in caplog.text


class TestClearGpuCacheBetweenSubprocesses:
    """Test GPU cache clearing at subprocess boundaries."""

    def test_clears_cache_when_torch_available(self, caplog):
        """Test that torch.cuda.empty_cache() is called when torch is available."""
        import unittest.mock
        import logging

        caplog.set_level(logging.DEBUG)

        mock_torch = unittest.mock.MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with unittest.mock.patch.dict("sys.modules", {"torch": mock_torch}):
            clear_gpu_cache_between_subprocesses()
            mock_torch.cuda.empty_cache.assert_called_once()
            assert "cache cleared" in caplog.text.lower()

    def test_graceful_when_torch_unavailable(self, caplog):
        """Test that function handles torch unavailable gracefully."""
        import unittest.mock

        caplog.set_level(logging.DEBUG)
        with unittest.mock.patch.dict("sys.modules", {"torch": None}):
            clear_gpu_cache_between_subprocesses()
            # Should not raise exception
            assert (
                "cache cleared" in caplog.text.lower() or "torch" in caplog.text.lower()
            )


class TestCollectTimingEnvironmentSnapshot:
    """Test environment snapshot collection for timing audits."""

    def test_snapshot_structure(self):
        """Test that snapshot dict has required keys."""
        import unittest.mock

        with unittest.mock.patch(
            "sol_execbench.core.bench.timing_isolation.detect_concurrent_gpu_processes",
            return_value=[],
        ):
            with unittest.mock.patch(
                "sol_execbench.core.bench.clock_lock.are_clocks_locked",
                return_value=False,
            ):
                with unittest.mock.patch(
                    "sol_execbench.core.platform.environment.collect_environment_snapshot",
                    return_value=self._mock_snapshot(),
                ):
                    snapshot = collect_timing_environment_snapshot()

                    assert "schema_version" in snapshot
                    assert "generated_at" in snapshot
                    assert "gpu_processes" in snapshot
                    assert "clocks_locked" in snapshot
                    assert "tools_available" in snapshot
                    assert "warnings" in snapshot

    def test_schema_version(self):
        """Test that schema version is set correctly."""
        import unittest.mock

        with unittest.mock.patch(
            "sol_execbench.core.bench.timing_isolation.detect_concurrent_gpu_processes",
            return_value=[],
        ):
            with unittest.mock.patch(
                "sol_execbench.core.bench.clock_lock.are_clocks_locked",
                return_value=False,
            ):
                with unittest.mock.patch(
                    "sol_execbench.core.platform.environment.collect_environment_snapshot",
                    return_value=self._mock_snapshot(),
                ):
                    snapshot = collect_timing_environment_snapshot()
                    assert (
                        snapshot["schema_version"]
                        == "sol_execbench.timing_isolation_snapshot.v1"
                    )

    def _mock_snapshot(self):
        """Helper to create a mock environment snapshot."""
        from sol_execbench.core.platform.environment import EnvironmentSnapshot

        return EnvironmentSnapshot(
            generated_at="2026-06-10T00:00:00Z",
            collection_status=EnvironmentEvidenceStatus.AVAILABLE,
        )


class TestIntegrationPreflightAudit:
    """Integration test for pre-flight audit flow."""

    def test_preflight_audit_flow(self, caplog):
        """Test complete pre-flight audit: detection → verification → snapshot."""
        import unittest.mock
        import logging
        from sol_execbench.core.bench import timing_isolation

        caplog.set_level(logging.INFO)

        with unittest.mock.patch.object(
            timing_isolation,
            "detect_concurrent_gpu_processes",
            return_value=[{"pid": 12345, "device": "0000:01:00.1", "name": "python"}],
        ):
            with unittest.mock.patch(
                "sol_execbench.core.bench.clock_lock.verify_clocks",
                return_value=True,
            ):
                with unittest.mock.patch(
                    "sol_execbench.core.bench.clock_lock.are_clocks_locked",
                    return_value=True,
                ):
                    with unittest.mock.patch(
                        "sol_execbench.core.platform.environment.collect_environment_snapshot",
                        return_value=self._mock_snapshot(),
                    ):
                        # Step 1: Detect concurrent processes
                        processes = timing_isolation.detect_concurrent_gpu_processes()
                        assert len(processes) == 1

                        # Step 2: Verify clock state
                        clocks_ok = timing_isolation.verify_clock_state_with_warning(
                            context="batch_start"
                        )
                        assert clocks_ok is True

                        # Step 3: Collect snapshot
                        snapshot = (
                            timing_isolation.collect_timing_environment_snapshot()
                        )
                        assert (
                            snapshot["schema_version"]
                            == "sol_execbench.timing_isolation_snapshot.v1"
                        )

                        # Verify audit flow completed
                        assert "concurrent" in caplog.text.lower() or len(processes) > 0
                        assert "clock" in caplog.text.lower()

    def _mock_snapshot(self):
        """Helper to create a mock environment snapshot."""
        from sol_execbench.core.platform.environment import EnvironmentSnapshot

        return EnvironmentSnapshot(
            generated_at="2026-06-10T00:00:00Z",
            collection_status=EnvironmentEvidenceStatus.AVAILABLE,
        )


class TestValidateGpuDeviceIsolation:
    """Test GPU device isolation validation."""

    def test_detect_gpu_count_counts_unique_amd_smi_gpu_ids(self):
        """amd-smi list --json reports one object per device."""
        import unittest.mock

        mock_output = json.dumps(
            [
                {"gpu": 0, "bdf": "0000:01:00.0"},
                {"gpu": 1, "bdf": "0000:02:00.0"},
            ]
        )
        with unittest.mock.patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                ["amd-smi", "list", "--json"],
                returncode=0,
                stdout=mock_output,
                stderr="",
            ),
        ):
            assert _detect_gpu_count() == 2

    def test_single_gpu_is_isolated(self):
        """Single GPU system is always isolated regardless of ROCR_VISIBLE_DEVICES."""
        import unittest.mock

        with unittest.mock.patch(
            "sol_execbench.core.bench.timing_isolation._detect_gpu_count",
            return_value=1,
        ):
            with unittest.mock.patch.dict("os.environ", {}, clear=False):
                result = validate_gpu_device_isolation()
                assert result["isolated"] is True
                assert result["gpu_count"] == 1

    def test_multi_gpu_with_restriction_is_isolated(self):
        """Multi-GPU with ROCR_VISIBLE_DEVICES set is isolated."""
        import unittest.mock

        with unittest.mock.patch(
            "sol_execbench.core.bench.timing_isolation._detect_gpu_count",
            return_value=4,
        ):
            with unittest.mock.patch.dict(
                "os.environ", {"ROCR_VISIBLE_DEVICES": "0"}, clear=False
            ):
                result = validate_gpu_device_isolation()
                assert result["isolated"] is True
                assert result["gpu_count"] == 4
                assert result["rocr_visible_devices"] == "0"

    def test_multi_gpu_without_restriction_not_isolated(self):
        """Multi-GPU without ROCR_VISIBLE_DEVICES is not isolated."""
        import unittest.mock

        with unittest.mock.patch(
            "sol_execbench.core.bench.timing_isolation._detect_gpu_count",
            return_value=4,
        ):
            with unittest.mock.patch.dict("os.environ", {}, clear=False):
                # Ensure ROCR_VISIBLE_DEVICES is not set
                os.environ.pop("ROCR_VISIBLE_DEVICES", None)
                result = validate_gpu_device_isolation()
                assert result["isolated"] is False
                assert result["gpu_count"] == 4
                assert any("multi_gpu_no_restriction" in w for w in result["warnings"])

    def test_gpu_device_sets_env_var(self):
        """Providing gpu_device sets ROCR_VISIBLE_DEVICES."""
        import unittest.mock

        with unittest.mock.patch(
            "sol_execbench.core.bench.timing_isolation._detect_gpu_count",
            return_value=2,
        ):
            with unittest.mock.patch.dict("os.environ", {}, clear=False):
                result = validate_gpu_device_isolation(gpu_device=1)
                assert result["isolated"] is True
                assert result["gpu_device_set"] is True
                assert os.environ.get("ROCR_VISIBLE_DEVICES") == "1"

    def test_gpu_count_unknown(self):
        """Unknown GPU count (amd-smi unavailable) generates warning."""
        import unittest.mock

        with unittest.mock.patch(
            "sol_execbench.core.bench.timing_isolation._detect_gpu_count",
            return_value=0,
        ):
            result = validate_gpu_device_isolation()
            assert result["gpu_count"] == 0
            assert any("gpu_count_unknown" in w for w in result["warnings"])
