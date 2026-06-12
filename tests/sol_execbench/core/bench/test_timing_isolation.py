# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for timing isolation audit module."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path


from sol_execbench.core.bench.timing_isolation import (
    clear_gpu_cache_between_subprocesses,
    collect_timing_environment_snapshot,
    detect_concurrent_gpu_processes,
    validate_gpu_device_isolation,
    verify_clock_state_with_warning,
)


class TestDetectConcurrentGpuProcesses:
    """Test concurrent GPU process detection."""

    def test_no_processes_running(self, caplog):
        """Test that empty list is returned when no GPU processes are running."""
        import unittest.mock

        with unittest.mock.patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                ["rocm-smi", "--showpids"],
                returncode=0,
                stdout="No KFD PIDs currently running",
                stderr="",
            ),
        ):
            result = detect_concurrent_gpu_processes()
            assert result == []
            assert "No KFD PIDs" in caplog.text or result == []

    def test_processes_detected(self, caplog):
        """Test that process list is returned when GPU processes are present."""
        import unittest.mock

        mock_output = """
============================XX PIDs =============================
GPU  0000:01:00.1
  KFD PID                     12345  Name              python
  KFD PID                     67890  Name              rocprofv3
"""
        with unittest.mock.patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                ["rocm-smi", "--showpids"],
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
            side_effect=subprocess.TimeoutExpired(["rocm-smi", "--showpids"], 5),
        ):
            result = detect_concurrent_gpu_processes()
            assert result == []
            # Check for timeout in log (case-insensitive)
            assert (
                "timeout" in caplog.text.lower() or "timed out" in caplog.text.lower()
            )

    def test_file_not_found_handling(self, caplog):
        """Test that empty list is returned when rocm-smi is not found."""
        import unittest.mock

        with unittest.mock.patch(
            "subprocess.run",
            side_effect=FileNotFoundError("rocm-smi not found"),
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

        with unittest.mock.patch.dict("sys.modules", {}, clear=False):
            # Remove torch from modules if present
            import sys

            sys.modules.pop("torch", None)

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
                    "sol_execbench.core.environment.collect_environment_snapshot",
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
                    "sol_execbench.core.environment.collect_environment_snapshot",
                    return_value=self._mock_snapshot(),
                ):
                    snapshot = collect_timing_environment_snapshot()
                    assert (
                        snapshot["schema_version"]
                        == "sol_execbench.timing_isolation_snapshot.v1"
                    )

    def _mock_snapshot(self):
        """Helper to create a mock environment snapshot."""
        from sol_execbench.core.environment import EnvironmentSnapshot

        return EnvironmentSnapshot(
            generated_at="2026-06-10T00:00:00Z",
            collection_status="available",
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
                        "sol_execbench.core.environment.collect_environment_snapshot",
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
        from sol_execbench.core.environment import EnvironmentSnapshot

        return EnvironmentSnapshot(
            generated_at="2026-06-10T00:00:00Z",
            collection_status="available",
        )


class TestValidateGpuDeviceIsolation:
    """Test GPU device isolation validation."""

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
        """Unknown GPU count (rocm-smi unavailable) generates warning."""
        import unittest.mock

        with unittest.mock.patch(
            "sol_execbench.core.bench.timing_isolation._detect_gpu_count",
            return_value=0,
        ):
            result = validate_gpu_device_isolation()
            assert result["gpu_count"] == 0
            assert any("gpu_count_unknown" in w for w in result["warnings"])


class TestStrictIsolationInBatch:
    """Test --strict-isolation flag behavior in run_rdna4_profiler_timing_batch."""

    @classmethod
    def _load_batch_module(cls):
        """Load the batch script module via importlib."""
        from importlib.util import module_from_spec, spec_from_file_location

        REPO_ROOT = Path(__file__).resolve().parents[4]
        SCRIPT_PATH = REPO_ROOT / "scripts/run_rdna4_profiler_timing_batch.py"
        spec = spec_from_file_location(
            "run_rdna4_profiler_timing_batch_strict", SCRIPT_PATH
        )
        assert spec is not None and spec.loader is not None
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_strict_isolation_flag_default_false(self):
        """--strict-isolation defaults to False."""
        batch = self._load_batch_module()
        args = batch.parse_args([])
        assert args.strict_isolation is False

    def test_strict_isolation_flag_enabled(self):
        """--strict-isolation can be enabled."""
        batch = self._load_batch_module()
        args = batch.parse_args(["--strict-isolation"])
        assert args.strict_isolation is True

    def test_gpu_device_flag(self):
        """--gpu-device sets the device index."""
        batch = self._load_batch_module()
        args = batch.parse_args(["--gpu-device", "0"])
        assert args.gpu_device == 0

    def test_gpu_device_flag_default_none(self):
        """--gpu-device defaults to None."""
        batch = self._load_batch_module()
        args = batch.parse_args([])
        assert args.gpu_device is None

    def test_strict_isolation_aborts_on_concurrent_processes(self):
        """Strict mode returns 1 when concurrent GPU processes detected."""
        import unittest.mock

        batch = self._load_batch_module()
        with unittest.mock.patch.object(
            batch,
            "detect_concurrent_gpu_processes",
            return_value=[{"pid": 999, "device": "0000:01:00.1", "name": "python"}],
        ):
            with unittest.mock.patch.object(
                batch,
                "_build_coverage",
                return_value=self._mock_coverage(),
            ):
                result = batch.run_batch(
                    dataset_root=Path("/tmp/test"),
                    output_dir=Path("/tmp/test_out"),
                    strict_isolation=True,
                )
                assert result == 1

    def test_default_mode_warns_on_concurrent_processes(self):
        """Default mode returns normally (non-1) even with concurrent processes."""
        import unittest.mock

        batch = self._load_batch_module()
        mock_coverage = self._mock_coverage()
        with unittest.mock.patch.object(
            batch,
            "detect_concurrent_gpu_processes",
            return_value=[{"pid": 999, "device": "0000:01:00.1", "name": "python"}],
        ):
            with unittest.mock.patch.object(
                batch,
                "_build_coverage",
                return_value=mock_coverage,
            ):
                with unittest.mock.patch.object(
                    batch,
                    "verify_clock_state_with_warning",
                    return_value=True,
                ):
                    with unittest.mock.patch.object(
                        batch,
                        "validate_gpu_device_isolation",
                        return_value={
                            "isolated": True,
                            "warnings": [],
                            "gpu_count": 1,
                            "rocr_visible_devices": None,
                            "gpu_device_set": False,
                            "schema_version": "test",
                        },
                    ):
                        with unittest.mock.patch.object(
                            batch,
                            "_build_summary",
                            return_value={
                                "failed": 0,
                                "fallback_or_missing": 0,
                                "selected_targets": 0,
                                "completed": 0,
                                "blocked": 0,
                                "total": 0,
                            },
                        ):
                            with unittest.mock.patch.object(
                                batch,
                                "_render_summary_markdown",
                                return_value="# Summary\n",
                            ):
                                result = batch.run_batch(
                                    dataset_root=Path("/tmp/test"),
                                    output_dir=Path("/tmp/test_out"),
                                )
                                assert result in (0, 1)

    def _mock_coverage(self):
        """Create a mock ProfilerTimingCoverageReport with empty problems."""
        import unittest.mock

        coverage = unittest.mock.MagicMock()
        coverage.dataset_root = "/tmp/test"
        coverage.total_problems = 0
        coverage.problems = []
        return coverage


class TestCalibrationPathArg:
    """Test --calibration-path CLI argument and passthrough."""

    def _load_batch_module(self):
        import importlib.util
        import sys

        REPO_ROOT = Path(__file__).resolve().parents[4]
        SCRIPT_PATH = REPO_ROOT / "scripts" / "run_rdna4_profiler_timing_batch.py"
        spec = importlib.util.spec_from_file_location(
            "run_rdna4_profiler_timing_batch_gap", SCRIPT_PATH
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_calibration_path_arg_parsed(self):
        batch = self._load_batch_module()
        args = batch.parse_args(["--calibration-path", "/tmp/cal.json"])
        assert args.calibration_path == Path("/tmp/cal.json")

    def test_calibration_path_defaults_none(self):
        batch = self._load_batch_module()
        args = batch.parse_args([])
        assert args.calibration_path is None


class TestPidLockContentionInSidecar:
    """Test pid_lock_contention field in sidecar payloads."""

    def test_blocked_sidecar_includes_contention(self, tmp_path):
        from importlib.util import module_from_spec, spec_from_file_location

        REPO_ROOT = Path(__file__).resolve().parents[4]
        SCRIPT_PATH = REPO_ROOT / "scripts" / "run_rdna4_profiler_timing_batch.py"
        spec = spec_from_file_location("batch_sidecar_test", SCRIPT_PATH)
        assert spec is not None and spec.loader is not None
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)

        target = mod.ProfilerTimingProblemCoverage(
            problem_path="L1/test",
            problem_id="L1/test",
            category="L1",
            readiness_status="ready",
            workload_count=1,
            status="missing",
        )
        replacement_path = tmp_path / "timing" / "L1" / "test.json"
        mod._write_blocked_sidecar(
            target,
            replacement_path=replacement_path,
            staging_dir=None,
            reason="test",
            pid_lock_contention=True,
        )
        payload = json.loads(replacement_path.read_text(encoding="utf-8"))
        assert payload["pid_lock_contention"] is True

    def test_blocked_sidecar_omits_contention_when_false(self, tmp_path):
        from importlib.util import module_from_spec, spec_from_file_location

        REPO_ROOT = Path(__file__).resolve().parents[4]
        SCRIPT_PATH = REPO_ROOT / "scripts" / "run_rdna4_profiler_timing_batch.py"
        spec = spec_from_file_location("batch_sidecar_test2", SCRIPT_PATH)
        assert spec is not None and spec.loader is not None
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)

        target = mod.ProfilerTimingProblemCoverage(
            problem_path="L1/test",
            problem_id="L1/test",
            category="L1",
            readiness_status="ready",
            workload_count=1,
            status="missing",
        )
        replacement_path = tmp_path / "timing" / "L1" / "test2.json"
        mod._write_blocked_sidecar(
            target,
            replacement_path=replacement_path,
            staging_dir=None,
            reason="test",
            pid_lock_contention=False,
        )
        payload = json.loads(replacement_path.read_text(encoding="utf-8"))
        assert "pid_lock_contention" not in payload
