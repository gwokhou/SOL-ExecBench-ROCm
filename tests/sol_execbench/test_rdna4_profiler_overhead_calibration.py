# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for rocprofv3 overhead calibration and profiler_overhead_ms integration."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3TimingEvidence,
    build_timing_evidence,
    _read_overhead_calibration,
)
from sol_execbench.core.bench.timing_policy import (
    TimingActivityDomain,
    TimingBackend,
    TimingPolicy,
)


def _make_policy(**overrides) -> TimingPolicy:
    defaults = {
        "source_type": object(),
        "activity_domain": TimingActivityDomain.KERNEL_ACTIVITY,
        "backend": TimingBackend.ROCPROFV3,
        "aggregation_rule": "sum",
        "interpretation": "kernel-only",
        "fallback_applied": False,
        "reason": "",
    }
    defaults.update(overrides)
    return TimingPolicy(**defaults)


class TestProfilerOverheadInEvidence:
    """Test profiler_overhead_ms field in Rocprofv3TimingEvidence."""

    def test_overhead_none_by_default(self):
        evidence = Rocprofv3TimingEvidence(
            tool_version="rocprofv3",
            gpu_architecture="gfx1200",
            activity_domain=TimingActivityDomain.KERNEL_ACTIVITY,
            aggregation_rule="sum",
            backend=TimingBackend.ROCPROFV3,
            interpretation="kernel-only",
            parsed_rows=(),
        )
        assert evidence.profiler_overhead_ms is None

    def test_overhead_in_to_dict(self):
        evidence = Rocprofv3TimingEvidence(
            tool_version="rocprofv3",
            gpu_architecture="gfx1200",
            activity_domain=TimingActivityDomain.KERNEL_ACTIVITY,
            aggregation_rule="sum",
            backend=TimingBackend.ROCPROFV3,
            interpretation="kernel-only",
            parsed_rows=(),
            profiler_overhead_ms=0.021,
        )
        d = evidence.to_dict()
        assert d["profiler_overhead_ms"] == 0.021

    def test_overhead_none_in_to_dict(self):
        evidence = Rocprofv3TimingEvidence(
            tool_version="rocprofv3",
            gpu_architecture="gfx1200",
            activity_domain=TimingActivityDomain.KERNEL_ACTIVITY,
            aggregation_rule="sum",
            backend=TimingBackend.ROCPROFV3,
            interpretation="kernel-only",
            parsed_rows=(),
        )
        d = evidence.to_dict()
        assert d["profiler_overhead_ms"] is None


class TestBuildTimingEvidenceWithOverhead:
    """Test build_timing_evidence passes profiler_overhead_ms through."""

    def test_build_with_overhead(self):
        policy = _make_policy()
        csv = (
            "Name,Dispatch_ID,Duration_ns,Calls,Queue_ID,Signal\nkernel,0,1000,1,0,0\n"
        )
        evidence = build_timing_evidence(
            policy=policy,
            csv_content=csv,
            tool_version="rocprofv3",
            gpu_architecture="gfx1200",
            profiler_overhead_ms=0.015,
        )
        assert evidence.profiler_overhead_ms == 0.015

    def test_build_without_overhead(self):
        policy = _make_policy()
        csv = (
            "Name,Dispatch_ID,Duration_ns,Calls,Queue_ID,Signal\nkernel,0,1000,1,0,0\n"
        )
        evidence = build_timing_evidence(
            policy=policy,
            csv_content=csv,
            tool_version="rocprofv3",
            gpu_architecture="gfx1200",
        )
        assert evidence.profiler_overhead_ms is None


class TestReadOverheadCalibration:
    """Test _read_overhead_calibration function."""

    def test_reads_valid_calibration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_path = Path(tmpdir) / "cal.json"
            cal_path.write_text(
                json.dumps({"overhead_ms": 0.023}) + "\n",
                encoding="utf-8",
            )
            result = _read_overhead_calibration(cal_path)
            assert result == 0.023

    def test_returns_none_for_missing_file(self):
        result = _read_overhead_calibration(Path("/nonexistent/file.json"))
        assert result is None

    def test_returns_none_for_none_path(self):
        result = _read_overhead_calibration(None)
        assert result is None

    def test_returns_none_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_path = Path(tmpdir) / "cal.json"
            cal_path.write_text("not json", encoding="utf-8")
            result = _read_overhead_calibration(cal_path)
            assert result is None

    def test_returns_none_for_missing_overhead_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_path = Path(tmpdir) / "cal.json"
            cal_path.write_text(
                json.dumps({"baseline_median_ms": 0.05}) + "\n",
                encoding="utf-8",
            )
            result = _read_overhead_calibration(cal_path)
            assert result is None


class TestCalibrationScriptArgs:
    """Test argument parsing for the calibration script."""

    def _load_script(self):
        from importlib.util import module_from_spec, spec_from_file_location

        REPO_ROOT = Path(__file__).resolve().parents[2]
        SCRIPT_PATH = (
            REPO_ROOT
            / "scripts/internal/rdna4/run_rdna4_profiler_overhead_calibration.py"
        )
        spec = spec_from_file_location(
            "run_rdna4_profiler_overhead_calibration", SCRIPT_PATH
        )
        assert spec is not None and spec.loader is not None
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_default_strict_isolation_true(self):
        mod = self._load_script()
        args = mod.parse_args([])
        assert args.strict_isolation is True

    def test_no_strict_isolation_disables(self):
        mod = self._load_script()
        args = mod.parse_args(["--no-strict-isolation"])
        assert args.strict_isolation is False

    def test_gpu_device_flag(self):
        mod = self._load_script()
        args = mod.parse_args(["--gpu-device", "1"])
        assert args.gpu_device == 1

    def test_lock_clocks_enabled_by_default(self):
        mod = self._load_script()
        args = mod.parse_args([])
        assert args.lock_clocks is True

    def test_no_lock_clocks_disables_setup(self):
        mod = self._load_script()
        args = mod.parse_args(["--no-lock-clocks"])
        assert args.lock_clocks is False

    def test_reset_clocks_enabled_by_default(self):
        mod = self._load_script()
        args = mod.parse_args([])
        assert args.reset_clocks is True

    def test_iterations_flag(self):
        mod = self._load_script()
        args = mod.parse_args(["--iterations", "50"])
        assert args.iterations == 50


class TestCalibrationJsonSchema:
    """Test calibration JSON output schema validation."""

    def test_schema_version_constant(self):
        from importlib.util import module_from_spec, spec_from_file_location

        REPO_ROOT = Path(__file__).resolve().parents[2]
        SCRIPT_PATH = (
            REPO_ROOT
            / "scripts/internal/rdna4/run_rdna4_profiler_overhead_calibration.py"
        )
        spec = spec_from_file_location(
            "run_rdna4_profiler_overhead_calibration_schema", SCRIPT_PATH
        )
        assert spec is not None and spec.loader is not None
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert (
            mod.CALIBRATION_SCHEMA_VERSION
            == "sol_execbench.rocprofv3_overhead_calibration.v1"
        )


class TestCalibrationClockSetup:
    """Test calibration clock setup/teardown without touching real hardware."""

    def _load_script(self):
        from importlib.util import module_from_spec, spec_from_file_location

        REPO_ROOT = Path(__file__).resolve().parents[2]
        SCRIPT_PATH = (
            REPO_ROOT
            / "scripts/internal/rdna4/run_rdna4_profiler_overhead_calibration.py"
        )
        spec = spec_from_file_location(
            "run_rdna4_profiler_overhead_calibration_clock_setup", SCRIPT_PATH
        )
        assert spec is not None and spec.loader is not None
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_setup_locks_unlocked_gpu_and_teardown_resets(self, monkeypatch):
        mod = self._load_script()
        calls: list[str] = []

        monkeypatch.delenv("SOL_EXECBENCH_CLOCKS_LOCKED", raising=False)
        monkeypatch.setattr(mod, "verify_clocks", lambda: False)
        monkeypatch.setattr(mod, "lock_clocks", lambda: calls.append("lock") is None)
        monkeypatch.setattr(mod, "unlock_clocks", lambda: calls.append("unlock"))

        state = mod._setup_calibration_clocks(
            manage_clocks=True,
            strict_isolation=True,
        )
        assert state.clock_locked is True
        assert state.lock_acquired is True
        assert calls == ["lock"]

        mod._teardown_calibration_clocks(state, reset_clocks=True)
        assert calls == ["lock", "unlock"]

    def test_setup_preserves_external_clock_lock(self, monkeypatch):
        mod = self._load_script()
        calls: list[str] = []

        monkeypatch.setenv("SOL_EXECBENCH_CLOCKS_LOCKED", "external")
        monkeypatch.setattr(mod, "verify_clocks", lambda: True)
        monkeypatch.setattr(mod, "lock_clocks", lambda: calls.append("lock") is None)
        monkeypatch.setattr(mod, "unlock_clocks", lambda: calls.append("unlock"))

        state = mod._setup_calibration_clocks(
            manage_clocks=True,
            strict_isolation=True,
        )
        assert state.clock_locked is True
        assert state.lock_acquired is False

        mod._teardown_calibration_clocks(state, reset_clocks=True)
        assert calls == []

    def test_run_with_rocprofv3_passes_output_file(self, monkeypatch, tmp_path):
        mod = self._load_script()
        import sol_execbench.core.bench.rocm_profiler as rocm_profiler

        captured: dict[str, object] = {}

        def fake_build(command, **kwargs):
            captured["application_command"] = command
            captured["kwargs"] = kwargs
            return ["rocprofv3", "--", *command]

        def fake_run(command, **kwargs):
            captured["run_command"] = command
            captured["run_kwargs"] = kwargs
            return subprocess.CompletedProcess(command, 0, stdout="[1.25, 1.5]\n")

        monkeypatch.setattr(rocm_profiler, "build_rocprofv3_command", fake_build)
        monkeypatch.setattr(subprocess, "run", fake_run)

        tensor = SimpleNamespace(shape=(4,))
        temp_dir = tmp_path / "tmp" / "rdna4-overhead-calibration"
        durations = mod._run_with_rocprofv3(
            tensor,
            tensor,
            tensor,
            iterations=2,
            temp_dir=temp_dir,
        )

        assert durations == [1.25, 1.5]
        assert captured["kwargs"]["output_file"] == "rocprofv3-overhead-calibration"
        output_directory = Path(captured["kwargs"]["output_directory"])
        assert output_directory.is_relative_to(temp_dir)
