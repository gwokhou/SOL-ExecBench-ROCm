# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for rocprofv3 overhead calibration and profiler_overhead_ms integration."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import cast

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
    from sol_execbench.core.bench.timing_policy import TimingSourceType

    defaults = {
        "source_type": cast(TimingSourceType, object()),
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
        SCRIPT_PATH = REPO_ROOT / "scripts/run_rdna4_profiler_overhead_calibration.py"
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

    def test_iterations_flag(self):
        mod = self._load_script()
        args = mod.parse_args(["--iterations", "50"])
        assert args.iterations == 50


class TestCalibrationJsonSchema:
    """Test calibration JSON output schema validation."""

    def test_schema_version_constant(self):
        from importlib.util import module_from_spec, spec_from_file_location

        REPO_ROOT = Path(__file__).resolve().parents[2]
        SCRIPT_PATH = REPO_ROOT / "scripts/run_rdna4_profiler_overhead_calibration.py"
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
