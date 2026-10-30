# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from sol_execbench.core.platform import amdsmi_telemetry as telemetry


class Header(ctypes.Structure):
    _fields_ = [
        ("format_revision", ctypes.c_uint8),
        ("content_revision", ctypes.c_uint8),
        ("structure_size", ctypes.c_uint16),
    ]


class Metrics(ctypes.Structure):
    _fields_ = [
        ("common_header", Header),
        ("throttle_status", ctypes.c_uint32),
        ("indep_throttle_status", ctypes.c_uint64),
    ]


def fake_api(tmp_path, raw, rc=0):
    binding = tmp_path / "binding.py"
    binding.write_text("# fixture")
    calls = []

    def query(handle, pointer):
        calls.append(("query", handle))
        metrics = pointer._obj
        metrics.throttle_status = raw
        metrics.indep_throttle_status = (1 << 36) if raw else 0
        metrics.common_header = Header(1, 3, 120)
        return rc

    api = SimpleNamespace(
        amdsmi_wrapper=SimpleNamespace(
            amdsmi_gpu_metrics_t=Metrics,
            amdsmi_get_gpu_metrics_info=query,
            AMDSMI_STATUS_SUCCESS=0,
        ),
        MaxUIntegerTypes=SimpleNamespace(
            UINT32_T=(1 << 32) - 1, UINT64_T=(1 << 64) - 1
        ),
        _validate_if_max_uint=lambda value, limit, **kwargs: (
            "N/A" if value == limit else bool(value)
        ),
        amdsmi_get_lib_version=lambda: {"major": 27, "minor": 1},
        __file__=str(binding),
        amdsmi_get_gpu_device_uuid=lambda h: "target",
        amdsmi_get_gpu_device_bdf=lambda h: "0000:05:00.0",
        amdsmi_init=lambda: calls.append("init"),
        amdsmi_shut_down=lambda: calls.append("shutdown"),
        amdsmi_get_processor_handles=lambda: [42],
    )
    return api, calls


@pytest.mark.parametrize(
    ("raw", "converted"), [(0, False), (2, True), (0xFFFFFFFF, "N/A")]
)
def test_capture_keeps_raw_before_binding_bool_conversion(
    tmp_path, monkeypatch, raw, converted
):
    api, calls = fake_api(tmp_path, raw)
    monkeypatch.setattr(telemetry, "_load_api", lambda _: api)
    monkeypatch.setattr(
        telemetry,
        "run_in_process_group_bounded",
        lambda *a, **k: subprocess.CompletedProcess(
            [], 0, "AMDSMI Tool: 27.1.0", ""
        ),
    )
    result = telemetry.collect_throttle_evidence(Path("/usr/bin/amd-smi"), 0)
    raw_result = result["raw"]
    assert isinstance(raw_result, dict)
    field = cast(dict[str, object], raw_result.get("throttle_status"))
    assert isinstance(field, dict)
    assert field["value"] == raw and field["type"] == "int"
    assert field["binding_value"] == converted
    assert field["binding_type"] == type(converted).__name__
    assert result["metrics_header"] == {
        "format_revision": 1,
        "content_revision": 3,
        "structure_size": 120,
    }
    assert result["api_version"] == {"major": 27, "minor": 1}
    assert calls == ["init", ("query", 42), "shutdown"]


def test_query_failure_still_closes_management_handle(tmp_path, monkeypatch):
    api, calls = fake_api(tmp_path, 0, rc=5)
    monkeypatch.setattr(telemetry, "_load_api", lambda _: api)
    monkeypatch.setattr(
        telemetry,
        "run_in_process_group_bounded",
        lambda *a, **k: subprocess.CompletedProcess(
            [], 0, "AMDSMI Tool: 27.1.0", ""
        ),
    )
    with pytest.raises(RuntimeError, match="query failed: 5"):
        telemetry.collect_throttle_evidence(Path("/usr/bin/amd-smi"), 0)
    assert calls[-1] == "shutdown"


def test_producer_keeps_cli_and_raw_observations(load_script, monkeypatch):
    import json

    producer = load_script(
        "scripts/internal/rdna4/run_rdna4_resource_peak_calibration.py"
    )
    raw = {
        "raw": {"throttle_status": {"value": 0xFFFFFFFF, "type": "int"}},
        "metrics_header": {
            "format_revision": 1,
            "content_revision": 3,
            "structure_size": 120,
        },
        "api_version": {"major": 27},
        "cli_version": "27.1",
    }
    commands = []

    def run(command):
        commands.append(command)
        value = (
            {"gpu_data": [{"power": {"throttle_status": "THROTTLED"}}]}
            if command[1] == "metric"
            else raw
        )
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    monkeypatch.setattr(producer, "_run", run)
    result = producer._telemetry_snapshot("/usr/bin/amd-smi")
    assert result["throttle_status"] == "UNKNOWN"
    assert result["throttle_status_cli"] == "THROTTLED"
    assert result["throttle_evidence"] == raw
    assert len(commands) == 2
