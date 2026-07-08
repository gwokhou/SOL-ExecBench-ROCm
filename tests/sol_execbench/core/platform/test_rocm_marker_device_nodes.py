# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for ROCm pytest marker device-node diagnostics."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


def _test_conftest() -> Any:
    return importlib.import_module("conftest")


def test_rocm_gpu_info_reports_missing_device_nodes_before_torch_probe(
    monkeypatch,
) -> None:
    conftest = _test_conftest()
    monkeypatch.setattr(sys, "platform", "linux")

    available, gfx_arch, reason = conftest._rocm_gpu_info(
        path_exists=lambda _path: False
    )

    assert available is False
    assert gfx_arch == ""
    assert "ROCm device nodes unavailable in current execution environment" in reason
    assert "/dev/kfd" in reason
    assert "/dev/dri" in reason
    assert "Codex or container sandbox" in reason


def test_missing_rocm_device_nodes_only_reports_absent_nodes(monkeypatch) -> None:
    conftest = _test_conftest()
    monkeypatch.setattr(sys, "platform", "linux")

    missing = conftest._missing_rocm_device_nodes(
        path_exists=lambda path: path != Path("/dev/kfd")
    )

    assert missing == ("/dev/kfd",)
