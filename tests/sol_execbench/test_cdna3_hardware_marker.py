# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""CDNA3 hardware-gated pytest surface and CPU-safe marker checks."""

from __future__ import annotations

import importlib
from typing import Any

import pytest


def _test_conftest() -> Any:
    return importlib.import_module("conftest")


def _skip_reasons(item: Any) -> list[str]:
    return [marker.mark.kwargs["reason"] for marker in item.added_markers]


class _FakeConfig:
    def __init__(self) -> None:
        self.registered_markers: list[str] = []

    def addinivalue_line(self, _name: str, line: str) -> None:
        self.registered_markers.append(line)

    def getoption(self, _name: str, default: str = "") -> str:
        return default


class _FakeItem:
    def __init__(self, *marker_names: str) -> None:
        self._marker_names = set(marker_names)
        self.added_markers: list[Any] = []
        self.keywords: dict[str, Any] = {}

    def iter_markers(self, name: str | None = None) -> tuple[Any, ...]:
        if name in self._marker_names:
            return (object(),)
        return ()

    def add_marker(self, marker: Any) -> None:
        self.added_markers.append(marker)


def test_cdna3_marker_is_registered_with_concrete_hardware_semantics() -> None:
    conftest = _test_conftest()
    config = _FakeConfig()

    conftest.pytest_configure(config)

    assert any(
        marker == "requires_cdna3: test requires an AMD CDNA 3 GPU, such as gfx942"
        for marker in config.registered_markers
    )


@pytest.mark.parametrize(
    ("gfx_arch", "expected"),
    [
        ("gfx940", True),
        ("gfx941", True),
        ("gfx942", True),
        ("gfx942:sramecc+:xnack-", True),
        ("gfx1200", False),
        ("", False),
    ],
)
def test_cdna3_architecture_detection_is_gfx94_family(
    gfx_arch: str, expected: bool
) -> None:
    conftest = _test_conftest()

    assert conftest._is_cdna3(gfx_arch) is expected


def test_cdna3_marker_skips_rdna4_with_detected_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conftest = _test_conftest()
    item = _FakeItem("requires_cdna3")

    monkeypatch.setattr(conftest, "_rocm_gpu_info", lambda: (True, "gfx1200", ""))
    monkeypatch.setattr(conftest, "_has_rocm_dev_headers", lambda: False)
    monkeypatch.setattr(conftest, "_has_ck_headers", lambda: False)
    monkeypatch.setattr(conftest, "_has_rocwmma_headers", lambda: False)

    conftest.pytest_collection_modifyitems(_FakeConfig(), [item])

    assert _skip_reasons(item) == [
        "requires AMD CDNA 3 ROCm GPU (detected gfx1200)"
    ]


def test_cdna3_marker_skips_missing_rocm_with_missing_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conftest = _test_conftest()
    item = _FakeItem("requires_cdna3")

    monkeypatch.setattr(
        conftest,
        "_rocm_gpu_info",
        lambda: (False, "", "PyTorch is not a ROCm build"),
    )
    monkeypatch.setattr(conftest, "_has_rocm_dev_headers", lambda: False)
    monkeypatch.setattr(conftest, "_has_ck_headers", lambda: False)
    monkeypatch.setattr(conftest, "_has_rocwmma_headers", lambda: False)

    conftest.pytest_collection_modifyitems(_FakeConfig(), [item])

    assert _skip_reasons(item) == ["PyTorch is not a ROCm build"]


def test_cdna3_marker_allows_detected_gfx94_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conftest = _test_conftest()
    item = _FakeItem("requires_cdna3")

    monkeypatch.setattr(conftest, "_rocm_gpu_info", lambda: (True, "gfx942", ""))
    monkeypatch.setattr(conftest, "_has_rocm_dev_headers", lambda: False)
    monkeypatch.setattr(conftest, "_has_ck_headers", lambda: False)
    monkeypatch.setattr(conftest, "_has_rocwmma_headers", lambda: False)

    conftest.pytest_collection_modifyitems(_FakeConfig(), [item])

    assert _skip_reasons(item) == []


@pytest.mark.requires_cdna3
@pytest.mark.requires_rocm
@pytest.mark.xdist_group("serial")
def test_cdna3_hardware_marker_runs_only_on_gfx94_rocm_device() -> None:
    conftest = _test_conftest()

    rocm_available, gfx_arch, rocm_skip_reason = conftest._rocm_gpu_info()

    assert rocm_available, rocm_skip_reason
    assert conftest._is_cdna3(gfx_arch)
    assert gfx_arch.startswith("gfx94")
