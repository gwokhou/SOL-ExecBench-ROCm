# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for native GPU execution visibility and KFD partition checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from sol_execbench.core.bench import gpu_execution
from sol_execbench.core.bench.clock_ownership import (
    ClockDevice,
    ClockOwnershipError,
)

UUID_A = "11111111-0000-1000-8000-000000000001"
UUID_B = "11111111-0000-1000-8000-000000000002"
BDF = "0000:05:00.0"


def test_resolve_gpu_device_supports_kfd_partition_and_rejects_legacy_ordinals(
    tmp_path: Path, monkeypatch
) -> None:
    pci_dir = tmp_path / "pci" / BDF
    pci_dir.mkdir(parents=True)
    (pci_dir / "unique_id").write_text("aaaa000000000001\n", encoding="utf-8")

    kfd_root = tmp_path / "kfd"
    for idx, uid_int in ((1, 0xAAAA000000000001), (2, 0xAAAA000000000002)):
        node = kfd_root / str(idx)
        node.mkdir(parents=True)
        (node / "properties").write_text(
            f"simd_count 64\ndomain 0\nlocation_id 1280\nunique_id {uid_int}\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(gpu_execution, "PCI_DEVICES_ROOT", tmp_path / "pci")
    monkeypatch.setattr(gpu_execution, "KFD_TOPOLOGY_NODES_ROOT", kfd_root)
    monkeypatch.setattr(
        gpu_execution,
        "observe_clock_devices",
        lambda: (
            ClockDevice(bdf=BDF, uuid=UUID_A),
            ClockDevice(bdf=BDF, uuid=UUID_B),
        ),
    )
    monkeypatch.setenv("GPU_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setenv("GPU_LOCK_UUID", UUID_B)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-aaaa000000000002")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("GPU_DEVICE_ORDINAL", raising=False)

    assert gpu_execution.resolve_gpu_device() == ClockDevice(
        bdf=BDF, uuid=UUID_B
    )

    monkeypatch.setenv("GPU_DEVICE_ORDINAL", "1")
    with pytest.raises(ClockOwnershipError, match="GPU_DEVICE_ORDINAL"):
        gpu_execution.resolve_gpu_device()
    monkeypatch.delenv("GPU_DEVICE_ORDINAL")

    # Reject legacy CUDA visibility remapping if set to another index.
    legacy_var = "CUDA" + "_VISIBLE_DEVICES"
    monkeypatch.setenv(legacy_var, "1")
    with pytest.raises(ClockOwnershipError, match="preserve the single"):
        gpu_execution.resolve_gpu_device()
