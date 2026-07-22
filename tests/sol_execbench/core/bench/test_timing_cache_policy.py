from __future__ import annotations

import torch

from sol_execbench.core.bench.timing import (
    _get_empty_cache_for_benchmark,
)
from sol_execbench.core.platform.runtime import CacheClearPolicy


def test_cache_clear_buffer_uses_resolved_target_policy(monkeypatch) -> None:
    observed = {}
    sentinel = object()

    def fake_empty(size, *, dtype, device):
        observed.update(size=size, dtype=dtype, device=device)
        return sentinel

    monkeypatch.setattr(torch, "empty", fake_empty)

    policy = CacheClearPolicy(
        detected_l2_bytes=4 * 1024**2,
        clear_buffer_bytes=8 * 1024**2,
        source="torch_device_properties",
    )
    assert _get_empty_cache_for_benchmark("rocm-test-device", policy) is sentinel
    assert observed == {
        "size": 8 * 1024**2,
        "dtype": torch.int8,
        "device": "rocm-test-device",
    }
