from __future__ import annotations

import torch

from sol_execbench.core.bench.timing import (
    L2_CLEAR_BUFFER_BYTES,
    _get_empty_cache_for_benchmark,
)


def test_cache_clear_buffer_matches_paper_protocol(monkeypatch) -> None:
    observed = {}
    sentinel = object()

    def fake_empty(size, *, dtype, device):
        observed.update(size=size, dtype=dtype, device=device)
        return sentinel

    monkeypatch.setattr(torch, "empty", fake_empty)

    assert _get_empty_cache_for_benchmark("rocm-test-device") is sentinel
    assert observed == {
        "size": L2_CLEAR_BUFFER_BYTES,
        "dtype": torch.int8,
        "device": "rocm-test-device",
    }
