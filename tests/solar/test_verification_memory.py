# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from solar.verification import verify


def test_release_case_memory_collects_cycles_before_emptying_cache(
    monkeypatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(verify.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: events.append("empty_cache"),
    )

    verify._release_case_memory("cuda:0")

    assert events == ["gc", "empty_cache"]


def test_release_case_memory_ignores_cpu_device(monkeypatch) -> None:
    monkeypatch.setattr(
        verify.gc,
        "collect",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected GC")),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    verify._release_case_memory("cpu")
