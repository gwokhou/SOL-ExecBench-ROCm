# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for optional, diagnostic-only Origami integration."""

from __future__ import annotations

from types import SimpleNamespace

from sol_execbench.core.scoring.amd_sol.origami import (
    OrigamiGemmProvider,
    OrigamiGemmRequest,
)


def _request() -> OrigamiGemmRequest:
    return OrigamiGemmRequest(
        m=128,
        n=256,
        k=64,
        batch=2,
        a_dtype="bf16",
        b_dtype="bf16",
        c_dtype="fp32",
        d_dtype="bf16",
    )


def test_origami_provider_requires_explicit_candidate_configs() -> None:
    estimate = OrigamiGemmProvider().estimate(_request(), configs=())

    assert estimate.status == "unavailable"
    assert estimate.reason_code == "missing_origami_candidate_configs"
    assert estimate.is_theoretical_lower_bound is False


def test_origami_provider_records_diagnostic_prediction_and_config() -> None:
    captured: dict[str, object] = {}
    config = SimpleNamespace(
        mt=SimpleNamespace(m=128, n=128, k=32),
        mi=SimpleNamespace(m=16, n=16, k=16),
        occupancy=4,
    )

    class FakeOrigami:
        __version__ = "test-1"
        transpose_t = SimpleNamespace(T="T", N="N")

        @staticmethod
        def get_hardware_for_device(device: int) -> str:
            captured["device"] = device
            return "gfx1200"

        @staticmethod
        def problem_t() -> SimpleNamespace:
            problem = SimpleNamespace()
            captured["problem"] = problem
            return problem

        @staticmethod
        def dim3_t(m: int, n: int, k: int) -> tuple[int, int, int]:
            return m, n, k

        @staticmethod
        def string_to_datatype(dtype: str) -> str:
            return f"dtype:{dtype}"

        @staticmethod
        def select_config(problem, hardware, configs):
            captured["selected"] = (problem, hardware, configs)
            return SimpleNamespace(latency=0.125, config=config)

    estimate = OrigamiGemmProvider(FakeOrigami).estimate(_request(), configs=(config,))

    assert estimate.status == "diagnostic"
    assert estimate.predicted_latency_ms == 0.125
    assert estimate.provider_version == "test-1"
    assert estimate.selected_config == {
        "mt_m": 128,
        "mt_n": 128,
        "mt_k": 32,
        "mi_m": 16,
        "mi_n": 16,
        "mi_k": 16,
        "occupancy": 4,
    }
    assert estimate.is_theoretical_lower_bound is False
    assert captured["device"] == 0
    assert captured["problem"].size == (128, 256, 64)
