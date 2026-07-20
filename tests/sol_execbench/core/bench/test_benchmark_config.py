from __future__ import annotations

from sol_execbench.core.bench.config.benchmark_config import (
    CUSTOM_ROCM_TIMING_PROTOCOL,
    OFFICIAL_ROCM_TIMING_PROTOCOL,
    BenchmarkConfig,
)


def test_default_config_uses_rocm_protocol_with_paper_iteration_counts():
    config = BenchmarkConfig()

    assert config.warmup_runs == 10
    assert config.iterations == 50
    assert config.trials == 3
    assert config.min_measurement_time_seconds is None
    assert config.lock_clocks is True
    assert config.timing_protocol == OFFICIAL_ROCM_TIMING_PROTOCOL


def test_timing_override_is_explicitly_diagnostic():
    config = BenchmarkConfig(iterations=1)

    assert config.timing_protocol == CUSTOM_ROCM_TIMING_PROTOCOL


def test_unlocked_timing_cannot_claim_official_rocm_protocol():
    config = BenchmarkConfig(lock_clocks=False)

    assert config.timing_protocol == CUSTOM_ROCM_TIMING_PROTOCOL
