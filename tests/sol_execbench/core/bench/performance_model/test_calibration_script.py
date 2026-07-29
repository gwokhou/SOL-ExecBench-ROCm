from __future__ import annotations

import pytest

from sol_execbench.core.bench.performance_model.calibration import (
    MetricSample,
    ProbeBatch,
    build_calibration_parameters,
    parse_probe_metrics,
)

_METRICS = (
    ("dispatch_floor_ms", "device", 0.01, "ms"),
    ("valu_flop_per_ms", "fp32", 1000.0, "flop/ms"),
    ("wmma_flop_per_ms", "active_waves1", 5000.0, "flop/ms"),
    ("wmma_flop_per_ms", "active_waves3", 7000.0, "flop/ms"),
    ("wmma_flop_per_ms", "active_waves64", 10000.0, "flop/ms"),
    ("memory_byte_per_ms", "256KiB", 4000.0, "byte/ms"),
    ("memory_byte_per_ms", "16MiB", 3000.0, "byte/ms"),
    ("memory_byte_per_ms", "256MiB", 2000.0, "byte/ms"),
    ("memory_byte_per_ms", "contiguous", 2000.0, "byte/ms"),
    ("memory_byte_per_ms", "transpose", 1000.0, "byte/ms"),
    ("memory_byte_per_ms", "stride127", 500.0, "byte/ms"),
    ("lds_byte_per_ms", "normal", 8000.0, "byte/ms"),
    ("lds_byte_per_ms", "bank_conflict", 2000.0, "byte/ms"),
    ("reduction_op_per_ms", "width256", 600.0, "op/ms"),
    ("barrier_penalty_ms", "width256", 0.001, "ms/barrier"),
)
_FROZEN = {
    "wmma_variant": "active_waves64",
    "reduction_variant": "width256",
    "vram_variant": "256MiB",
    "l2_variant": "256KiB",
    "l3_variant": "16MiB",
}


def _metrics(scale: float) -> tuple[MetricSample, ...]:
    return tuple(
        MetricSample(name, variant, value * scale, unit)
        for name, variant, value, unit in _METRICS
    )


def test_calibration_metric_parser_and_frozen_parameter_build() -> None:
    parsed = parse_probe_metrics(
        "ignored\nMETRIC dispatch_floor_ms device 0.01 ms\n",
    )
    assert parsed == [MetricSample("dispatch_floor_ms", "device", 0.01, "ms")]
    held_out = [
        ProbeBatch(
            phase="held_out_after_configuration_freeze",
            process_batch=index,
            mode="all",
            metrics=_metrics(1.0 + index * 0.01),
            clocks_locked=True,
        )
        for index in range(5)
    ]
    parameters = build_calibration_parameters(held_out, _FROZEN)

    names = {parameter.name for parameter in parameters}
    assert {
        "dispatch_floor_ms",
        "l2_byte_per_ms",
        "l3_byte_per_ms",
        "vram_byte_per_ms",
        "transpose_access_efficiency",
        "stride_access_efficiency",
        "lds_bank_conflict_efficiency",
        "edge_wmma_efficiency",
        "irregular_wmma_efficiency",
    } <= names
    assert all(
        parameter.confidence_interval[0]
        <= parameter.value
        <= parameter.confidence_interval[1]
        for parameter in parameters
    )


def test_calibration_requires_five_independent_process_batches() -> None:
    held_out = [
        ProbeBatch(
            phase="held_out_after_configuration_freeze",
            process_batch=0,
            mode="all",
            metrics=_metrics(1.0) * 5,
            clocks_locked=True,
        ),
    ]

    with pytest.raises(ValueError, match="five process batches"):
        build_calibration_parameters(held_out, _FROZEN)
