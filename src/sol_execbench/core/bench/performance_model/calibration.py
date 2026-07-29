# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Pure parsing and statistics for diagnostic calibration evidence."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from sol_execbench.core.bench.performance_model.models import (
    CalibrationParameter,
)

METRIC_PREFIX = "METRIC "
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_729


@dataclass(frozen=True, slots=True)
class MetricSample:
    """One tagged device-event measurement."""

    name: str
    variant: str
    value: float
    unit: str


@dataclass(frozen=True, slots=True)
class ProbeBatch:
    """One fresh-process result for a probe mode."""

    phase: str
    process_batch: int
    mode: str
    metrics: tuple[MetricSample, ...]
    clocks_locked: bool

    def to_dict(self) -> dict[str, object]:
        """Return deterministic audit data."""
        return {
            "phase": self.phase,
            "process_batch": self.process_batch,
            "mode": self.mode,
            "clocks_locked": self.clocks_locked,
            "metrics": [
                {
                    "name": metric.name,
                    "variant": metric.variant,
                    "value": metric.value,
                    "unit": metric.unit,
                }
                for metric in self.metrics
            ],
        }


def parse_probe_metrics(output: str) -> list[MetricSample]:
    """Parse strict tagged metric lines while ignoring diagnostic output."""
    metrics: list[MetricSample] = []
    for line in output.splitlines():
        if not line.startswith(METRIC_PREFIX):
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"malformed diagnostic metric: {line}")
        value = float(fields[3])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"invalid diagnostic metric: {line}")
        metrics.append(MetricSample(fields[1], fields[2], value, fields[4]))
    return metrics


def freeze_probe_configuration(
    tuning: Sequence[ProbeBatch],
) -> dict[str, str]:
    """Select tuning variants before independent held-out collection."""
    if not tuning or any(
        batch.phase != "tuning" or not batch.clocks_locked for batch in tuning
    ):
        raise ValueError("tuning evidence must be locked and tuning-only")
    grouped = _group_values(tuning)
    return {
        "wmma_variant": _best_variant(grouped, "wmma_flop_per_ms"),
        "reduction_variant": _best_variant(grouped, "reduction_op_per_ms"),
        "vram_variant": "256MiB",
        "l2_variant": "256KiB",
        "l3_variant": "16MiB",
    }


def build_calibration_parameters(
    held_out: Sequence[ProbeBatch],
    frozen: dict[str, str],
) -> list[CalibrationParameter]:
    """Build rates, working-set transitions, and efficiency parameters."""
    if not held_out or any(
        batch.phase != "held_out_after_configuration_freeze"
        or not batch.clocks_locked
        for batch in held_out
    ):
        raise ValueError(
            "held-out evidence must be locked and collected after freeze",
        )
    grouped = _group_process_medians(held_out)
    parameters = [
        _parameter(
            grouped,
            source_name="dispatch_floor_ms",
            variant="device",
            target_name="dispatch_floor_ms",
        ),
        _parameter(
            grouped,
            source_name="valu_flop_per_ms",
            variant="fp32",
            target_name="valu_flop_per_ms",
        ),
        _parameter(
            grouped,
            source_name="wmma_flop_per_ms",
            variant=frozen["wmma_variant"],
            target_name="wmma_flop_per_ms",
        ),
        _parameter(
            grouped,
            source_name="memory_byte_per_ms",
            variant=frozen["vram_variant"],
            target_name="vram_byte_per_ms",
            applicability=(64.0 * 2**20, 256.0 * 2**20),
        ),
        _parameter(
            grouped,
            source_name="lds_byte_per_ms",
            variant="normal",
            target_name="lds_byte_per_ms",
        ),
        _parameter(
            grouped,
            source_name="reduction_op_per_ms",
            variant=frozen["reduction_variant"],
            target_name="reduction_op_per_ms",
        ),
        _parameter(
            grouped,
            source_name="barrier_penalty_ms",
            variant=frozen["reduction_variant"],
            target_name="barrier_penalty_ms",
        ),
    ]
    parameters.extend(_hierarchy_and_efficiency_parameters(grouped, frozen))
    return parameters


def _group_values(
    batches: Iterable[ProbeBatch],
) -> dict[tuple[str, str, str], list[float]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for batch in batches:
        for metric in batch.metrics:
            grouped[(metric.name, metric.variant, metric.unit)].append(
                metric.value,
            )
    return dict(grouped)


def _group_process_medians(
    batches: Iterable[ProbeBatch],
) -> dict[tuple[str, str, str], list[float]]:
    grouped: dict[
        tuple[str, str, str],
        dict[int, list[float]],
    ] = defaultdict(lambda: defaultdict(list))
    for batch in batches:
        for metric in batch.metrics:
            grouped[(metric.name, metric.variant, metric.unit)][
                batch.process_batch
            ].append(metric.value)
    return {
        key: [
            statistics.median(by_process[index]) for index in sorted(by_process)
        ]
        for key, by_process in grouped.items()
    }


def _best_variant(
    grouped: dict[tuple[str, str, str], list[float]],
    metric_name: str,
) -> str:
    candidates = {
        variant: statistics.median(values)
        for (name, variant, _unit), values in grouped.items()
        if name == metric_name
    }
    if not candidates:
        raise RuntimeError(f"tuning lacks {metric_name}")
    return max(candidates, key=lambda variant: (candidates[variant], variant))


def _bootstrap_interval(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 5:
        raise ValueError(
            "held-out parameter requires at least five process batches"
        )
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0,
        array.size,
        size=(BOOTSTRAP_REPLICATES, array.size),
    )
    medians = np.median(array[indices], axis=1)
    lower, upper = np.percentile(medians, [2.5, 97.5])
    value = statistics.median(values)
    return min(float(lower), value), max(float(upper), value)


def _parameter(
    grouped: dict[tuple[str, str, str], list[float]],
    *,
    source_name: str,
    variant: str,
    target_name: str,
    applicability: tuple[float, float] | None = None,
) -> CalibrationParameter:
    matches = [
        (unit, values)
        for (name, observed_variant, unit), values in grouped.items()
        if name == source_name and observed_variant == variant
    ]
    if len(matches) != 1:
        raise RuntimeError(f"held-out evidence lacks {source_name}:{variant}")
    unit, values = matches[0]
    value = statistics.median(values)
    return CalibrationParameter(
        name=target_name,
        value=value,
        unit=unit,
        confidence_interval=_bootstrap_interval(values),
        applicability=applicability,
    )


def _ratio_parameter(
    grouped: dict[tuple[str, str, str], list[float]],
    *,
    source_name: str,
    numerator_variant: str,
    denominator_variant: str,
    target_name: str,
) -> CalibrationParameter:
    numerator = _metric_values(grouped, source_name, numerator_variant)
    denominator = _metric_values(grouped, source_name, denominator_variant)
    count = min(len(numerator), len(denominator))
    ratios = [
        numerator[index] / denominator[index]
        for index in range(count)
        if denominator[index] > 0
    ]
    value = statistics.median(ratios)
    return CalibrationParameter(
        name=target_name,
        value=value,
        unit="ratio",
        confidence_interval=_bootstrap_interval(ratios),
        applicability=(0.0, 1.0),
    )


def _metric_values(
    grouped: dict[tuple[str, str, str], list[float]],
    name: str,
    variant: str,
) -> list[float]:
    matches = [
        values
        for (observed_name, observed_variant, _unit), values in grouped.items()
        if observed_name == name and observed_variant == variant
    ]
    if len(matches) != 1:
        raise RuntimeError(f"held-out evidence lacks {name}:{variant}")
    return matches[0]


def _hierarchy_and_efficiency_parameters(
    grouped: dict[tuple[str, str, str], list[float]],
    frozen: dict[str, str],
) -> list[CalibrationParameter]:
    return [
        _parameter(
            grouped,
            source_name="memory_byte_per_ms",
            variant=frozen["l2_variant"],
            target_name="l2_byte_per_ms",
            applicability=(0.0, 256.0 * 2**10),
        ),
        _parameter(
            grouped,
            source_name="memory_byte_per_ms",
            variant=frozen["l3_variant"],
            target_name="l3_byte_per_ms",
            applicability=(2.0 * 2**20, 16.0 * 2**20),
        ),
        _ratio_parameter(
            grouped,
            source_name="memory_byte_per_ms",
            numerator_variant="transpose",
            denominator_variant="contiguous",
            target_name="transpose_access_efficiency",
        ),
        _ratio_parameter(
            grouped,
            source_name="memory_byte_per_ms",
            numerator_variant="stride127",
            denominator_variant="contiguous",
            target_name="stride_access_efficiency",
        ),
        _ratio_parameter(
            grouped,
            source_name="lds_byte_per_ms",
            numerator_variant="bank_conflict",
            denominator_variant="normal",
            target_name="lds_bank_conflict_efficiency",
        ),
        _ratio_parameter(
            grouped,
            source_name="wmma_flop_per_ms",
            numerator_variant="active_waves3",
            denominator_variant=frozen["wmma_variant"],
            target_name="irregular_wmma_efficiency",
        ),
        _ratio_parameter(
            grouped,
            source_name="wmma_flop_per_ms",
            numerator_variant="active_waves1",
            denominator_variant=frozen["wmma_variant"],
            target_name="edge_wmma_efficiency",
        ),
    ]


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "MetricSample",
    "ProbeBatch",
    "build_calibration_parameters",
    "freeze_probe_configuration",
    "parse_probe_metrics",
]
