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
    ApplicabilityDimension,
    CalibrationParameter,
    CalibrationParameterName,
    CalibrationSurface,
    CalibrationSurfaceCell,
    CalibrationSurfaceName,
    CalibrationUnit,
)
from sol_execbench.core.bench.performance_model.vram_policy import MIB

METRIC_PREFIX = "METRIC "
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_729
_INDEXED_READ_LOCALITY_INTERVALS = (
    (0.0, 0.33),
    (0.34, 0.66),
    (0.67, 1.0),
)
_INDEXED_READ_WORKING_SET_INTERVALS = (
    (0.0, 2.0 * MIB),
    (2.0 * MIB + 1e-6, 64.0 * MIB),
    (64.0 * MIB + 1e-6, 128.0 * MIB),
    (128.0 * MIB + 1e-6, 256.0 * MIB),
    (256.0 * MIB + 1e-6, 512.0 * MIB),
)
_INDEXED_READ_ELEMENT_INTERVALS = ((2.0, 2.0), (4.0, 4.0))
_INDEXED_READ_DIMENSIONS = (
    ApplicabilityDimension.INDEX_LOCALITY,
    ApplicabilityDimension.WORKING_SET_BYTES,
    ApplicabilityDimension.ELEMENT_BYTES,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricSample:
    """One tagged device-event measurement."""

    name: str
    variant: str
    value: float
    unit: str


@dataclass(frozen=True, slots=True, kw_only=True)
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
        metrics.append(
            MetricSample(
                name=fields[1], variant=fields[2], value=value, unit=fields[4]
            )
        )
    return metrics


_VRAM_VARIANT_BYTES = {
    "256MiB": 256 * MIB,
    "512MiB": 512 * MIB,
}


def freeze_probe_configuration(
    tuning: Sequence[ProbeBatch],
    *,
    vram_variant: str = "256MiB",
) -> dict[str, str]:
    """Select tuning variants before independent held-out collection."""
    if not tuning or any(
        batch.phase != "tuning" or not batch.clocks_locked for batch in tuning
    ):
        raise ValueError("tuning evidence must be locked and tuning-only")
    grouped = _group_values(tuning)
    if vram_variant not in _VRAM_VARIANT_BYTES:
        raise ValueError(
            f"unsupported VRAM calibration variant: {vram_variant}"
        )
    return {
        "wmma_variant": _best_variant(grouped, "wmma_flop_per_ms"),
        "reduction_variant": _best_variant(grouped, "reduction_op_per_ms"),
        "vram_variant": vram_variant,
        "l2_variant": "256KiB",
        "l3_variant": "16MiB",
    }


def build_calibration_parameters(
    held_out: Sequence[ProbeBatch],
    frozen: dict[str, str],
) -> list[CalibrationParameter]:
    """Build rates, working-set transitions, and efficiency parameters."""
    if not held_out or any(
        batch.phase != "parameter_estimation_after_configuration_freeze"
        or not batch.clocks_locked
        for batch in held_out
    ):
        raise ValueError(
            "parameter-estimation evidence must be locked and collected "
            "after freeze",
        )
    grouped = _group_process_medians(held_out)
    parameters = [
        _parameter(
            grouped,
            source_name="dispatch_floor_ms",
            variant="device",
            target_name=CalibrationParameterName.DISPATCH_FLOOR_MS,
            expected_unit=CalibrationUnit.MS,
        ),
        _parameter(
            grouped,
            source_name="wmma_flop_per_ms",
            variant=frozen["wmma_variant"],
            target_name=CalibrationParameterName.WMMA_F16_F32_FLOP_PER_MS,
            expected_unit=CalibrationUnit.FLOP_PER_MS,
        ),
        _parameter(
            grouped,
            source_name="memory_byte_per_ms",
            variant=frozen["vram_variant"],
            target_name=CalibrationParameterName.VRAM_BYTE_PER_MS,
            expected_unit=CalibrationUnit.BYTE_PER_MS,
            applicability=(
                64.0 * 2**20,
                float(_VRAM_VARIANT_BYTES[frozen["vram_variant"]]),
            ),
            applicability_dimension=(ApplicabilityDimension.WORKING_SET_BYTES),
        ),
        _parameter(
            grouped,
            source_name="lds_byte_per_ms",
            variant="normal",
            target_name=CalibrationParameterName.LDS_BYTE_PER_MS,
            expected_unit=CalibrationUnit.BYTE_PER_MS,
        ),
        _parameter(
            grouped,
            source_name="lds_bank_conflict_penalty_ms",
            variant="bank_conflict",
            target_name=(CalibrationParameterName.LDS_BANK_CONFLICT_PENALTY_MS),
            expected_unit=CalibrationUnit.MS_PER_EVENT,
        ),
        _parameter(
            grouped,
            source_name="fp32_matrix_flop_per_ms",
            variant="dense",
            target_name=CalibrationParameterName.FP32_MATRIX_FLOP_PER_MS,
            expected_unit=CalibrationUnit.FLOP_PER_MS,
        ),
        _parameter(
            grouped,
            source_name="indexed_address_op_per_ms",
            variant="device",
            target_name=CalibrationParameterName.INDEXED_ADDRESS_OP_PER_MS,
            expected_unit=CalibrationUnit.ITEM_PER_MS,
        ),
    ]
    parameters.extend(_valu_parameters(grouped))
    parameters.extend(_reduction_parameters(grouped))
    parameters.extend(_hierarchy_and_efficiency_parameters(grouped, frozen))
    return parameters


def build_calibration_surfaces(
    held_out: Sequence[ProbeBatch],
) -> list[CalibrationSurface]:
    """Build the four required multidimensional calibration surfaces."""
    if not held_out or any(
        batch.phase != "parameter_estimation_after_configuration_freeze"
        or not batch.clocks_locked
        for batch in held_out
    ):
        raise ValueError("surface evidence must be locked and held out")
    grouped = _group_process_medians(held_out)
    return [
        _calibration_surface(
            grouped,
            metric_name="indexed_read_item_per_ms",
            name=CalibrationSurfaceName.INDEXED_READ,
            unit=CalibrationUnit.ITEM_PER_MS,
            dimensions=(
                ApplicabilityDimension.INDEX_LOCALITY,
                ApplicabilityDimension.WORKING_SET_BYTES,
                ApplicabilityDimension.ELEMENT_BYTES,
            ),
        ),
        _calibration_surface(
            grouped,
            metric_name="atomic_update_item_per_ms",
            name=CalibrationSurfaceName.ATOMIC_UPDATE,
            unit=CalibrationUnit.ITEM_PER_MS,
            dimensions=(
                ApplicabilityDimension.COLLISION_FRACTION,
                ApplicabilityDimension.MAX_MULTIPLICITY,
                ApplicabilityDimension.ELEMENT_BYTES,
            ),
        ),
        _calibration_surface(
            grouped,
            metric_name="residency_ratio",
            name=CalibrationSurfaceName.RESIDENCY,
            unit=CalibrationUnit.RATIO,
            dimensions=(
                ApplicabilityDimension.ACTIVE_WAVES,
                ApplicabilityDimension.ELEMENT_BYTES,
            ),
        ),
        _overlap_calibration_surface(grouped),
    ]


def validate_indexed_read_surface_capacity(
    surface: CalibrationSurface,
    maximum_working_set_bytes: int,
) -> None:
    """Require the exact capacity-governed indexed-read cell matrix."""
    if (
        surface.name is not CalibrationSurfaceName.INDEXED_READ
        or surface.unit is not CalibrationUnit.ITEM_PER_MS
    ):
        raise ValueError(
            "indexed-read capacity validation received wrong surface"
        )
    working_sets = tuple(
        interval
        for interval in _INDEXED_READ_WORKING_SET_INTERVALS
        if interval[1] <= maximum_working_set_bytes
    )
    expected = {
        (
            locality,
            working_set,
            element,
        )
        for locality in _INDEXED_READ_LOCALITY_INTERVALS
        for working_set in working_sets
        for element in _INDEXED_READ_ELEMENT_INTERVALS
    }
    actual = {
        tuple(
            cell.coordinates[dimension]
            for dimension in _INDEXED_READ_DIMENSIONS
        )
        for cell in surface.cells
        if set(cell.coordinates) == set(_INDEXED_READ_DIMENSIONS)
    }
    if actual != expected or len(surface.cells) != len(expected):
        raise ValueError(
            "indexed-read calibration surface does not exactly cover "
            "the frozen VRAM capacity tier"
        )


def _overlap_calibration_surface(
    grouped: dict[tuple[str, str, str], list[float]],
) -> CalibrationSurface:
    valu_rate = statistics.median(
        grouped[("valu_item_per_ms", "simple_fp32", "item/ms")]
    )
    memory_rate = statistics.median(
        grouped[("memory_byte_per_ms", "256MiB", "byte/ms")]
    )
    count = float(32 * 2**20 // 4)
    memory_time = 2.0 * count * 4.0 / memory_rate
    cells = []
    for (name, variant, unit), values in grouped.items():
        if name != "overlap_ratio" or unit != CalibrationUnit.RATIO.value:
            continue
        raw = dict(item.split("=", 1) for item in variant.split(","))
        repetitions = raw.get("arithmetic_repetitions")
        concurrent = raw.get("concurrent_dispatches")
        if repetitions is None or concurrent is None:
            raise ValueError(f"invalid overlap surface variant: {variant}")
        mix = (
            1.0
            if repetitions == "pure_compute"
            else _overlap_compute_mix(
                count,
                int(repetitions),
                valu_rate,
                memory_time,
            )
        )
        concurrent_interval = _surface_interval(concurrent, variant)
        cells.append(
            CalibrationSurfaceCell(
                coordinates={
                    ApplicabilityDimension.RESOURCE_MIX: (mix, mix),
                    ApplicabilityDimension.CONCURRENT_DISPATCHES: (
                        concurrent_interval
                    ),
                },
                value=statistics.median(values),
                confidence_interval=_bootstrap_interval(values),
            )
        )
    if not cells:
        raise ValueError("surface evidence lacks overlap_ratio")
    return CalibrationSurface(
        name=CalibrationSurfaceName.OVERLAP,
        unit=CalibrationUnit.RATIO,
        cells=sorted(
            cells,
            key=lambda cell: cell.coordinates[
                ApplicabilityDimension.RESOURCE_MIX
            ],
        ),
    )


def _overlap_compute_mix(
    count: float,
    repetitions: int,
    valu_rate: float,
    memory_time: float,
) -> float:
    compute_time = count * repetitions / valu_rate
    return compute_time / (compute_time + memory_time)


def _surface_interval(value: str, variant: str) -> tuple[float, float]:
    lower, separator, upper = value.partition(":")
    if not separator:
        raise ValueError(f"invalid surface variant: {variant}")
    return float(lower), float(upper)


def _calibration_surface(
    grouped: dict[tuple[str, str, str], list[float]],
    *,
    metric_name: str,
    name: CalibrationSurfaceName,
    unit: CalibrationUnit,
    dimensions: tuple[ApplicabilityDimension, ...],
) -> CalibrationSurface:
    cells = [
        CalibrationSurfaceCell(
            coordinates=_surface_coordinates(variant, dimensions),
            value=statistics.median(values),
            confidence_interval=_bootstrap_interval(values),
        )
        for (observed_name, variant, observed_unit), values in grouped.items()
        if observed_name == metric_name and observed_unit == unit.value
    ]
    if not cells:
        raise ValueError(f"surface evidence lacks {metric_name}")
    return CalibrationSurface(name=name, unit=unit, cells=cells)


def _surface_coordinates(
    variant: str,
    dimensions: tuple[ApplicabilityDimension, ...],
) -> dict[ApplicabilityDimension, tuple[float, float]]:
    raw = {}
    for item in variant.split(","):
        key, separator, interval = item.partition("=")
        lower, range_separator, upper = interval.partition(":")
        if not separator or not range_separator:
            raise ValueError(f"invalid surface variant: {variant}")
        raw[key] = (float(lower), float(upper))
    expected = {dimension.value for dimension in dimensions}
    if set(raw) != expected:
        raise ValueError(f"surface variant dimensions mismatch: {variant}")
    return {dimension: raw[dimension.value] for dimension in dimensions}


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
    target_name: CalibrationParameterName,
    expected_unit: CalibrationUnit,
    applicability: tuple[float, float] | None = None,
    applicability_dimension: ApplicabilityDimension | None = None,
) -> CalibrationParameter:
    matches = [
        (unit, values)
        for (name, observed_variant, unit), values in grouped.items()
        if name == source_name and observed_variant == variant
    ]
    if len(matches) != 1:
        raise RuntimeError(f"held-out evidence lacks {source_name}:{variant}")
    unit, values = matches[0]
    if unit != expected_unit:
        raise RuntimeError(
            f"{source_name}:{variant} has unit {unit}, expected {expected_unit}"
        )
    value = statistics.median(values)
    return CalibrationParameter(
        name=target_name,
        value=value,
        unit=expected_unit,
        confidence_interval=_bootstrap_interval(values),
        applicability=applicability,
        applicability_dimension=applicability_dimension,
    )


def _ratio_parameter(
    grouped: dict[tuple[str, str, str], list[float]],
    *,
    source_name: str,
    numerator_variant: str,
    denominator_variant: str,
    target_name: CalibrationParameterName,
    applicability: tuple[float, float] | None = None,
    applicability_dimension: ApplicabilityDimension | None = None,
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
        unit=CalibrationUnit.RATIO,
        confidence_interval=_bootstrap_interval(ratios),
        applicability=applicability,
        applicability_dimension=applicability_dimension,
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
            target_name=CalibrationParameterName.L2_BYTE_PER_MS,
            expected_unit=CalibrationUnit.BYTE_PER_MS,
            applicability=(0.0, 2.0 * 2**20),
            applicability_dimension=(ApplicabilityDimension.WORKING_SET_BYTES),
        ),
        _parameter(
            grouped,
            source_name="memory_byte_per_ms",
            variant=frozen["l3_variant"],
            target_name=CalibrationParameterName.L3_BYTE_PER_MS,
            expected_unit=CalibrationUnit.BYTE_PER_MS,
            applicability=(2.0 * 2**20, 64.0 * 2**20),
            applicability_dimension=(ApplicabilityDimension.WORKING_SET_BYTES),
        ),
        _ratio_parameter(
            grouped,
            source_name="memory_byte_per_ms",
            numerator_variant="transpose",
            denominator_variant="contiguous",
            target_name=CalibrationParameterName.TRANSPOSE_EFFICIENCY,
        ),
        _ratio_parameter(
            grouped,
            source_name="wmma_flop_per_ms",
            numerator_variant=(f"irregular_tile17_{frozen['wmma_variant']}"),
            denominator_variant=frozen["wmma_variant"],
            target_name=CalibrationParameterName.IRREGULAR_WMMA_EFFICIENCY,
            applicability=(1.0, 15.0),
            applicability_dimension=ApplicabilityDimension.TILE_REMAINDER,
        ),
        _ratio_parameter(
            grouped,
            source_name="wmma_flop_per_ms",
            numerator_variant=f"edge_tile15_{frozen['wmma_variant']}",
            denominator_variant=frozen["wmma_variant"],
            target_name=CalibrationParameterName.EDGE_WMMA_EFFICIENCY,
            applicability=(1.0, 15.0),
            applicability_dimension=ApplicabilityDimension.TILE_REMAINDER,
        ),
        _ratio_parameter(
            grouped,
            source_name="fp32_matrix_flop_per_ms",
            numerator_variant="strided",
            denominator_variant="dense",
            target_name=CalibrationParameterName.STRIDED_MATMUL_EFFICIENCY,
        ),
    ]


def _valu_parameters(
    grouped: dict[tuple[str, str, str], list[float]],
) -> list[CalibrationParameter]:
    variants = {
        CalibrationParameterName.VALU_SIMPLE_FP32_PER_MS: "simple_fp32",
        CalibrationParameterName.VALU_SIMPLE_BF16_PER_MS: "simple_bf16",
        CalibrationParameterName.VALU_TRANSCENDENTAL_FP32_PER_MS: (
            "transcendental_fp32"
        ),
        CalibrationParameterName.VALU_TRANSCENDENTAL_BF16_PER_MS: (
            "transcendental_bf16"
        ),
        CalibrationParameterName.VALU_COMPOSITE_FP32_PER_MS: "composite_fp32",
        CalibrationParameterName.VALU_COMPOSITE_BF16_PER_MS: "composite_bf16",
    }
    return [
        _parameter(
            grouped,
            source_name="valu_item_per_ms",
            variant=variant,
            target_name=name,
            expected_unit=CalibrationUnit.ITEM_PER_MS,
        )
        for name, variant in variants.items()
    ]


def _reduction_parameters(
    grouped: dict[tuple[str, str, str], list[float]],
) -> list[CalibrationParameter]:
    parameters: list[CalibrationParameter] = []
    for width in (32, 64, 128, 256, 512, 1024):
        applicability = (float(width), float(width))
        variant = f"width{width}"
        parameters.extend(
            (
                _parameter(
                    grouped,
                    source_name="reduction_op_per_ms",
                    variant=variant,
                    target_name=CalibrationParameterName.REDUCTION_OP_PER_MS,
                    expected_unit=CalibrationUnit.ITEM_PER_MS,
                    applicability=applicability,
                    applicability_dimension=(
                        ApplicabilityDimension.REDUCTION_WIDTH
                    ),
                ),
                _parameter(
                    grouped,
                    source_name="barrier_penalty_ms",
                    variant=variant,
                    target_name=CalibrationParameterName.BARRIER_PENALTY_MS,
                    expected_unit=CalibrationUnit.MS_PER_EVENT,
                    applicability=applicability,
                    applicability_dimension=(
                        ApplicabilityDimension.REDUCTION_WIDTH
                    ),
                ),
                _parameter(
                    grouped,
                    source_name="reduction_op_per_ms",
                    variant=variant,
                    target_name=(
                        CalibrationParameterName.SOFTMAX_REDUCTION_OP_PER_MS
                    ),
                    expected_unit=CalibrationUnit.ITEM_PER_MS,
                    applicability=applicability,
                    applicability_dimension=(
                        ApplicabilityDimension.REDUCTION_WIDTH
                    ),
                ),
            )
        )
    return parameters


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "MetricSample",
    "ProbeBatch",
    "build_calibration_parameters",
    "build_calibration_surfaces",
    "freeze_probe_configuration",
    "parse_probe_metrics",
    "validate_indexed_read_surface_capacity",
]
