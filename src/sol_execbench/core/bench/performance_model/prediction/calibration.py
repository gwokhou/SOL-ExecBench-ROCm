# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Calibration lookup and interpolation primitives."""

from __future__ import annotations

import math
from itertools import pairwise

from sol_execbench.core.bench.performance_model.models import (
    ApplicabilityDimension,
    CalibrationParameter,
    CalibrationParameterName,
    CalibrationSurfaceCell,
    CalibrationSurfaceName,
    CalibrationUnit,
    DiagnosticCalibrationProfile,
    ElementwiseOperationClass,
    PredictionComponent,
    TensorDType,
)
from sol_execbench.core.bench.performance_model.prediction.contracts import (
    _PredictionUnavailableError,
)

_F16_WMMA_FLOPS = 2.0 * 16.0 * 16.0 * 16.0
_DEFAULT_WAVE_SIZE = 32.0
_FP32_BYTES = 4.0


def _memory_parameter(
    working_set_bytes: float,
    calibration: DiagnosticCalibrationProfile,
) -> CalibrationParameter:
    for name in (
        CalibrationParameterName.L2_BYTE_PER_MS,
        CalibrationParameterName.L3_BYTE_PER_MS,
        CalibrationParameterName.VRAM_BYTE_PER_MS,
    ):
        parameter = calibration.parameter(name)
        if parameter is not None and _applies(parameter, working_set_bytes):
            return parameter
    raise _PredictionUnavailableError(
        "calibration_out_of_range:working_set_bytes"
    )


def _required(
    calibration: DiagnosticCalibrationProfile,
    name: CalibrationParameterName,
    *,
    coordinate: float | None = None,
) -> CalibrationParameter:
    parameter = calibration.parameter(name, coordinate)
    if parameter is None and coordinate is not None:
        parameter = _interpolated_parameter(calibration, name, coordinate)
    if parameter is None:
        raise _PredictionUnavailableError(
            f"missing_calibration_parameter:{name}"
        )
    if coordinate is not None and not _applies(parameter, coordinate):
        raise _PredictionUnavailableError(f"calibration_out_of_range:{name}")
    return parameter


def _interpolated_parameter(
    calibration: DiagnosticCalibrationProfile,
    name: CalibrationParameterName,
    coordinate: float,
) -> CalibrationParameter | None:
    """Interpolate strictly between adjacent point-calibrated parameters."""
    points = sorted(
        (
            parameter.applicability[0],
            parameter,
        )
        for parameter in calibration.parameters
        if parameter.name is name
        and parameter.applicability is not None
        and math.isclose(
            parameter.applicability[0],
            parameter.applicability[1],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    if not points or coordinate < points[0][0] or coordinate > points[-1][0]:
        return None
    pair = next(
        (
            (left, right)
            for left, right in pairwise(points)
            if left[0] < coordinate < right[0]
            and left[1].unit is right[1].unit
            and left[1].applicability_dimension
            is right[1].applicability_dimension
        ),
        None,
    )
    if pair is None:
        return None
    left, right = pair
    weight = (coordinate - left[0]) / (right[0] - left[0])
    return CalibrationParameter(
        name=name,
        value=_linear_interpolation(
            left[1].value,
            right[1].value,
            weight,
        ),
        unit=left[1].unit,
        confidence_interval=(
            _linear_interpolation(
                left[1].confidence_interval[0],
                right[1].confidence_interval[0],
                weight,
            ),
            _linear_interpolation(
                left[1].confidence_interval[1],
                right[1].confidence_interval[1],
                weight,
            ),
        ),
        applicability=(coordinate, coordinate),
        applicability_dimension=left[1].applicability_dimension,
    )


def _applies(parameter: CalibrationParameter, coordinate: float) -> bool:
    if parameter.applicability is None:
        return True
    lower, upper = parameter.applicability
    return lower <= coordinate <= upper


def _elementwise_parameter(
    dtype: TensorDType,
    operation: ElementwiseOperationClass,
) -> CalibrationParameterName:
    names = {
        (TensorDType.FLOAT32, ElementwiseOperationClass.SIMPLE): (
            CalibrationParameterName.VALU_SIMPLE_FP32_PER_MS
        ),
        (TensorDType.BFLOAT16, ElementwiseOperationClass.SIMPLE): (
            CalibrationParameterName.VALU_SIMPLE_BF16_PER_MS
        ),
        (TensorDType.FLOAT32, ElementwiseOperationClass.TRANSCENDENTAL): (
            CalibrationParameterName.VALU_TRANSCENDENTAL_FP32_PER_MS
        ),
        (TensorDType.BFLOAT16, ElementwiseOperationClass.TRANSCENDENTAL): (
            CalibrationParameterName.VALU_TRANSCENDENTAL_BF16_PER_MS
        ),
        (TensorDType.FLOAT32, ElementwiseOperationClass.COMPOSITE): (
            CalibrationParameterName.VALU_COMPOSITE_FP32_PER_MS
        ),
        (TensorDType.BFLOAT16, ElementwiseOperationClass.COMPOSITE): (
            CalibrationParameterName.VALU_COMPOSITE_BF16_PER_MS
        ),
    }
    try:
        return names[(dtype, operation)]
    except KeyError as error:
        raise _PredictionUnavailableError(
            f"unsupported_compute_dtype:{dtype}"
        ) from error


def _scaled_component(
    name: str,
    amount: float,
    parameter: CalibrationParameter,
    *,
    multiply: bool = False,
    dispatch_id: str | None = None,
) -> PredictionComponent:
    lower_parameter, upper_parameter = parameter.confidence_interval
    if multiply:
        estimate = amount * parameter.value
        lower = amount * lower_parameter
        upper = amount * upper_parameter
    else:
        estimate = amount / parameter.value
        lower = amount / upper_parameter
        upper = amount / lower_parameter
    return PredictionComponent(
        name=name,
        time_ms=estimate,
        lower_ms=min(lower, estimate),
        upper_ms=max(upper, estimate),
        dispatch_id=dispatch_id,
    )


def _apply_efficiency(
    component: PredictionComponent,
    efficiency: CalibrationParameter,
) -> PredictionComponent:
    lower_efficiency, upper_efficiency = efficiency.confidence_interval
    estimate = component.time_ms / efficiency.value
    lower = component.lower_ms / upper_efficiency
    upper = component.upper_ms / lower_efficiency
    return component.model_copy(
        update={
            "time_ms": estimate,
            "lower_ms": min(lower, estimate),
            "upper_ms": max(upper, estimate),
        },
    )


def _interpolated_overlap_cell(
    calibration: DiagnosticCalibrationProfile,
    mix: float,
    concurrent: int,
) -> CalibrationSurfaceCell:
    surface = calibration.surface(CalibrationSurfaceName.OVERLAP)
    if surface is None or surface.unit is not CalibrationUnit.RATIO:
        raise _PredictionUnavailableError(
            f"missing_calibration_surface:{CalibrationSurfaceName.OVERLAP}"
        )
    points = sorted(
        (
            cell.coordinates[ApplicabilityDimension.RESOURCE_MIX][0],
            cell,
        )
        for cell in surface.cells
        if _surface_coordinate_contains(
            cell,
            ApplicabilityDimension.CONCURRENT_DISPATCHES,
            float(concurrent),
        )
    )
    if not points or mix < points[0][0] or mix > points[-1][0]:
        raise _PredictionUnavailableError(
            f"calibration_out_of_range:{CalibrationSurfaceName.OVERLAP}"
        )
    for coordinate, cell in points:
        if math.isclose(mix, coordinate, rel_tol=0.0, abs_tol=1e-12):
            return cell
    left, right = next(
        (left, right)
        for left, right in pairwise(points)
        if left[0] < mix < right[0]
    )
    weight = (mix - left[0]) / (right[0] - left[0])
    return CalibrationSurfaceCell(
        coordinates={
            ApplicabilityDimension.RESOURCE_MIX: (mix, mix),
            ApplicabilityDimension.CONCURRENT_DISPATCHES: (
                float(concurrent),
                float(concurrent),
            ),
        },
        value=_linear_interpolation(left[1].value, right[1].value, weight),
        confidence_interval=(
            _linear_interpolation(
                left[1].confidence_interval[0],
                right[1].confidence_interval[0],
                weight,
            ),
            _linear_interpolation(
                left[1].confidence_interval[1],
                right[1].confidence_interval[1],
                weight,
            ),
        ),
    )


def _surface_coordinate_contains(
    cell: CalibrationSurfaceCell,
    dimension: ApplicabilityDimension,
    value: float,
) -> bool:
    lower, upper = cell.coordinates[dimension]
    return lower <= value <= upper


def _linear_interpolation(left: float, right: float, weight: float) -> float:
    return left + weight * (right - left)
