# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Evidence contracts and optional ROCm Compute Profiler support."""

from sol_execbench.core.scoring.hardware_calibration.models import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationCandidate,
    CalibrationIsaValidation,
    HardwareCalibrationArtifact,
    hardware_calibration_artifact_from_dict,
)
from sol_execbench.core.scoring.hardware_calibration.rocprof_compute import (
    ensure_profiler_environment,
    parse_roofline_metrics,
    run_rocprof_compute_bench_only,
)
from sol_execbench.core.scoring.hardware_calibration.statistics import (
    select_conservative_value,
)
from sol_execbench.core.scoring.hardware_calibration.shape_aware_roofline import (
    SHAPE_AWARE_ROOFLINE_SCHEMA_VERSION,
    ShapeAwareRooflineArtifact,
    ShapeAwareRooflineCase,
    shape_aware_roofline_from_dict,
)

__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationCandidate",
    "CalibrationIsaValidation",
    "HardwareCalibrationArtifact",
    "ensure_profiler_environment",
    "hardware_calibration_artifact_from_dict",
    "parse_roofline_metrics",
    "SHAPE_AWARE_ROOFLINE_SCHEMA_VERSION",
    "ShapeAwareRooflineArtifact",
    "ShapeAwareRooflineCase",
    "shape_aware_roofline_from_dict",
    "run_rocprof_compute_bench_only",
    "select_conservative_value",
]
