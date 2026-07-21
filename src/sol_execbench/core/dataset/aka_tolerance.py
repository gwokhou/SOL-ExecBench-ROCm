# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tolerance calibration helpers for AKA-derived problems.

The SOL-ExecBench paper (arXiv 2603.19173, §4.4) calibrates per-workload
correctness tolerances by probing the reference and applying a 1.25x safety
margin. For deterministic PyTorch references the dominant signal is the compute
dtype's representable error, so we provide dtype-aware defaults scaled by the
paper's margin. These defaults are used during problem authoring; a full runtime
probe can refine them per problem later.
"""

from __future__ import annotations

from typing import Mapping

from sol_execbench.core.data.workload import ToleranceSpec

# Per-dtype sane defaults (max_atol, max_rtol). FP4/FP8 kept loose because the
# reference itself quantizes; integer/bool are exact.
_DTYPE_DEFAULTS: dict[str, tuple[float, float]] = {
    "float64": (1e-9, 1e-9),
    "float32": (1e-5, 1e-5),
    "float16": (1e-3, 1e-3),
    "bfloat16": (1e-2, 1e-2),
    "float8_e4m3fn": (1e-1, 1e-1),
    "float8_e5m2": (1e-1, 1e-1),
    "float4_e2m1": (5e-1, 5e-1),
    "float4_e2m1fn_x2": (5e-1, 5e-1),
    "int64": (0.0, 0.0),
    "int32": (0.0, 0.0),
    "int16": (0.0, 0.0),
    "int8": (0.0, 0.0),
    "bool": (0.0, 0.0),
}

DEFAULT_MARGIN = 1.25
_REQUIRED_MATCHED_RATIO = 0.99
_MIN_ATOL_FLOOR = 1e-9


def dtype_default_tolerance(
    dtype: str, *, margin: float = DEFAULT_MARGIN
) -> ToleranceSpec:
    """Return a calibrated tolerance for a primary compute dtype."""
    atol, rtol = _DTYPE_DEFAULTS.get(dtype, _DTYPE_DEFAULTS["float32"])
    scaled_atol = max(_MIN_ATOL_FLOOR, atol * margin)
    scaled_rtol = rtol * margin
    return ToleranceSpec(
        max_atol=scaled_atol,
        max_rtol=scaled_rtol,
        required_matched_ratio=_REQUIRED_MATCHED_RATIO,
        max_error_cap=None,
        allow_negative_inf=False,
    )


def calibrate_tolerance(
    definition_outputs: Mapping[str, object],
    *,
    primary_dtype: str,
    margin: float = DEFAULT_MARGIN,
) -> ToleranceSpec:
    """Calibrate a workload tolerance from the problem's primary dtype.

    ``definition_outputs`` is accepted for API symmetry with a future runtime
    probe; the current implementation derives the tolerance from ``primary_dtype``.
    """
    del definition_outputs  # reserved for a future probe-based implementation
    return dtype_default_tolerance(primary_dtype, margin=margin)


__all__ = ["DEFAULT_MARGIN", "calibrate_tolerance", "dtype_default_tolerance"]
