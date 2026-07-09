# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Static ROCm readiness classification for dataset inventory sidecars."""

from __future__ import annotations

from .classifier import classify_rocm_readiness, classify_workload_readiness
from .hints import (
    BLACKWELL_LOW_PRECISION_DTYPES,
    FLASHINFER_RUNTIME_BUCKETS,
    FLASHINFER_RUNTIME_BUCKET_TO_REASON,
    FLASHINFER_RUNTIME_REFERENCE_HINTS,
    FLASHINFER_SIMPLE_REFERENCE_TOKENS,
    LOW_PRECISION_DTYPES,
    _blackwell_low_precision,
    _flashinfer_reference_is_runtime_dependent,
    _flashinfer_semantic_bucket,
    _low_precision_or_quant,
    _read_reference_text,
    _reference_has_nvidia_blocker,
    _solution_runtime_hints,
    _unsupported_dtype_failure,
)
from .io import write_dataset_readiness
from .models import (
    READINESS_SCHEMA_VERSION,
    READINESS_SEVERITY,
    DatasetReadiness,
    DatasetReadinessClaimBoundary,
    LayeredEvidence,
    ProblemReadinessRecord,
    ReadinessBlockerReport,
    ReadinessClass,
    ReadinessReason,
    WorkloadReadinessRecord,
)
from .reports import _blocker, _reason, _worst_status

__all__ = [
    "BLACKWELL_LOW_PRECISION_DTYPES",
    "DatasetReadiness",
    "DatasetReadinessClaimBoundary",
    "FLASHINFER_RUNTIME_BUCKETS",
    "FLASHINFER_RUNTIME_BUCKET_TO_REASON",
    "FLASHINFER_RUNTIME_REFERENCE_HINTS",
    "FLASHINFER_SIMPLE_REFERENCE_TOKENS",
    "LOW_PRECISION_DTYPES",
    "LayeredEvidence",
    "ProblemReadinessRecord",
    "READINESS_SCHEMA_VERSION",
    "READINESS_SEVERITY",
    "ReadinessBlockerReport",
    "ReadinessClass",
    "ReadinessReason",
    "WorkloadReadinessRecord",
    "_blackwell_low_precision",
    "_blocker",
    "_flashinfer_reference_is_runtime_dependent",
    "_flashinfer_semantic_bucket",
    "_low_precision_or_quant",
    "_read_reference_text",
    "_reason",
    "_reference_has_nvidia_blocker",
    "_solution_runtime_hints",
    "_unsupported_dtype_failure",
    "_worst_status",
    "classify_rocm_readiness",
    "classify_workload_readiness",
    "write_dataset_readiness",
]
