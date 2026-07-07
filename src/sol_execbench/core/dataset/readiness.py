# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Static ROCm readiness classification for dataset inventory sidecars."""

from __future__ import annotations

from .readiness_classifier import classify_rocm_readiness, classify_workload_readiness
from .readiness_hints import (
    BLACKWELL_LOW_PRECISION_DTYPES,
    FLASHINFER_RUNTIME_BUCKETS,
    FLASHINFER_RUNTIME_BUCKET_TO_REASON,
    FLASHINFER_RUNTIME_REFERENCE_HINTS,
    FLASHINFER_SIMPLE_REFERENCE_TOKENS,
    LOW_PRECISION_DTYPES,
    _blackwell_low_precision,  # noqa: F401
    _flashinfer_reference_is_runtime_dependent,  # noqa: F401
    _flashinfer_semantic_bucket,  # noqa: F401
    _low_precision_or_quant,  # noqa: F401
    _read_reference_text,  # noqa: F401
    _reference_has_nvidia_blocker,  # noqa: F401
    _solution_runtime_hints,  # noqa: F401
    _unsupported_dtype_failure,  # noqa: F401
)
from .readiness_io import write_dataset_readiness
from .readiness_models import (
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
from .readiness_reports import _blocker, _reason, _worst_status  # noqa: F401

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
    "classify_rocm_readiness",
    "classify_workload_readiness",
    "write_dataset_readiness",
]
