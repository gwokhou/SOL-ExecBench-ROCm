# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Evaluation stability diagnostic sidecar helpers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from sol_execbench.core.data.json_utils import stable_model_checksum, stable_model_json
from sol_execbench.core.dataset.manifest import DatasetManifestChecksum


EVALUATION_STABILITY_SCHEMA_VERSION = "sol_execbench.evaluation_stability.v1"


STABILITY_STATUS_KEYS = (
    "backend_unsupported",
    "clock_unlocked",
    "gpu_contention",
    "insufficient_samples",
    "missing_timing",
    "multi_instance_interference",
    "noisy",
    "profiler_overhead_risk",
    "stable",
)


STABILITY_STATUS_PRIORITY = (
    "backend_unsupported",
    "missing_timing",
    "insufficient_samples",
    "clock_unlocked",
    "gpu_contention",
    "profiler_overhead_risk",
    "multi_instance_interference",
    "noisy",
    "stable",
)


SOURCE_CHECKSUM_KEYS = (
    "report_checksum",
    "timing_evidence_checksum",
    "checksum",
)


CLAIM_BOUNDARY_TEXT = (
    "This report is timing-quality interpretation only: not correctness "
    "authority, not score authority, not paper parity, not leaderboard "
    "authority, not native-host validation, and not new-hardware validation."
)


class StabilitySourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    path: str | None = None
    schema_version: str | None = None
    checksum: str | None = None


class RuntimeDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = 0
    min_ms: float | None = None
    max_ms: float | None = None
    mean_ms: float | None = None
    median_ms: float | None = None
    population_stddev_ms: float | None = None
    coefficient_of_variation: float | None = None
    selected_statistic: str = "median_ms"
    selected_runtime_ms: float | None = None


class StabilityWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    workload_ref: str
    backend: str | None = None
    activity_domain: str | None = None
    warmup_runs: int | None = None
    measured_repeat_count: int
    clock_locked: bool | None = None
    synchronization_policy: str
    stability_status: str
    reason_codes: list[str]
    runtime_distribution: RuntimeDistribution
    source_trace_refs: list[str] = Field(default_factory=list)


class StabilityStatusTotals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend_unsupported: int = 0
    clock_unlocked: int = 0
    gpu_contention: int = 0
    insufficient_samples: int = 0
    missing_timing: int = 0
    multi_instance_interference: int = 0
    noisy: int = 0
    profiler_overhead_risk: int = 0
    stable: int = 0

    def add(self, status: str) -> None:
        if status not in STABILITY_STATUS_KEYS:
            raise ValueError(f"Unknown stability status: {status}")
        setattr(self, status, getattr(self, status) + 1)


class StabilityClaimBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timing_quality_interpretation: bool = True
    correctness_authority: bool = False
    score_authority: bool = False
    paper_parity: bool = False
    leaderboard_authority: bool = False
    native_host_validation: bool = False
    new_hardware_validation: bool = False


class EvaluationStabilityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = EVALUATION_STABILITY_SCHEMA_VERSION
    created_at: str
    sources: list[StabilitySourceRef]
    status_totals: StabilityStatusTotals
    workloads: list[StabilityWorkload]
    claim_boundary: StabilityClaimBoundary = Field(
        default_factory=StabilityClaimBoundary
    )
    report_checksum: DatasetManifestChecksum | None = None

    def with_checksum(self) -> "EvaluationStabilityReport":
        return self.model_copy(
            update={
                "report_checksum": DatasetManifestChecksum(
                    value=stable_model_checksum(self, "report_checksum")
                )
            }
        )

    def to_json(self) -> str:
        return stable_model_json(self)
