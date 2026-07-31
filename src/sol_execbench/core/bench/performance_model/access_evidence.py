# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Candidate-independent access-pattern evidence for indexed workloads."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import torch
from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.diagnostic_sidecar import (
    CurrentDiagnosticSidecarAuthority,
    DiagnosticSidecarStatus,
)
from sol_execbench.core.data.base_model import StrictArtifactModel
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.integrity import SHA256Digest, sha256_file
from sol_execbench.core.integrity.schema_versions import (
    PERFORMANCE_ACCESS_EVIDENCE_SCHEMA_VERSION,
)

MAX_EXACT_INDEX_ELEMENTS = 4 * 1024 * 1024
_HISTOGRAM_KEYS = ("1", "2-3", "4-7", "8-15", "16-31", "32+")
_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class AccessPatternSummary(StrictArtifactModel):
    """De-identified locality and collision statistics for one integer input."""

    model_config = _CONFIG

    input_name: str = Field(min_length=1)
    dtype: Literal[DType.INT32, DType.INT64]
    element_count: int = Field(ge=1)
    sampled_element_count: int = Field(ge=1)
    exact: bool
    minimum_index: int
    maximum_index: int
    unique_index_count: int = Field(ge=1)
    duplicate_fraction: float = Field(ge=0.0, le=1.0)
    maximum_multiplicity: int = Field(ge=1)
    multiplicity_histogram: dict[str, int]
    adjacent_same_fraction: float = Field(ge=0.0, le=1.0)
    adjacent_unit_stride_fraction: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def summary_is_consistent(self) -> AccessPatternSummary:
        """Reject impossible sample sizes and histogram vocabularies."""
        if self.sampled_element_count > self.element_count:
            raise ValueError("access sample exceeds input element count")
        if self.exact != (self.sampled_element_count == self.element_count):
            raise ValueError("access exact flag contradicts sample size")
        if set(self.multiplicity_histogram) != set(_HISTOGRAM_KEYS):
            raise ValueError("access multiplicity histogram keys are invalid")
        if self.minimum_index > self.maximum_index:
            raise ValueError("access index bounds are reversed")
        return self


class WorkloadAccessEvidence(StrictArtifactModel):
    """Access summaries bound to one canonical workload input."""

    model_config = _CONFIG

    workload_uuid: str = Field(min_length=1)
    canonical_input_sha256: SHA256Digest
    patterns: list[AccessPatternSummary] = Field(default_factory=list)

    @model_validator(mode="after")
    def patterns_are_unique(self) -> WorkloadAccessEvidence:
        """Reject duplicate summaries for the same named input."""
        names = [pattern.input_name for pattern in self.patterns]
        if len(names) != len(set(names)):
            raise ValueError("access evidence repeats input name")
        return self


class PerformanceAccessEvidenceSidecar(CurrentDiagnosticSidecarAuthority):
    """Content-bound access summaries generated before candidate timing."""

    model_config = _CONFIG
    current_schema_version = PERFORMANCE_ACCESS_EVIDENCE_SCHEMA_VERSION

    schema_version: Literal["sol_execbench.performance_access_evidence.v1"] = (
        PERFORMANCE_ACCESS_EVIDENCE_SCHEMA_VERSION
    )
    status: DiagnosticSidecarStatus
    run_id: SHA256Digest
    trace_sha256: SHA256Digest
    workloads: list[WorkloadAccessEvidence] = Field(min_length=1)
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_matches_workloads(self) -> PerformanceAccessEvidenceSidecar:
        """Require unique workloads and explicit partial reasons."""
        workload_ids = [workload.workload_uuid for workload in self.workloads]
        if len(workload_ids) != len(set(workload_ids)):
            raise ValueError("access evidence repeats workload UUID")
        if (
            self.status is DiagnosticSidecarStatus.AVAILABLE
            and self.reason_codes
        ):
            raise ValueError("available access evidence cannot carry reasons")
        if (
            self.status is not DiagnosticSidecarStatus.AVAILABLE
            and not self.reason_codes
        ):
            raise ValueError("partial access evidence requires reasons")
        return self

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible sidecar data."""
        return self.model_dump(mode="json")


def build_performance_access_evidence(
    *,
    trace_path: Path,
    workloads: Sequence[WorkloadAccessEvidence],
) -> PerformanceAccessEvidenceSidecar:
    """Bind trusted access summaries to the canonical trace artifact."""
    trace_sha256 = sha256_file(trace_path)
    return PerformanceAccessEvidenceSidecar(
        status=DiagnosticSidecarStatus.AVAILABLE,
        run_id=trace_sha256,
        trace_sha256=trace_sha256,
        workloads=list(workloads),
        reason_codes=[],
    )


def summarize_integer_inputs(
    inputs: Mapping[str, object],
) -> list[AccessPatternSummary]:
    """Summarize INT32/INT64 tensors without retaining their raw values."""
    summaries: list[AccessPatternSummary] = []
    for name, value in inputs.items():
        if not isinstance(value, torch.Tensor) or value.dtype not in {
            torch.int32,
            torch.int64,
        }:
            continue
        summaries.append(_summarize_tensor(name, value))
    return summaries


def _summarize_tensor(name: str, value: torch.Tensor) -> AccessPatternSummary:
    flat = value.detach().reshape(-1)
    element_count = flat.numel()
    if element_count <= 0:
        raise ValueError(f"integer access input is empty: {name}")
    stride = max(1, math.ceil(element_count / MAX_EXACT_INDEX_ELEMENTS))
    sample = flat[::stride][:MAX_EXACT_INDEX_ELEMENTS].to(device="cpu")
    unique, counts = torch.unique(sample, sorted=False, return_counts=True)
    count_values = counts.to(dtype=torch.int64)
    adjacent = sample[1:] - sample[:-1]
    sampled = sample.numel()
    return AccessPatternSummary(
        input_name=name,
        dtype=DType.INT32 if value.dtype is torch.int32 else DType.INT64,
        element_count=element_count,
        sampled_element_count=sampled,
        exact=sampled == element_count,
        minimum_index=int(sample.min().item()),
        maximum_index=int(sample.max().item()),
        unique_index_count=unique.numel(),
        duplicate_fraction=1.0 - unique.numel() / sampled,
        maximum_multiplicity=int(count_values.max().item()),
        multiplicity_histogram=_multiplicity_histogram(count_values),
        adjacent_same_fraction=_fraction(adjacent == 0),
        adjacent_unit_stride_fraction=_fraction(adjacent.abs() == 1),
    )


def _multiplicity_histogram(counts: torch.Tensor) -> dict[str, int]:
    return {
        "1": int((counts == 1).sum().item()),
        "2-3": int(((counts >= 2) & (counts <= 3)).sum().item()),
        "4-7": int(((counts >= 4) & (counts <= 7)).sum().item()),
        "8-15": int(((counts >= 8) & (counts <= 15)).sum().item()),
        "16-31": int(((counts >= 16) & (counts <= 31)).sum().item()),
        "32+": int((counts >= 32).sum().item()),
    }


def _fraction(mask: torch.Tensor) -> float:
    return (
        float(mask.to(dtype=torch.float64).mean().item())
        if mask.numel()
        else 0.0
    )


__all__ = [
    "MAX_EXACT_INDEX_ELEMENTS",
    "AccessPatternSummary",
    "PerformanceAccessEvidenceSidecar",
    "WorkloadAccessEvidence",
    "build_performance_access_evidence",
    "summarize_integer_inputs",
]
