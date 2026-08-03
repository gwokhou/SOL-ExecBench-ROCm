# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Canonical-sample timing evidence for diagnostic uncertainty."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.diagnostic_sidecar import (
    CurrentDiagnosticSidecarAuthority,
)
from sol_execbench.core.bench.performance_model.access_evidence import (
    AccessPatternSummary,
)
from sol_execbench.core.data.base_model import (
    StrictArtifactModel,
)
from sol_execbench.core.data.json_utils import load_jsonl_file
from sol_execbench.core.data.trace import CacheClearEvidence, Trace
from sol_execbench.core.integrity import SHA256Digest, sha256_file
from sol_execbench.core.integrity.schema_versions import (
    PERFORMANCE_TIMING_EVIDENCE_SCHEMA_VERSION,
)

TIMING_BOOTSTRAP_SEED = 20_260_729
TIMING_BOOTSTRAP_REPLICATES = 10_000
RAW_TIMING_FILENAME = "performance-timing-raw.jsonl"

_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    allow_inf_nan=False,
)


class RawPerformanceTimingRecord(StrictArtifactModel):
    """Trusted-driver timing samples before trace identity is available."""

    model_config = _MODEL_CONFIG

    workload_uuid: str = Field(min_length=1)
    input_sha256: SHA256Digest
    latency_ms: float = Field(gt=0.0)
    trial_samples_ms: list[list[float]] = Field(min_length=1)
    warmup_runs: int = Field(ge=0)
    timing_protocol: str
    access_patterns: list[AccessPatternSummary] = Field(default_factory=list)

    @model_validator(mode="after")
    def samples_are_valid(self) -> RawPerformanceTimingRecord:
        """Require every trial to contain positive samples."""
        if any(
            not samples or any(sample <= 0.0 for sample in samples)
            for samples in self.trial_samples_ms
        ):
            raise ValueError("timing trials require positive samples")
        return self


class WorkloadTimingEvidence(StrictArtifactModel):
    """Canonical samples and hierarchical-bootstrap interval for one workload."""

    model_config = _MODEL_CONFIG

    workload_uuid: str
    input_sha256: SHA256Digest
    latency_ms: float = Field(gt=0.0)
    lower_ms: float = Field(gt=0.0)
    upper_ms: float = Field(gt=0.0)
    trial_samples_ms: list[list[float]] = Field(min_length=1)
    warmup_runs: int = Field(ge=0)
    timing_protocol: str
    cache_clear: CacheClearEvidence | None = None

    @model_validator(mode="after")
    def interval_contains_latency(self) -> WorkloadTimingEvidence:
        """Require a valid uncertainty interval."""
        if not self.lower_ms <= self.latency_ms <= self.upper_ms:
            raise ValueError("timing interval does not contain latency")
        return self


class PerformanceTimingEvidenceSidecar(CurrentDiagnosticSidecarAuthority):
    """Diagnostic sample evidence from the same events as canonical Trace."""

    model_config = _MODEL_CONFIG
    current_schema_version = PERFORMANCE_TIMING_EVIDENCE_SCHEMA_VERSION

    schema_version: Literal["sol_execbench.performance_timing_evidence.v3"] = (
        PERFORMANCE_TIMING_EVIDENCE_SCHEMA_VERSION
    )
    run_id: SHA256Digest
    trace_sha256: SHA256Digest
    solution_sha256: SHA256Digest
    bootstrap_seed: Literal[20_260_729] = TIMING_BOOTSTRAP_SEED
    bootstrap_replicates: Literal[10_000] = TIMING_BOOTSTRAP_REPLICATES
    workloads: list[WorkloadTimingEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def workloads_are_unique(self) -> PerformanceTimingEvidenceSidecar:
        """Reject duplicate workload timing evidence."""
        ids = [workload.workload_uuid for workload in self.workloads]
        if len(ids) != len(set(ids)):
            raise ValueError("timing evidence repeats workload UUID")
        return self

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible sidecar data."""
        return self.model_dump(mode="json")


def build_performance_timing_evidence(
    *,
    raw_path: Path,
    trace_path: Path,
    traces: list[Trace],
    solution_sha256: str,
) -> PerformanceTimingEvidenceSidecar:
    """Bind trusted raw timing samples to canonical Trace rows."""
    raw = load_jsonl_file(RawPerformanceTimingRecord, raw_path)
    by_uuid = {record.workload_uuid: record for record in raw}
    if len(by_uuid) != len(raw):
        raise ValueError("raw timing evidence repeats workload UUID")
    expected = {
        trace.workload.uuid
        for trace in traces
        if trace.evaluation is not None
        and trace.evaluation.performance is not None
    }
    if set(by_uuid) != expected:
        raise ValueError("raw timing workload identity mismatch")
    workloads = [
        _workload_timing(trace, by_uuid[trace.workload.uuid])
        for trace in traces
        if trace.workload.uuid in by_uuid
    ]
    trace_sha256 = sha256_file(trace_path)
    return PerformanceTimingEvidenceSidecar(
        run_id=trace_sha256,
        trace_sha256=trace_sha256,
        solution_sha256=solution_sha256,
        workloads=workloads,
    )


def _workload_timing(
    trace: Trace,
    raw: RawPerformanceTimingRecord,
) -> WorkloadTimingEvidence:
    evaluation = trace.evaluation
    if evaluation is None or evaluation.performance is None:
        raise ValueError("timing evidence requires canonical performance")
    measured = evaluation.performance.latency_ms
    if not math.isclose(measured, raw.latency_ms, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("raw timing latency does not match canonical trace")
    lower, upper = hierarchical_bootstrap_interval(raw.trial_samples_ms)
    return WorkloadTimingEvidence(
        workload_uuid=raw.workload_uuid,
        input_sha256=raw.input_sha256,
        latency_ms=measured,
        lower_ms=min(lower, measured),
        upper_ms=max(upper, measured),
        trial_samples_ms=raw.trial_samples_ms,
        warmup_runs=raw.warmup_runs,
        timing_protocol=raw.timing_protocol,
        cache_clear=evaluation.performance.cache_clear,
    )


def hierarchical_bootstrap_interval(
    trials: list[list[float]],
) -> tuple[float, float]:
    """Return a deterministic hierarchical-bootstrap 95% interval."""
    if not trials or any(not trial for trial in trials):
        raise ValueError("hierarchical bootstrap requires nonempty trials")
    generator = np.random.default_rng(TIMING_BOOTSTRAP_SEED)
    draws = np.empty(TIMING_BOOTSTRAP_REPLICATES, dtype=np.float64)
    trial_count = len(trials)
    for draw_index in range(TIMING_BOOTSTRAP_REPLICATES):
        selected = generator.integers(0, trial_count, size=trial_count)
        means = [
            _resampled_mean(trials[index], generator) for index in selected
        ]
        draws[draw_index] = statistics.mean(means)
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return float(lower), float(upper)


def _resampled_mean(
    samples: list[float],
    generator: np.random.Generator,
) -> float:
    array = np.asarray(samples, dtype=np.float64)
    indices = generator.integers(0, array.size, size=array.size)
    return float(np.mean(array[indices]))


__all__ = [
    "RAW_TIMING_FILENAME",
    "TIMING_BOOTSTRAP_REPLICATES",
    "TIMING_BOOTSTRAP_SEED",
    "PerformanceTimingEvidenceSidecar",
    "RawPerformanceTimingRecord",
    "WorkloadTimingEvidence",
    "build_performance_timing_evidence",
    "hierarchical_bootstrap_interval",
]
