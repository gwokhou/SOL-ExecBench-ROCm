# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Paired whole-stack screening; consumes measurements, never acquires authority."""

import argparse
import math
import random
import statistics
from enum import StrEnum
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field

from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)

BLOCKS = 10
WORKLOADS = 24
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261011
IMPROVEMENT_THRESHOLD = 1.05


class MeasurementKind(StrEnum):
    """Compilation and execution must never share one comparison."""

    COLD_COMPILE = "cold_compile"
    HOT_EXECUTION = "hot_execution"


class RatioSummary(TypedDict):
    """Direction-explicit paired screening result."""

    old_over_new: float
    paired_95_percent_ci: list[float]
    verdict: str


class UpgradeComparison(TypedDict):
    """Complete comparison, retaining each preregistered workload."""

    authority: str
    ratio_direction: str
    whole_stack: RatioSummary
    workloads: dict[str, RatioSummary]
    limitations: list[str]


class PairedMeasurement(BaseModel):
    """One successful same-candidate, same-GPU paired observation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    workload: str = Field(min_length=1)
    block: int = Field(ge=0, lt=BLOCKS)
    candidate_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    measurement_kind: MeasurementKind
    gpu_uuid: str = Field(min_length=1)
    old_image: str = Field(pattern="^sha256:[0-9a-f]{64}$")
    new_image: str = Field(pattern="^sha256:[0-9a-f]{64}$")
    old_first: bool
    correct: bool
    old_ms: float = Field(gt=0)
    new_ms: float = Field(gt=0)


def summarize(log_ratios: list[float]) -> RatioSummary:
    """Bootstrap independent paired blocks; report screening uncertainty."""
    rng = random.Random(BOOTSTRAP_SEED)
    samples = sorted(
        math.exp(statistics.mean(rng.choices(log_ratios, k=BLOCKS)))
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    lower = samples[int(BOOTSTRAP_REPLICATES * 0.025)]
    upper = samples[int(BOOTSTRAP_REPLICATES * 0.975)]
    ratio = math.exp(statistics.mean(log_ratios))
    verdict = "uncertain_or_below_threshold"
    if ratio > IMPROVEMENT_THRESHOLD and lower > 1:
        verdict = "screening_improvement"
    elif ratio < 1 / IMPROVEMENT_THRESHOLD and upper < 1:
        verdict = "screening_regression"
    return {
        "old_over_new": ratio,
        "paired_95_percent_ci": [lower, upper],
        "verdict": verdict,
    }


def analyze(
    rows: list[PairedMeasurement], workload_ids: list[str]
) -> UpgradeComparison:
    """Reject incomplete, confounded or incorrect sets instead of dropping rows."""
    if len(set(workload_ids)) != WORKLOADS or len(workload_ids) != WORKLOADS:
        raise ValueError("expected exactly 24 preregistered workload IDs")
    indexed = {(row.workload, row.block): row for row in rows}
    expected = {
        (workload, block)
        for workload in workload_ids
        for block in range(BLOCKS)
    }
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("missing, duplicate or unexpected paired measurements")
    if not all(row.correct for row in rows):
        raise ValueError("correctness failure prohibits a performance verdict")
    identities = {
        (
            row.gpu_uuid,
            row.old_image,
            row.new_image,
            row.protocol_sha256,
            row.measurement_kind,
        )
        for row in rows
    }
    if len(identities) != 1 or rows[0].old_image == rows[0].new_image:
        raise ValueError("stack or GPU identity is inconsistent")
    per_workload = {}
    ratios = []
    for workload in workload_ids:
        paired = [indexed[workload, block] for block in range(BLOCKS)]
        if len({row.candidate_sha256 for row in paired}) != 1:
            raise ValueError("candidate changed between paired blocks")
        if any(row.old_first != (row.block % 2 == 0) for row in paired):
            raise ValueError(
                "paired order does not match preregistered alternation"
            )
        values = [math.log(row.old_ms / row.new_ms) for row in paired]
        ratios.append(values)
        per_workload[workload] = summarize(values)
    aggregate = [
        statistics.mean(values) for values in zip(*ratios, strict=True)
    ]
    return {
        "authority": "development_screening_only",
        "ratio_direction": "old_over_new_greater_than_one_is_faster",
        "whole_stack": summarize(aggregate),
        "workloads": per_workload,
        "limitations": [
            "No per-component causal attribution",
            "No multiple-comparison correction",
        ],
    }


def main() -> None:
    """Analyze a complete prequalified set; cold and hot inputs must be separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = load_json_value(args.input)
    rows = [PairedMeasurement.model_validate(row) for row in payload["pairs"]]
    atomic_write_json_value(args.output, analyze(rows, payload["workload_ids"]))


if __name__ == "__main__":
    main()
