# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict parsing tests for compact scoring baseline artifacts."""

from __future__ import annotations

import pytest

from sol_execbench.core.scoring.baseline_artifact import (
    BASELINE_ARTIFACT_SCHEMA_VERSION,
    scoring_baseline_artifact_from_dict,
)


def _payload(entries: list[dict]) -> dict:
    return {
        "schema_version": BASELINE_ARTIFACT_SCHEMA_VERSION,
        "release": "v2.14",
        "source": "release_baseline_bundle",
        "entries": entries,
    }


@pytest.mark.parametrize("latency", [float("nan"), float("inf"), 0.0, -1.0])
def test_baseline_rejects_nonfinite_or_nonpositive_latency(latency: float) -> None:
    with pytest.raises(ValueError, match="latency_ms"):
        scoring_baseline_artifact_from_dict(
            _payload(
                [
                    {
                        "definition": "gemm",
                        "workload_uuid": "w1",
                        "latency_ms": latency,
                    }
                ]
            )
        )


def test_baseline_rejects_duplicate_definition_workload_key() -> None:
    with pytest.raises(ValueError, match="duplicate baseline"):
        scoring_baseline_artifact_from_dict(
            _payload(
                [
                    {"definition": "gemm", "workload_uuid": "w1", "latency_ms": 1.0},
                    {"definition": "gemm", "workload_uuid": "w1", "latency_ms": 2.0},
                ]
            )
        )
