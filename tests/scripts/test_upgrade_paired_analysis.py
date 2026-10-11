# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""A failed or incomplete workload must not disappear from upgrade comparisons."""

import pytest

from scripts.internal.upgrade.paired_analysis import (
    MeasurementKind,
    PairedMeasurement,
    analyze,
)


@pytest.fixture
def paired_rows():
    ids = [f"workload-{index}" for index in range(24)]
    rows = [
        PairedMeasurement(
            workload=workload,
            block=block,
            candidate_sha256="a" * 64,
            protocol_sha256="d" * 64,
            measurement_kind=MeasurementKind.HOT_EXECUTION,
            gpu_uuid="gpu",
            old_image="sha256:" + "b" * 64,
            new_image="sha256:" + "c" * 64,
            old_first=block % 2 == 0,
            correct=True,
            old_ms=2,
            new_ms=1,
        )
        for workload in ids
        for block in range(10)
    ]
    return rows, ids


def test_paired_ratio_direction_and_equal_workload_aggregate(paired_rows):
    rows, ids = paired_rows
    report = analyze(rows, ids)
    aggregate = report["whole_stack"]
    assert aggregate["old_over_new"] == pytest.approx(2)
    assert aggregate["verdict"] == "screening_improvement"


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "incorrect", "identity", "order"],
    ids=["missing", "duplicate", "incorrect", "identity", "order"],
)
def test_invalid_pairs_cannot_produce_a_speedup(paired_rows, mutation):
    rows, ids = paired_rows
    match mutation:
        case "missing":
            rows.pop()
        case "duplicate":
            rows.append(rows[0])
        case "incorrect":
            rows[0] = rows[0].model_copy(update={"correct": False})
        case "identity":
            rows[0] = rows[0].model_copy(update={"gpu_uuid": "other"})
        case "order":
            rows[0] = rows[0].model_copy(update={"old_first": False})
    with pytest.raises(ValueError, match="paired|correctness|identity"):
        analyze(rows, ids)
