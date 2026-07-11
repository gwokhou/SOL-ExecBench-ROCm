# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic release baseline evidence contracts."""

from __future__ import annotations

import pytest

from sol_execbench.core.scoring.release_baseline import (
    ReleaseBaselineBundle,
    ReleaseBaselineWorkload,
    ReleaseProvenance,
    load_release_baseline_bundle,
    write_release_baseline_bundle,
)


def _bundle_fixture() -> ReleaseBaselineBundle:
    return ReleaseBaselineBundle(
        release="v2.14-gfx1200-rocm7.1",
        suite_manifest_ref="suite.json",
        suite_manifest_sha256="a" * 64,
        baseline_artifact_ref="scoring-baseline.json",
        baseline_artifact_sha256="b" * 64,
        provenance=ReleaseProvenance(solution="hipblaslt", solution_sha256="c" * 64),
        workloads=(
            ReleaseBaselineWorkload("z", "2", "blocked", None, ("missing_baseline",)),
            ReleaseBaselineWorkload("a", "1", "derived", 1.0, ("model_not_validated",)),
        ),
        latency_tolerance_rel=0.05,
    )


def test_release_bundle_serializes_full_denominator_in_key_order() -> None:
    bundle = _bundle_fixture()

    assert [row["definition"] for row in bundle.to_dict()["workloads"]] == ["a", "z"]
    assert bundle.summary == {"total": 2, "official": 0, "derived": 1, "blocked": 1}


def test_release_workload_rejects_unknown_classification() -> None:
    with pytest.raises(ValueError, match="classification"):
        ReleaseBaselineWorkload("gemm", "w1", "provisional", 1.0, ())


def test_release_bundle_rejects_duplicate_workload_keys() -> None:
    row = ReleaseBaselineWorkload("gemm", "w1", "blocked", None, ("missing_baseline",))
    with pytest.raises(ValueError, match="duplicate workload"):
        ReleaseBaselineBundle(
            release="v2.14",
            suite_manifest_ref="suite.json",
            suite_manifest_sha256="a" * 64,
            baseline_artifact_ref="baseline.json",
            baseline_artifact_sha256="b" * 64,
            provenance=ReleaseProvenance(
                solution="hipblaslt", solution_sha256="c" * 64
            ),
            workloads=(row, row),
            latency_tolerance_rel=0.05,
        )


def test_release_bundle_json_round_trip_is_stable(tmp_path) -> None:
    bundle = _bundle_fixture()
    path = tmp_path / "bundle.json"

    write_release_baseline_bundle(bundle, path)

    assert load_release_baseline_bundle(path).to_dict() == bundle.to_dict()
