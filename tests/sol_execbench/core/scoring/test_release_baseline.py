# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic release baseline evidence contracts."""

from __future__ import annotations

import json

import pytest

from sol_execbench.core.scoring.release_baseline import (
    ReleaseBaselineBundle,
    ReleaseBaselineWorkload,
    ReleaseProvenance,
    build_release_baseline_bundle,
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


def _write_trace(tmp_path, records: list[dict]) -> object:
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def _passed_trace(definition: str, workload_uuid: str, latency: float) -> dict:
    return {
        "definition": definition,
        "workload": {"uuid": workload_uuid},
        "evaluation": {"status": "PASSED", "performance": {"latency_ms": latency}},
    }


def test_builder_writes_compact_entries_and_keeps_missing_suite_rows_blocked(
    tmp_path,
) -> None:
    baseline, bundle = build_release_baseline_bundle(
        suite_workloads=[
            {"definition": "gemm", "workload_uuid": "w1"},
            {"definition": "gemm", "workload_uuid": "w2"},
        ],
        trace_path=_write_trace(tmp_path, [_passed_trace("gemm", "w1", 1.25)]),
        release="v2.14",
        provenance=ReleaseProvenance(solution="hipblaslt", solution_sha256="c" * 64),
        authority_by_key={},
        latency_tolerance_rel=0.05,
    )

    assert [(entry.definition, entry.workload_uuid) for entry in baseline.entries] == [
        ("gemm", "w1")
    ]
    assert {row.workload_uuid: row.classification for row in bundle.workloads} == {
        "w1": "official",
        "w2": "blocked",
    }


@pytest.mark.parametrize("latency", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_latency_becomes_blocked_and_is_excluded_from_compact_artifact(
    tmp_path, latency: float
) -> None:
    baseline, bundle = build_release_baseline_bundle(
        suite_workloads=[{"definition": "gemm", "workload_uuid": "w1"}],
        trace_path=_write_trace(tmp_path, [_passed_trace("gemm", "w1", latency)]),
        release="v2.14",
        provenance=ReleaseProvenance(solution="hipblaslt", solution_sha256="c" * 64),
        authority_by_key={},
        latency_tolerance_rel=0.05,
    )

    assert baseline.entries == ()
    assert bundle.workloads[0].classification == "blocked"
    assert "invalid_baseline_latency" in bundle.workloads[0].blocker_reason_codes
