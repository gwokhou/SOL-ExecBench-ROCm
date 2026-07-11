# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Public contract tests for release baseline publication evidence."""

from __future__ import annotations

import hashlib

import pytest

from sol_execbench.core.scoring.release_baseline import (
    RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION,
    RELEASE_BASELINE_VERIFICATION_SCHEMA_VERSION,
    ReleaseBaselineBundle,
    ReleaseBaselineVerification,
    ReleaseBaselineWorkload,
    ReleaseProvenance,
    load_release_baseline_bundle,
    sha256_file,
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


def test_release_bundle_serializes_full_denominator_in_key_order():
    bundle = _bundle_fixture()

    assert [row["definition"] for row in bundle.to_dict()["workloads"]] == ["a", "z"]
    assert bundle.summary == {"total": 2, "official": 0, "derived": 1, "blocked": 1}
    assert bundle.to_dict()["schema_version"] == RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION


def test_release_workload_rejects_unknown_classification():
    with pytest.raises(ValueError, match="classification"):
        ReleaseBaselineWorkload("gemm", "w1", "provisional", 1.0, ())


def test_release_bundle_rejects_duplicate_workload_keys():
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


def test_release_bundle_json_round_trip_is_stable(tmp_path):
    bundle = _bundle_fixture()
    path = tmp_path / "bundle.json"

    write_release_baseline_bundle(bundle, path)

    assert load_release_baseline_bundle(path).to_dict() == bundle.to_dict()


def test_release_contract_rejects_invalid_identity_digest_and_latency():
    with pytest.raises(ValueError, match="definition"):
        ReleaseBaselineWorkload("", "w1", "official", 1.0, ())
    with pytest.raises(ValueError, match="sha256"):
        ReleaseProvenance(solution="hipblaslt", solution_sha256="A" * 64)
    with pytest.raises(ValueError, match="latency_ms"):
        ReleaseBaselineWorkload("gemm", "w1", "official", float("inf"), ())
    with pytest.raises(ValueError, match="latency_tolerance_rel"):
        ReleaseBaselineBundle(
            release="v2.14",
            suite_manifest_ref="suite.json",
            suite_manifest_sha256="a" * 64,
            baseline_artifact_ref="baseline.json",
            baseline_artifact_sha256="b" * 64,
            provenance=ReleaseProvenance(
                solution="hipblaslt", solution_sha256="c" * 64
            ),
            workloads=(),
            latency_tolerance_rel=0.0,
        )


def test_release_bundle_includes_optional_immutable_evidence_refs():
    row = ReleaseBaselineWorkload(
        "gemm",
        "w1",
        "official",
        1.0,
        (),
        trace_ref="trace.json",
        trace_sha256="d" * 64,
        bound_ref="bound.json",
        bound_sha256="e" * 64,
        hardware_model_ref="model.json",
        hardware_model_sha256="f" * 64,
    )
    payload = ReleaseBaselineBundle(
        release="v2.14",
        suite_manifest_ref="suite.json",
        suite_manifest_sha256="a" * 64,
        baseline_artifact_ref="baseline.json",
        baseline_artifact_sha256="b" * 64,
        provenance=ReleaseProvenance(solution="hipblaslt", solution_sha256="c" * 64),
        workloads=(row,),
        latency_tolerance_rel=0.05,
    ).to_dict()

    assert payload["workloads"] == [
        {
            "definition": "gemm",
            "workload_uuid": "w1",
            "classification": "official",
            "latency_ms": 1.0,
            "blocker_reason_codes": [],
            "trace_ref": "trace.json",
            "trace_sha256": "d" * 64,
            "bound_ref": "bound.json",
            "bound_sha256": "e" * 64,
            "hardware_model_ref": "model.json",
            "hardware_model_sha256": "f" * 64,
        }
    ]


def test_release_verification_and_sha256_file_are_deterministic(tmp_path):
    path = tmp_path / "artifact.txt"
    path.write_text("release evidence", encoding="utf-8")
    verification = ReleaseBaselineVerification(
        release="v2.14",
        bundle_ref="bundle.json",
        bundle_sha256="a" * 64,
        rerun_trace_ref="rerun.json",
        rerun_trace_sha256="b" * 64,
    )

    assert (
        verification.to_dict()["schema_version"]
        == RELEASE_BASELINE_VERIFICATION_SCHEMA_VERSION
    )
    assert sha256_file(path) == hashlib.sha256(b"release evidence").hexdigest()
