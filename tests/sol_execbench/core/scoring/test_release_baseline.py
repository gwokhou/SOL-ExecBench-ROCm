# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Public contract tests for release baseline publication evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from sol_execbench.core.scoring.release_baseline import (
    AuthorityInput,
    RELEASE_BASELINE_BUNDLE_SCHEMA_VERSION,
    RELEASE_BASELINE_VERIFICATION_SCHEMA_VERSION,
    ReleaseBaselineBundle,
    ReleaseBaselineVerification,
    ReleaseBaselineWorkload,
    ReleaseProvenance,
    build_release_baseline_bundle,
    load_release_baseline_bundle,
    sha256_file,
    write_release_baseline_bundle,
    write_release_baseline_outputs,
    verify_release_baseline_rerun,
    write_release_baseline_verification,
)


def _provenance(tmp_path: Path) -> ReleaseProvenance:
    solution = tmp_path / "solution.py"
    solution.write_text("solution", encoding="utf-8")
    return ReleaseProvenance(
        solution=str(solution), solution_sha256=sha256_file(solution)
    )


def _passed_trace(
    definition: str, workload_uuid: str, latency: float
) -> dict[str, Any]:
    return {
        "definition": definition,
        "workload": {"uuid": workload_uuid},
        "evaluation": {
            "status": "PASSED",
            "performance": {"latency_ms": latency},
        },
    }


def _write_trace(tmp_path: Path, *traces: dict[str, Any]) -> Path:
    path = tmp_path / "trace.jsonl"
    path.write_text(
        "".join(json.dumps(trace, allow_nan=True) + "\n" for trace in traces),
        encoding="utf-8",
    )
    return path


def _verification_provenance(
    tmp_path: Path, *, solution_sha256: str | None = None
) -> ReleaseProvenance:
    provenance = _provenance(tmp_path)
    return ReleaseProvenance(
        solution=provenance.solution,
        solution_sha256=solution_sha256 or provenance.solution_sha256,
        environment_fingerprint="environment-a",
        clock_policy="locked",
        compiler_build_id="rocm-7.1",
        timing_policy="latency_ms",
        suite_manifest_sha256="a" * 64,
    )


def _official_bundle(tmp_path: Path, *, latency: float = 10.0) -> ReleaseBaselineBundle:
    return ReleaseBaselineBundle(
        release="v2.14",
        suite_manifest_ref="suite.json",
        suite_manifest_sha256="a" * 64,
        baseline_artifact_ref="baseline.json",
        baseline_artifact_sha256="b" * 64,
        provenance=_verification_provenance(tmp_path),
        workloads=(
            ReleaseBaselineWorkload(
                "gemm",
                "w1",
                "official",
                latency,
                (),
                trace_ref="baseline.jsonl",
                trace_sha256="c" * 64,
                bound_ref="bound.json",
                bound_sha256="d" * 64,
                hardware_model_ref="model.json",
                hardware_model_sha256="e" * 64,
            ),
        ),
        latency_tolerance_rel=0.05,
    )


def _rerun_trace(latency: float = 10.0, **evidence: str) -> dict[str, Any]:
    trace = _passed_trace("gemm", "w1", latency)
    trace["evaluation"]["release_baseline"] = {
        "bound_sha256": evidence.get("bound_sha256", "d" * 64),
        "hardware_model_sha256": evidence.get("hardware_model_sha256", "e" * 64),
    }
    return trace


def _build_with_trace(
    tmp_path: Path,
    trace: dict[str, Any],
    authority_by_key: dict[tuple[str, str], AuthorityInput] | None = None,
):
    return build_release_baseline_bundle(
        suite_workloads=[{"definition": "gemm", "workload_uuid": "w1"}],
        trace_path=_write_trace(tmp_path, trace),
        release="v2.14",
        provenance=_provenance(tmp_path),
        authority_by_key=authority_by_key or {},
        latency_tolerance_rel=0.05,
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


def test_builder_writes_compact_entries_and_keeps_missing_suite_rows_blocked(tmp_path):
    baseline, bundle = build_release_baseline_bundle(
        suite_workloads=[
            {"definition": "gemm", "workload_uuid": "w1"},
            {"definition": "gemm", "workload_uuid": "w2"},
        ],
        trace_path=_write_trace(tmp_path, _passed_trace("gemm", "w1", 1.25)),
        release="v2.14",
        provenance=_provenance(tmp_path),
        authority_by_key={
            ("gemm", "w1"): AuthorityInput(),
        },
        latency_tolerance_rel=0.05,
    )

    assert [(entry.definition, entry.workload_uuid) for entry in baseline.entries] == [
        ("gemm", "w1")
    ]
    assert {row.workload_uuid: row.classification for row in bundle.workloads} == {
        "w1": "official",
        "w2": "blocked",
    }
    assert bundle.workloads[1].blocker_reason_codes == ("missing_trace_record",)


@pytest.mark.parametrize("latency", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_latency_becomes_blocked_and_is_excluded_from_compact_artifact(
    tmp_path, latency
):
    baseline, bundle = _build_with_trace(tmp_path, _passed_trace("gemm", "w1", latency))

    assert baseline.entries == ()
    assert bundle.workloads[0].classification == "blocked"
    assert "invalid_baseline_latency" in bundle.workloads[0].blocker_reason_codes


def test_duplicate_and_failed_trace_records_are_blocked(tmp_path):
    failed = _passed_trace("gemm", "w2", 1.0)
    failed["evaluation"] = {"status": "RUNTIME_ERROR"}
    baseline, bundle = build_release_baseline_bundle(
        suite_workloads=[
            {"definition": "gemm", "workload_uuid": "w1"},
            {"definition": "gemm", "workload_uuid": "w2"},
        ],
        trace_path=_write_trace(
            tmp_path,
            _passed_trace("gemm", "w1", 1.0),
            _passed_trace("gemm", "w1", 1.1),
            failed,
        ),
        release="v2.14",
        provenance=_provenance(tmp_path),
        authority_by_key={},
        latency_tolerance_rel=0.05,
    )

    assert baseline.entries == ()
    assert {
        row.workload_uuid: row.blocker_reason_codes for row in bundle.workloads
    } == {
        "w1": ("duplicate_trace_record",),
        "w2": ("trace_status_not_passed",),
    }


def test_authority_degrades_measured_trace_and_preserves_evidence_refs(tmp_path):
    baseline, bundle = _build_with_trace(
        tmp_path,
        _passed_trace("gemm", "w1", 1.0),
        {
            ("gemm", "w1"): AuthorityInput(
                official_blockers=("model_not_validated",),
                bound_ref="bound.json",
                bound_sha256="a" * 64,
                hardware_model_ref="model.json",
                hardware_model_sha256="b" * 64,
            )
        },
    )

    assert baseline.entries[0].source == "release_baseline_bundle"
    assert bundle.workloads[0].classification == "derived"
    assert bundle.workloads[0].blocker_reason_codes == ("model_not_validated",)
    assert bundle.workloads[0].bound_ref == "bound.json"
    assert bundle.workloads[0].hardware_model_ref == "model.json"


def test_missing_authority_input_makes_measured_trace_derived(tmp_path):
    _, bundle = _build_with_trace(tmp_path, _passed_trace("gemm", "w1", 1.0))

    assert bundle.workloads[0].classification == "derived"
    assert bundle.workloads[0].blocker_reason_codes == ("missing_authority_input",)


def test_builder_rejects_malformed_manifest_and_unrecognized_trace_identity(tmp_path):
    with pytest.raises(ValueError, match="duplicate suite workload"):
        build_release_baseline_bundle(
            suite_workloads=[
                {"definition": "gemm", "workload_uuid": "w1"},
                {"definition": "gemm", "workload_uuid": "w1"},
            ],
            trace_path=_write_trace(tmp_path, _passed_trace("gemm", "w1", 1.0)),
            release="v2.14",
            provenance=_provenance(tmp_path),
            authority_by_key={},
            latency_tolerance_rel=0.05,
        )
    with pytest.raises(ValueError, match="outside suite"):
        _build_with_trace(tmp_path, _passed_trace("other", "w1", 1.0))


def test_writer_atomically_links_bundle_to_written_baseline_digest(tmp_path):
    baseline, bundle = _build_with_trace(
        tmp_path,
        _passed_trace("gemm", "w1", 1.0),
        {("gemm", "w1"): AuthorityInput()},
    )
    baseline_path = tmp_path / "baseline.json"
    bundle_path = tmp_path / "bundle.json"

    written = write_release_baseline_outputs(
        baseline=baseline,
        bundle=bundle,
        baseline_path=baseline_path,
        bundle_path=bundle_path,
    )

    assert written.baseline_artifact_ref == str(baseline_path)
    assert written.baseline_artifact_sha256 == sha256_file(baseline_path)
    assert load_release_baseline_bundle(bundle_path) == written


def test_writer_rejects_one_path_for_both_linked_artifacts(tmp_path):
    baseline, bundle = _build_with_trace(
        tmp_path,
        _passed_trace("gemm", "w1", 1.0),
        {("gemm", "w1"): AuthorityInput()},
    )

    with pytest.raises(ValueError, match="different paths"):
        write_release_baseline_outputs(
            baseline=baseline,
            bundle=bundle,
            baseline_path=tmp_path / "output.json",
            bundle_path=tmp_path / "output.json",
        )


def test_verifier_accepts_latency_at_relative_tolerance_boundary(tmp_path):
    bundle = _official_bundle(tmp_path, latency=10.0)

    report = verify_release_baseline_rerun(
        bundle=bundle,
        rerun_trace_path=_write_trace(tmp_path, _rerun_trace(10.5)),
        rerun_provenance=bundle.provenance,
    )

    assert report.workloads[0].classification == "official"
    assert report.workloads[0].latency_delta_rel == pytest.approx(0.05)
    assert report.workloads[0].passed is True


def test_verifier_blocks_solution_hash_mismatch_even_when_latency_matches(tmp_path):
    bundle = _official_bundle(tmp_path)

    report = verify_release_baseline_rerun(
        bundle=bundle,
        rerun_trace_path=_write_trace(tmp_path, _rerun_trace()),
        rerun_provenance=_verification_provenance(tmp_path, solution_sha256="f" * 64),
    )

    assert "solution_hash_mismatch" in report.workloads[0].blocker_reason_codes


@pytest.mark.parametrize(
    ("reason", "trace_evidence", "provenance_kwargs", "latency"),
    [
        (
            "environment_fingerprint_mismatch",
            {},
            {"environment_fingerprint": "other"},
            10.0,
        ),
        ("timing_policy_mismatch", {}, {"timing_policy": "rocprofv3"}, 10.0),
        ("bound_checksum_mismatch", {"bound_sha256": "f" * 64}, {}, 10.0),
        ("latency_outside_tolerance", {}, {}, 10.6),
    ],
)
def test_verifier_reports_stable_blocker_codes(
    tmp_path, reason, trace_evidence, provenance_kwargs, latency
):
    bundle = _official_bundle(tmp_path)
    provenance = _verification_provenance(tmp_path)
    provenance = ReleaseProvenance(**{**provenance.to_dict(), **provenance_kwargs})

    report = verify_release_baseline_rerun(
        bundle=bundle,
        rerun_trace_path=_write_trace(
            tmp_path, _rerun_trace(latency, **trace_evidence)
        ),
        rerun_provenance=provenance,
    )

    assert report.workloads[0].classification == "blocked"
    assert reason in report.workloads[0].blocker_reason_codes


def test_verifier_writes_deterministic_output(tmp_path):
    bundle = _official_bundle(tmp_path)
    report = verify_release_baseline_rerun(
        bundle=bundle,
        rerun_trace_path=_write_trace(tmp_path, _rerun_trace()),
        rerun_provenance=bundle.provenance,
    )
    first, second = tmp_path / "first.json", tmp_path / "second.json"

    write_release_baseline_verification(report, first)
    write_release_baseline_verification(report, second)

    assert first.read_bytes() == second.read_bytes()


def test_verifier_preserves_matching_derived_classification_and_reason(tmp_path):
    official = _official_bundle(tmp_path)
    derived = ReleaseBaselineWorkload(
        **{
            **official.workloads[0].__dict__,
            "classification": "derived",
            "blocker_reason_codes": ("model_not_validated",),
        }
    )
    bundle = ReleaseBaselineBundle(**{**official.__dict__, "workloads": (derived,)})

    report = verify_release_baseline_rerun(
        bundle=bundle,
        rerun_trace_path=_write_trace(tmp_path, _rerun_trace()),
        rerun_provenance=bundle.provenance,
    )

    assert report.workloads[0].classification == "derived"
    assert report.workloads[0].blocker_reason_codes == ("model_not_validated",)


@pytest.mark.parametrize(
    ("path_name", "contents", "reason"),
    [
        ("missing.jsonl", None, "rerun_trace_unavailable"),
        ("malformed.jsonl", "not json\n", "rerun_trace_malformed"),
    ],
)
def test_verifier_returns_full_blocked_report_for_unreadable_or_malformed_trace(
    tmp_path, path_name, contents, reason
):
    path = tmp_path / path_name
    if contents is not None:
        path.write_text(contents, encoding="utf-8")

    report = verify_release_baseline_rerun(
        bundle=_official_bundle(tmp_path),
        rerun_trace_path=path,
        rerun_provenance=_verification_provenance(tmp_path),
    )

    assert len(report.workloads) == 1
    assert report.workloads[0].classification == "blocked"
    assert reason in report.workloads[0].blocker_reason_codes
