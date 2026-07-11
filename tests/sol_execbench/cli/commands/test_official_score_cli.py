# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""CLI tests for ``sol-execbench official-score`` (Phase 194-02 / GATE-03)."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.core.scoring.amd_score import (
    AMD_SCORE_CLAIM_LEVEL,
    AMD_SCORE_SCHEMA_VERSION,
    AmdNativeScore,
    AmdNativeSuiteReport,
    BoundEligibilityEvidence,
)
from sol_execbench.core.scoring.official_score import (
    BASELINE_COVERAGE_FAILED_BLOCKER,
    MISSING_BASELINE_BLOCKER,
    MISSING_SCORE_BLOCKER,
    OFFICIAL_AGGREGATION_POLICY,
    PLACEHOLDER_BASELINE_BLOCKER,
)
from sol_execbench.core.scoring.baseline_artifact import (
    ScoringBaselineArtifact,
    ScoringBaselineEntry,
)
from sol_execbench.core.scoring.release_baseline import (
    ReleaseBaselineBundle,
    ReleaseBaselineVerification,
    ReleaseBaselineVerificationWorkload,
    ReleaseBaselineWorkload,
    ReleaseProvenance,
    sha256_file,
    write_release_baseline_bundle,
    write_release_baseline_verification,
)
from sol_execbench.core.reports.reporting import CANONICAL_BENCHMARK_OUTPUT


def _score(
    *,
    baseline_source: str = "scoring_baseline",
    score: float | None = 0.75,
    baseline_latency_ms: float | None = 2.0,
    workload_uuid: str = "workload-1",
) -> AmdNativeScore:
    return AmdNativeScore(
        definition="matmul_demo",
        workload_uuid=workload_uuid,
        measured_latency_ms=1.0,
        baseline_latency_ms=baseline_latency_ms,
        sol_bound_ms=0.5,
        score=score,
        claim_level=AMD_SCORE_CLAIM_LEVEL,
        warnings=(),
        baseline_source=baseline_source,
        evidence_refs={
            "trace": "trace.jsonl#workload",
            "timing": "trace.jsonl#timing",
            "sol_bound": "bound.json#workload",
            "baseline": "baseline.json#workload",
            "hardware_model": "model.json#gfx1200",
        },
        derived_evidence_refs={},
        bound_eligibility=BoundEligibilityEvidence(
            amd_sol_status="scored",
            solar_status="scored",
            hardware_profile_state="measured",
            hardware_validation_status="validated",
            model_validation_status="validated",
            warnings=(),
        ),
    )


def _write_report(tmp_path: Path, scores: list[AmdNativeScore]) -> Path:
    report = AmdNativeSuiteReport(scores=tuple(scores))
    payload = report.to_dict()
    path = tmp_path / "amd-native-score.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_registry(
    tmp_path: Path,
    *,
    hardware: str = "gfx1200",
    timing_policy: str = "latency_ms",
    workload_key: str = "workload-1",
) -> Path:
    # A confirmed measured baseline registry entry (trace_ref points at a real
    # file so coverage validation does not classify it as stale).
    trace_ref = tmp_path / "trace.jsonl"
    trace_ref.write_text("{}\n", encoding="utf-8")
    registry = {
        "schema_version": "baseline_registry.v1",
        "sol_schema_version": "sol_execbench.measured_baseline_registry.v1",
        "generated_at": "2026-07-10T00:00:00Z",
        "target_id": "attention",
        "coverage_status": "confirmed",
        "expected_workload_keys": [workload_key],
        "source_artifact": str(trace_ref),
        "entries": [
            {
                "workload_key": workload_key,
                "workload_uuid": workload_key,
                "latency_ms": 2.0,
                "score": 0.5,
                "trace_ref": str(trace_ref),
                "source": "SOL-ExecBench-ROCm measured baseline trace",
                "provenance": {
                    "hardware": hardware,
                    "rocm_version": "7.1",
                    "sol_version": "sol-test",
                    "target_id": "attention",
                    "timing_policy": timing_policy,
                },
                "facts": {"latency_ms": 2.0},
            }
        ],
    }
    path = tmp_path / "measured-registry.json"
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return path


def _write_release_inputs(
    tmp_path: Path, report_path: Path
) -> tuple[Path, Path, Path, Path]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    scores = payload["scores"]
    baseline_path = tmp_path / "scoring-baseline.json"
    baseline = ScoringBaselineArtifact(
        entries=tuple(
            ScoringBaselineEntry(score["definition"], score["workload_uuid"], 2.0)
            for score in scores
        ),
        release="v2.14",
        source="release_baseline_bundle",
    )
    baseline_path.write_text(json.dumps(baseline.to_dict()), encoding="utf-8")
    workloads = tuple(
        ReleaseBaselineWorkload(
            score["definition"],
            score["workload_uuid"],
            "official",
            2.0,
            (),
            trace_ref="baseline.jsonl",
            trace_sha256="a" * 64,
            bound_ref="bound.json",
            bound_sha256="b" * 64,
            hardware_model_ref="model.json",
            hardware_model_sha256="c" * 64,
        )
        for score in scores
    )
    bundle_path = tmp_path / "release-bundle.json"
    bundle = ReleaseBaselineBundle(
        release="v2.14",
        suite_manifest_ref="suite.json",
        suite_manifest_sha256="d" * 64,
        baseline_artifact_ref=str(baseline_path),
        baseline_artifact_sha256=sha256_file(baseline_path),
        provenance=ReleaseProvenance(solution="hipblaslt", solution_sha256="e" * 64),
        workloads=workloads,
        latency_tolerance_rel=0.05,
    )
    write_release_baseline_bundle(bundle, bundle_path)
    verification_path = tmp_path / "release-verification.json"
    verification = ReleaseBaselineVerification(
        release="v2.14",
        bundle_ref=str(bundle_path),
        bundle_sha256=sha256_file(bundle_path),
        rerun_trace_ref="rerun.jsonl",
        rerun_trace_sha256="f" * 64,
        workloads=tuple(
            ReleaseBaselineVerificationWorkload(
                score["definition"],
                score["workload_uuid"],
                "official",
                "official",
                2.0,
                2.0,
                0.0,
                True,
                (),
            )
            for score in scores
        ),
    )
    write_release_baseline_verification(verification, verification_path)
    manifest_path = tmp_path / "suite.json"
    manifest_path.write_text(
        json.dumps(
            {
                "workloads": [
                    {
                        "definition": score["definition"],
                        "workload_uuid": score["workload_uuid"],
                    }
                    for score in scores
                ]
            }
        ),
        encoding="utf-8",
    )
    return baseline_path, bundle_path, verification_path, manifest_path


def _invoke(
    tmp_path: Path,
    *,
    report_path: Path,
    registry_path: Path,
    env_hardware: str | None = "gfx1200",
    env_timing_policy: str | None = "latency_ms",
):
    baseline_path, bundle_path, verification_path, manifest_path = (
        _write_release_inputs(tmp_path, report_path)
    )
    args = [
        "official-score",
        "--amd-native-score",
        str(report_path),
        "--measured-registry",
        str(registry_path),
        "--scoring-baseline",
        str(baseline_path),
        "--release-baseline-bundle",
        str(bundle_path),
        "--release-baseline-verification",
        str(verification_path),
        "--suite-manifest",
        str(manifest_path),
    ]
    args.extend(["--aggregation-policy", OFFICIAL_AGGREGATION_POLICY])
    if env_hardware is not None:
        args.extend(["--current-run-env-hardware", env_hardware])
    if env_timing_policy is not None:
        args.extend(["--current-run-env-timing-policy", env_timing_policy])
    args.extend(["--output", str(tmp_path / "official-score.json")])
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    return json.loads((tmp_path / "official-score.json").read_text(encoding="utf-8"))


def test_valid_run_emits_confirmed_score_with_no_blockers(tmp_path: Path) -> None:
    report_path = _write_report(tmp_path, [_score()])
    registry_path = _write_registry(tmp_path)

    payload = _invoke(
        tmp_path,
        report_path=report_path,
        registry_path=registry_path,
    )

    # GATE-03 valid case: no missing_score / missing_baseline / placeholder_baseline.
    assert payload["score_authority"] is True
    assert payload["aggregation_policy"] == OFFICIAL_AGGREGATION_POLICY
    assert payload["scored_count"] == 1
    assert payload["unscored_count"] == 0
    for blocker in (
        MISSING_SCORE_BLOCKER,
        MISSING_BASELINE_BLOCKER,
        PLACEHOLDER_BASELINE_BLOCKER,
        BASELINE_COVERAGE_FAILED_BLOCKER,
    ):
        assert blocker not in payload["blocker_summary"], blocker
    assert payload["schema_version"] == "sol_execbench.official_score_evidence.v1"
    assert payload["canonical_output"] == CANONICAL_BENCHMARK_OUTPUT


def test_missing_baseline_keeps_precise_blocker(tmp_path: Path) -> None:
    report_path = _write_report(
        tmp_path, [_score(baseline_source="missing", baseline_latency_ms=None)]
    )
    registry_path = _write_registry(tmp_path)

    payload = _invoke(
        tmp_path,
        report_path=report_path,
        registry_path=registry_path,
    )

    assert payload["score_authority"] is False
    assert MISSING_BASELINE_BLOCKER in payload["blocker_summary"]


def test_placeholder_baseline_keeps_precise_blocker(tmp_path: Path) -> None:
    report_path = _write_report(tmp_path, [_score(baseline_source="reference_latency")])
    registry_path = _write_registry(tmp_path)

    payload = _invoke(
        tmp_path,
        report_path=report_path,
        registry_path=registry_path,
    )

    assert payload["score_authority"] is False
    assert PLACEHOLDER_BASELINE_BLOCKER in payload["blocker_summary"]


def test_coverage_failure_blocks_score_and_propagates_codes(tmp_path: Path) -> None:
    report_path = _write_report(
        tmp_path, [_score(baseline_source="measured_baseline_registry")]
    )
    # Registry provenance hardware differs from the current-run env passed below.
    registry_path = _write_registry(tmp_path, hardware="gfx1200")

    payload = _invoke(
        tmp_path,
        report_path=report_path,
        registry_path=registry_path,
        env_hardware="gfx942",  # mismatch -> coverage failure
        env_timing_policy="latency_ms",
    )

    assert payload["score_authority"] is False
    assert BASELINE_COVERAGE_FAILED_BLOCKER in payload["blocker_summary"]
    assert "baseline_hardware_mismatch" in payload["blocker_summary"]


def test_placeholder_baseline_contributes_zero_to_suite_score(tmp_path: Path) -> None:
    report_path = _write_report(
        tmp_path,
        [
            _score(),
            _score(workload_uuid="workload-2", baseline_source="reference_latency"),
        ],
    )
    registry_path = _write_registry(tmp_path)

    payload = _invoke(
        tmp_path,
        report_path=report_path,
        registry_path=registry_path,
    )

    assert payload["score"] == 0.375
    assert payload["mean_score"] == 0.375
    assert payload["total_workload_count"] == 2
    assert payload["scored_count"] == 1
    assert payload["blocked_count"] == 1
    assert payload["zero_scored_count"] == 1
    assert payload["unscored_count"] == 1


def test_legacy_aggregation_policy_refuses_and_help_lists_only_policy(
    tmp_path: Path,
) -> None:
    report_path = _write_report(tmp_path, [_score()])
    registry_path = _write_registry(tmp_path)

    result = CliRunner().invoke(
        cli,
        [
            "official-score",
            "--amd-native-score",
            str(report_path),
            "--measured-registry",
            str(registry_path),
            "--aggregation-policy",
            "mean of per-workload SOL scores",
        ],
    )

    assert result.exit_code != 0
    assert OFFICIAL_AGGREGATION_POLICY in result.output


def test_missing_aggregation_policy_refuses(tmp_path: Path) -> None:
    report_path = _write_report(tmp_path, [_score()])
    registry_path = _write_registry(tmp_path)
    baseline_path, bundle_path, verification_path, manifest_path = (
        _write_release_inputs(tmp_path, report_path)
    )

    result = CliRunner().invoke(
        cli,
        [
            "official-score",
            "--amd-native-score",
            str(report_path),
            "--measured-registry",
            str(registry_path),
            "--scoring-baseline",
            str(baseline_path),
            "--release-baseline-bundle",
            str(bundle_path),
            "--release-baseline-verification",
            str(verification_path),
            "--suite-manifest",
            str(manifest_path),
        ],
    )

    assert result.exit_code != 0
    assert "aggregation-policy" in result.output.lower()


def test_report_loader_round_trips_amd_native_score_report(tmp_path: Path) -> None:
    # The from_dict loader (added in 194-02) must reconstruct a report that
    # round-trips through to_dict.
    from sol_execbench.core.scoring.amd_score import amd_native_suite_report_from_dict

    original = AmdNativeSuiteReport(scores=(_score(), _score(score=0.9)))
    payload = original.to_dict()
    restored = amd_native_suite_report_from_dict(payload)

    assert restored.schema_version == AMD_SCORE_SCHEMA_VERSION
    assert [s.score for s in restored.scores] == [0.75, 0.9]
    assert restored.to_dict() == payload
