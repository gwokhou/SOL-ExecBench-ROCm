# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""``sol-execbench score official`` CLI (GATE-03 confirmed-evidence emitter).

Emits official score evidence only from a release-scoped baseline, its
independent rerun verification, and an explicit suite denominator. A measured
baseline registry can still add diagnostic coverage checks, but is not ``T_b``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import click

from ...core.evidence.baseline_coverage import (
    CurrentRunEnvironment,
    validate_baseline_coverage,
)
from ...core.scoring.amd_score import amd_native_suite_report_from_dict
from ...core.scoring.official_score import (
    CandidateScoreEvidence,
    OFFICIAL_AGGREGATION_POLICY,
    build_official_score_suite_evidence,
)
from ...core.integrity.checksums import sha256_file
from ...core.scoring.release_baseline import load_official_release_baseline
from ..protocol import CliFailure, CliResult, artifact


@click.group("score", context_settings={"help_option_names": ["-h", "--help"]})
def score_cli() -> None:
    """Build score evidence under explicit authority rules."""


@score_cli.command("official")
@click.option(
    "--amd-native-score",
    "amd_native_score_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="AMD-native score report JSON (sol_execbench.amd_native_score.v1).",
)
@click.option(
    "--measured-registry",
    "measured_registry_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional measured-baseline coverage registry; not a scoring baseline.",
)
@click.option(
    "--scoring-baseline",
    "scoring_baseline_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Release scoring_baseline.v1 JSON.",
)
@click.option(
    "--release-baseline-bundle",
    "release_bundle_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="release_baseline_bundle.v1 matching --scoring-baseline.",
)
@click.option(
    "--release-baseline-verification",
    "release_verification_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Passing release_baseline_verification.v1 for the bundle.",
)
@click.option(
    "--suite-manifest",
    "suite_manifest_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Canonical JSON suite denominator (list or {workloads: [...]}) .",
)
@click.option(
    "--candidate-solution",
    "candidate_solution_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Exact candidate solution file used for the measured score.",
)
@click.option(
    "--candidate-trace",
    "candidate_trace_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Canonical candidate trace JSONL cited by every scored workload.",
)
@click.option(
    "--candidate-timing-evidence",
    "candidate_timing_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Timing evidence file cited by every scored workload.",
)
@click.option("--candidate-environment-fingerprint")
@click.option("--candidate-clock-policy")
@click.option("--candidate-timing-policy")
@click.option(
    "--aggregation-policy",
    "aggregation_policy",
    required=True,
    type=click.Choice([OFFICIAL_AGGREGATION_POLICY], case_sensitive=True),
    help="Required official score aggregation policy.",
)
@click.option(
    "--current-run-env-hardware",
    "env_hardware",
    default=None,
    help="Current-run hardware (e.g. gfx1200). Omit to skip hardware comparison.",
)
@click.option(
    "--current-run-env-rocm",
    "env_rocm",
    default=None,
    help="Current-run ROCm version. Omit to skip ROCm comparison.",
)
@click.option(
    "--current-run-env-target",
    "env_target",
    default=None,
    help="Current-run target id. Omit to skip target comparison.",
)
@click.option(
    "--current-run-env-timing-policy",
    "env_timing_policy",
    default=None,
    help="Current-run timing policy. Omit to skip timing-policy comparison.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write official_score_evidence.v1 JSON to this path (default: stdout).",
)
def _official_score_cli(
    amd_native_score_path: Path,
    measured_registry_path: Path | None,
    scoring_baseline_path: Path,
    release_bundle_path: Path,
    release_verification_path: Path,
    suite_manifest_path: Path,
    candidate_solution_path: Path | None,
    candidate_trace_path: Path | None,
    candidate_timing_path: Path | None,
    candidate_environment_fingerprint: str | None,
    candidate_clock_policy: str | None,
    candidate_timing_policy: str | None,
    aggregation_policy: str,
    env_hardware: str | None,
    env_rocm: str | None,
    env_target: str | None,
    env_timing_policy: str | None,
    output_path: Path | None,
) -> CliResult:
    """Emit official_score_evidence.v1 from SOL-provided benchmark evidence."""

    evidence_values = (
        candidate_solution_path,
        candidate_trace_path,
        candidate_timing_path,
        candidate_environment_fingerprint,
        candidate_clock_policy,
        candidate_timing_policy,
    )
    if any(value is not None for value in evidence_values) and not all(
        value is not None for value in evidence_values
    ):
        raise CliFailure(
            "candidate evidence options must be supplied all together",
            code="incomplete_candidate_evidence",
            details={
                "options": [
                    "--candidate-solution",
                    "--candidate-trace",
                    "--candidate-timing-evidence",
                    "--candidate-environment-fingerprint",
                    "--candidate-clock-policy",
                    "--candidate-timing-policy",
                ]
            },
        )
    try:
        report = amd_native_suite_report_from_dict(
            json.loads(amd_native_score_path.read_text(encoding="utf-8"))
        )
        release_baseline = load_official_release_baseline(
            baseline_path=scoring_baseline_path,
            bundle_path=release_bundle_path,
            verification_path=release_verification_path,
        )
        candidate_evidence = None
        if all(
            value is not None
            for value in (
                candidate_solution_path,
                candidate_trace_path,
                candidate_timing_path,
                candidate_environment_fingerprint,
                candidate_clock_policy,
                candidate_timing_policy,
            )
        ):
            assert candidate_solution_path is not None
            assert candidate_trace_path is not None
            assert candidate_timing_path is not None
            assert candidate_environment_fingerprint is not None
            assert candidate_clock_policy is not None
            assert candidate_timing_policy is not None
            candidate_evidence = CandidateScoreEvidence(
                solution_ref=str(candidate_solution_path),
                solution_sha256=sha256_file(candidate_solution_path),
                trace_ref=str(candidate_trace_path),
                trace_sha256=sha256_file(candidate_trace_path),
                timing_ref=str(candidate_timing_path),
                timing_sha256=sha256_file(candidate_timing_path),
                environment_fingerprint=candidate_environment_fingerprint,
                clock_policy=candidate_clock_policy,
                timing_policy=candidate_timing_policy,
            )
        expected_workloads = _suite_workloads(suite_manifest_path)
        coverage_report = None
        if measured_registry_path is not None:
            registry = json.loads(measured_registry_path.read_text(encoding="utf-8"))
            current_run_env = CurrentRunEnvironment(
                hardware=env_hardware,
                rocm_version=env_rocm,
                target_id=env_target,
                timing_policy=env_timing_policy,
            )
            coverage_report = validate_baseline_coverage(
                registry, current_run_environment=current_run_env
            )
        suite = build_official_score_suite_evidence(
            report.scores,
            aggregation_policy=aggregation_policy,
            coverage_report=coverage_report,
            release_baseline=release_baseline,
            candidate_evidence=candidate_evidence,
            expected_workloads=expected_workloads,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise click.ClickException(str(exc)) from exc
    payload = suite.to_dict()
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
        click.echo(f"wrote {output_path}")
    else:
        click.echo(serialized)
    return CliResult(
        data=payload,
        artifacts=(artifact(output_path, "json_file"),) if output_path else (),
    )


def _suite_workloads(path: Path) -> tuple[tuple[str, str], ...]:
    """Load the canonical official-score denominator from a suite manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("workloads") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("suite manifest must be a list or object with workloads")
    workloads: list[tuple[str, str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"suite workload {index} must be an object")
        typed_row = cast(dict[str, Any], row)
        definition = typed_row.get("definition")
        workload_uuid = typed_row.get("workload_uuid")
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError(f"suite workload {index} has invalid definition")
        if not isinstance(workload_uuid, str) or not workload_uuid.strip():
            raise ValueError(f"suite workload {index} has invalid workload_uuid")
        workloads.append((definition, workload_uuid))
    return tuple(workloads)


_official_score_cli = score_cli
_official_score_cli_command = score_cli.commands["official"]
setattr(
    _official_score_cli_command,
    "cli_constraints",
    [
        "--amd-native-score, --scoring-baseline, and --release-baseline-bundle are required",
        "the six candidate evidence options are all-or-none",
    ],
)
