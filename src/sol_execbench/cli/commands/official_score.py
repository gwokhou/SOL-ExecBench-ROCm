# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""``sol-execbench official-score`` CLI (GATE-03 confirmed-evidence emitter).

Emits an ``official_score_evidence.v1`` JSON artifact from SOL-provided
benchmark evidence (an AMD-native score report plus a measured baseline
registry), mirroring the agent_feedback / profile_summary sidecar-producer
pattern: input SOL artifacts -> output confirmed-evidence artifact. The score
aggregation policy is supplied explicitly via ``--aggregation-policy``, which
resolves the official score gate's previously-unresolved precondition without
adding the concept to ``AmdNativeSuiteReport``.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from ...core.evidence.baseline_coverage import (
    CurrentRunEnvironment,
    validate_baseline_coverage,
)
from ...core.scoring.amd_score import amd_native_suite_report_from_dict
from ...core.scoring.official_score import (
    OFFICIAL_AGGREGATION_POLICY,
    build_official_score_suite_evidence,
)


@click.command(
    "official-score",
    context_settings={"help_option_names": ["-h", "--help"]},
)
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
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Measured baseline registry JSON (baseline_export.py output).",
)
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
    measured_registry_path: Path,
    aggregation_policy: str,
    env_hardware: str | None,
    env_rocm: str | None,
    env_target: str | None,
    env_timing_policy: str | None,
    output_path: Path | None,
) -> None:
    """Emit official_score_evidence.v1 from SOL-provided benchmark evidence."""

    report = amd_native_suite_report_from_dict(
        json.loads(amd_native_score_path.read_text(encoding="utf-8"))
    )
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
    )
    payload = suite.to_dict()
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
        click.echo(f"wrote {output_path}")
    else:
        click.echo(serialized)
