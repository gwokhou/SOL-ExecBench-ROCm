# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic-only artifact commands."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from sol_execbench.cli.protocol import CliFailure, CliResult, artifact
from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackBuildIdentity,
    AgentFeedbackBuildRequest,
    build_agent_feedback_sidecar,
)
from sol_execbench.core.bench.performance_model.acceptance import (
    DiagnosticAcceptanceResult,
)
from sol_execbench.core.bench.performance_model.builder import (
    PerformanceDiagnosticBuildRequest,
    build_performance_diagnostic,
)
from sol_execbench.core.bench.performance_model.evidence_manifest import (
    PerformanceEvidenceArtifactKind,
    PerformanceEvidenceManifest,
    load_and_verify_performance_evidence_manifest,
)
from sol_execbench.core.bench.performance_model.governance import (
    evaluate_performance_diagnostic_governance,
    validate_performance_diagnostic_freshness,
)
from sol_execbench.core.bench.performance_model.models import (
    PerformanceDiagnosticSidecar,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
    load_jsonl_file,
)
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.integrity import sha256_file

console = Console(stderr=True)
_FILE = click.Path(exists=True, dir_okay=False, path_type=Path)
_OUTPUT = click.Path(dir_okay=False, path_type=Path)


@click.group(
    "diagnostics",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def diagnostics_cli() -> None:
    """Build governed diagnostic-only artifacts."""


@diagnostics_cli.command("performance")
@click.option("--evidence-manifest", type=_FILE, required=True)
@click.option("--solar-manifest", type=_FILE, required=True)
@click.option("--frontier-trace", type=_FILE)
@click.option("--calibration-profile", type=_FILE)
@click.option("--output", type=_OUTPUT, required=True)
def performance_diagnostics_cli(
    evidence_manifest: Path,
    solar_manifest: Path,
    frontier_trace: Path | None,
    calibration_profile: Path | None,
    output: Path,
) -> CliResult:
    """Build T_pred(IR/HW), L/C/R, and structured performance advice."""
    try:
        sidecar = build_performance_diagnostic(
            PerformanceDiagnosticBuildRequest(
                evidence_manifest_path=evidence_manifest,
                solar_manifest_path=solar_manifest,
                output_path=output,
                frontier_trace_path=frontier_trace,
                calibration_profile_path=calibration_profile,
            ),
        )
        atomic_write_json_value(output, sidecar.to_dict())
    except (OSError, ValueError) as error:
        raise CliFailure(
            str(error),
            code="performance_diagnostic_input_invalid",
            hint=(
                "Verify the performance/SOLAR manifests, hashes, candidate, "
                "GPU identity, timing evidence, and calibration compatibility."
            ),
        ) from error
    console.print(f"[green]Saved performance diagnostic to {output}[/green]")
    return CliResult(
        data={
            "status": sidecar.status,
            "workloads": len(sidecar.workloads),
            "diagnostic_only": True,
        },
        artifacts=(artifact(output, "performance_diagnostic_json"),),
    )


@diagnostics_cli.command("agent-feedback")
@click.option("--performance-diagnostic", type=_FILE, required=True)
@click.option("--evidence-manifest", type=_FILE, required=True)
@click.option("--acceptance", type=_FILE, required=True)
@click.option("--output", type=_OUTPUT, required=True)
def performance_agent_feedback_cli(
    performance_diagnostic: Path,
    evidence_manifest: Path,
    acceptance: Path,
    output: Path,
) -> CliResult:
    """Build Agent actions from a governed and accepted performance model."""
    try:
        diagnostic = load_json_file(
            PerformanceDiagnosticSidecar,
            performance_diagnostic,
        )
        manifest = load_and_verify_performance_evidence_manifest(
            evidence_manifest
        )
        acceptance_result = load_json_file(
            DiagnosticAcceptanceResult,
            acceptance,
        )
        trace_path = _manifest_trace_path(evidence_manifest, manifest)
        traces = load_jsonl_file(Trace, trace_path)
        _require_accepted_model(
            acceptance_result,
            diagnostic,
        )
        freshness = validate_performance_diagnostic_freshness(
            diagnostic,
            run_id=manifest.identity.run_id,
            candidate_sha256=manifest.identity.candidate_sha256,
            gpu_architecture=manifest.identity.gpu_architecture,
            trace_sha256=sha256_file(trace_path),
        )
        governance = evaluate_performance_diagnostic_governance(
            sidecar=diagnostic,
            freshness=freshness,
        )
        feedback = build_agent_feedback_sidecar(
            AgentFeedbackBuildRequest(
                traces=traces,
                identity=AgentFeedbackBuildIdentity(
                    trace_path=str(trace_path),
                    run_id=manifest.identity.run_id,
                    candidate_id=manifest.identity.candidate_sha256,
                    source_sha256=sha256_file(performance_diagnostic),
                ),
                performance_diagnostic=diagnostic,
                performance_governance=governance,
            )
        )
        atomic_write_json_value(output, feedback.to_dict())
    except (OSError, ValueError) as error:
        raise CliFailure(
            str(error),
            code="performance_agent_feedback_input_invalid",
            hint=(
                "Use a current diagnostic, its exact evidence manifest, and "
                "an accepted v2 model report for the same calibration."
            ),
        ) from error
    console.print(
        f"[green]Saved performance Agent feedback to {output}[/green]"
    )
    return CliResult(
        data={
            "status": feedback.status,
            "actions": len(feedback.items),
            "diagnostic_only": True,
        },
        artifacts=(artifact(output, "agent_feedback_json"),),
    )


def _manifest_trace_path(
    manifest_path: Path,
    manifest: PerformanceEvidenceManifest,
) -> Path:
    trace = manifest.artifact(PerformanceEvidenceArtifactKind.TRACE)
    if trace is None:
        raise ValueError("performance evidence lacks canonical trace")
    return (manifest_path.parent / trace.path).resolve()


def _require_accepted_model(
    acceptance: DiagnosticAcceptanceResult,
    diagnostic: PerformanceDiagnosticSidecar,
) -> None:
    if not acceptance.accepted:
        raise ValueError("diagnostic model acceptance failed")
    if acceptance.model_version != diagnostic.model_version:
        raise ValueError("diagnostic acceptance model mismatch")
    if diagnostic.calibration_identity != acceptance.calibration_identity:
        raise ValueError("diagnostic acceptance calibration mismatch")
    calibration_refs = [
        reference
        for reference in diagnostic.evidence
        if reference.kind == "calibration_profile"
    ]
    if (
        len(calibration_refs) != 1
        or calibration_refs[0].sha256 != acceptance.calibration_profile_sha256
    ):
        raise ValueError("diagnostic acceptance profile hash mismatch")


__all__ = [
    "diagnostics_cli",
    "performance_agent_feedback_cli",
    "performance_diagnostics_cli",
]
