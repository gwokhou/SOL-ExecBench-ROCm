# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic-only artifact commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click
from rich.console import Console

from sol_execbench.cli.protocol import (
    CliExitCode,
    CliFailure,
    CliResult,
    artifact,
)
from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackBuildIdentity,
    AgentFeedbackBuildRequest,
    AgentFeedbackSidecar,
    PerformanceAcceptanceStatus,
    build_agent_feedback_sidecar,
)
from sol_execbench.core.bench.performance_model.acceptance import (
    DiagnosticAcceptanceManifest,
    DiagnosticAcceptanceResult,
    evaluate_diagnostic_acceptance,
)
from sol_execbench.core.bench.performance_model.authoring import (
    build_diagnostic_acceptance,
    fit_diagnostic_inference_profile,
    verify_diagnostic_acceptance,
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
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.solar_bridge.performance import (
    load_manifest_semantic_characterization,
)

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
@click.option("--inference-profile", type=_FILE)
@click.option("--output", type=_OUTPUT, required=True)
def performance_diagnostics_cli(
    evidence_manifest: Path,
    solar_manifest: Path,
    frontier_trace: Path | None,
    calibration_profile: Path | None,
    inference_profile: Path | None,
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
                inference_profile_path=inference_profile,
            ),
            semantic_loader=load_manifest_semantic_characterization,
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


@diagnostics_cli.command("fit-performance-inference")
@click.option("--development-corpus", type=_FILE, required=True)
@click.option("--calibration-profile", type=_FILE, required=True)
@click.option("--output", type=_OUTPUT, required=True)
def fit_performance_inference_cli(
    development_corpus: Path,
    calibration_profile: Path,
    output: Path,
) -> CliResult:
    """Freeze conformal intervals and action thresholds from development data."""
    try:
        profile = fit_diagnostic_inference_profile(
            development_corpus_path=development_corpus,
            calibration_profile_path=calibration_profile,
            semantic_loader=load_manifest_semantic_characterization,
        )
        atomic_write_json_value(output, profile.model_dump(mode="json"))
    except (OSError, ValueError) as error:
        raise CliFailure(
            str(error),
            code="performance_inference_input_invalid",
            hint="Verify development corpus hashes and calibration identity.",
        ) from error
    return CliResult(
        data={
            "families": len(profile.conformal),
            "enabled_actions": len(profile.enabled_action_codes),
        },
        artifacts=(artifact(output, "diagnostic_inference_profile_json"),),
    )


@diagnostics_cli.command("accept-performance-model")
@click.option("--development-corpus", type=_FILE, required=True)
@click.option("--held-out-corpus", type=_FILE, required=True)
@click.option("--calibration-profile", type=_FILE, required=True)
@click.option("--inference-profile", type=_FILE, required=True)
@click.option("--manifest-output", type=_OUTPUT, required=True)
@click.option("--output", type=_OUTPUT, required=True)
def accept_performance_model_cli(
    development_corpus: Path,
    held_out_corpus: Path,
    calibration_profile: Path,
    inference_profile: Path,
    manifest_output: Path,
    output: Path,
) -> CliResult:
    """Run held-out acceptance using only content-addressed case evidence."""
    try:
        manifest, result = build_diagnostic_acceptance(
            development_corpus_path=development_corpus,
            held_out_corpus_path=held_out_corpus,
            calibration_profile_path=calibration_profile,
            inference_profile_path=inference_profile,
            semantic_loader=load_manifest_semantic_characterization,
        )
        atomic_write_json_value(
            manifest_output,
            manifest.model_dump(mode="json"),
        )
        atomic_write_json_value(output, result.model_dump(mode="json"))
    except (OSError, ValueError) as error:
        raise CliFailure(
            str(error),
            code="performance_acceptance_input_invalid",
            hint="Verify disjoint corpora and all cited artifact hashes.",
        ) from error
    return CliResult(
        data={
            "accepted": result.accepted,
            "case_count": result.case_count,
        },
        artifacts=(
            artifact(manifest_output, "diagnostic_acceptance_manifest_json"),
            artifact(output, "diagnostic_acceptance_result_json"),
        ),
        exit_code=(
            CliExitCode.SUCCESS
            if result.accepted
            else CliExitCode.RESULT_FAILED
        ),
    )


@diagnostics_cli.command("agent-feedback")
@click.option("--performance-diagnostic", type=_FILE, required=True)
@click.option("--evidence-manifest", type=_FILE, required=True)
@click.option("--acceptance", type=_FILE)
@click.option(
    "--acceptance-manifest",
    "acceptance_manifest_path",
    type=_FILE,
    help=(
        "Source manifest for --acceptance. Code-changing admission requires "
        "this and all four frozen source artifacts below."
    ),
)
@click.option("--development-corpus", type=_FILE)
@click.option("--held-out-corpus", type=_FILE)
@click.option("--calibration-profile", type=_FILE)
@click.option("--inference-profile", type=_FILE)
@click.option("--output", type=_OUTPUT, required=True)
def performance_agent_feedback_cli(
    performance_diagnostic: Path,
    evidence_manifest: Path,
    acceptance: Path | None,
    acceptance_manifest_path: Path | None,
    development_corpus: Path | None,
    held_out_corpus: Path | None,
    calibration_profile: Path | None,
    inference_profile: Path | None,
    output: Path,
) -> CliResult:
    """Build Agent actions from a governed and accepted performance model."""
    try:
        feedback = _build_performance_agent_feedback(
            performance_diagnostic,
            acceptance,
            acceptance_manifest_path,
            development_corpus,
            held_out_corpus,
            calibration_profile,
            inference_profile,
            evidence_manifest,
        )
        atomic_write_json_value(output, feedback.to_dict())
    except (OSError, ValueError) as error:
        raise CliFailure(
            str(error),
            code="performance_agent_feedback_input_invalid",
            hint=(
                "Use a current diagnostic, its exact evidence manifest, and "
                "the complete frozen source bundle for an accepted report."
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


def _build_performance_agent_feedback(
    performance_diagnostic: Path,
    acceptance: Path | None,
    acceptance_manifest_path: Path | None,
    development_corpus: Path | None,
    held_out_corpus: Path | None,
    calibration_profile: Path | None,
    inference_profile: Path | None,
    evidence_manifest: Path,
) -> AgentFeedbackSidecar:
    diagnostic = load_json_file(
        PerformanceDiagnosticSidecar,
        performance_diagnostic,
    )
    manifest = load_and_verify_performance_evidence_manifest(evidence_manifest)
    acceptance_result, acceptance_sources = _load_acceptance_inputs(
        acceptance,
        acceptance_manifest_path,
        development_corpus,
        held_out_corpus,
        calibration_profile,
        inference_profile,
    )
    trace_path = _manifest_trace_path(evidence_manifest, manifest)
    traces = load_jsonl_file(Trace, trace_path)
    acceptance_status, enabled_actions = _acceptance_admission(
        acceptance_result,
        diagnostic,
        acceptance_sources,
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
    return build_agent_feedback_sidecar(
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
            performance_acceptance_status=acceptance_status,
            enabled_performance_actions=enabled_actions,
        )
    )


@dataclass(frozen=True)
class _AcceptanceSources:
    """All frozen inputs needed to reconstruct an acceptance verdict."""

    manifest: DiagnosticAcceptanceManifest
    development_corpus: Path
    held_out_corpus: Path
    calibration_profile: Path
    inference_profile: Path


def _load_acceptance_inputs(
    acceptance_path: Path | None,
    manifest_path: Path | None,
    development_corpus: Path | None,
    held_out_corpus: Path | None,
    calibration_profile: Path | None,
    inference_profile: Path | None,
) -> tuple[DiagnosticAcceptanceResult | None, _AcceptanceSources | None]:
    """Require the complete source bundle whenever acceptance is supplied."""
    source_paths = (
        manifest_path,
        development_corpus,
        held_out_corpus,
        calibration_profile,
        inference_profile,
    )
    if acceptance_path is None:
        if any(path is not None for path in source_paths):
            raise ValueError("acceptance source inputs require --acceptance")
        return None, None
    if (
        manifest_path is None
        or development_corpus is None
        or held_out_corpus is None
        or calibration_profile is None
        or inference_profile is None
    ):
        raise ValueError(
            "--acceptance requires --acceptance-manifest, "
            "--development-corpus, --held-out-corpus, "
            "--calibration-profile, and --inference-profile"
        )
    result = load_json_file(DiagnosticAcceptanceResult, acceptance_path)
    sources = _AcceptanceSources(
        manifest=load_json_file(DiagnosticAcceptanceManifest, manifest_path),
        development_corpus=development_corpus,
        held_out_corpus=held_out_corpus,
        calibration_profile=calibration_profile,
        inference_profile=inference_profile,
    )
    return result, sources


def _manifest_trace_path(
    manifest_path: Path,
    manifest: PerformanceEvidenceManifest,
) -> Path:
    trace = manifest.artifact(PerformanceEvidenceArtifactKind.TRACE)
    if trace is None:
        raise ValueError("performance evidence lacks canonical trace")
    return (manifest_path.parent / trace.path).resolve()


_ACCEPTANCE_RECOMPUTED_FIELDS = (
    "accepted",
    "case_count",
    "family_case_counts",
    "family_empirical_coverage",
    "median_absolute_percentage_error",
    "p90_absolute_percentage_error",
    "action_metrics",
    "enabled_action_codes",
    "reason_codes",
)


def _verify_acceptance_against_manifest(
    acceptance: DiagnosticAcceptanceResult,
    manifest: DiagnosticAcceptanceManifest,
) -> None:
    """Re-derive the verdict from the cited manifest and reject any drift.

    Closes the shape-not-substance and self-checksum gaps (audit
    ``acceptance.py:170`` / ``acceptance.py:217``): when the source manifest is
    available the admission no longer trusts the result's self-declared metrics
    or its self-stamped ``manifest_sha256``.
    """
    expected_manifest_sha = stable_json_checksum(
        manifest.model_dump(mode="json")
    )
    if acceptance.manifest_sha256 != expected_manifest_sha:
        raise ValueError(
            "acceptance manifest hash does not match the cited manifest",
        )
    recomputed = evaluate_diagnostic_acceptance(manifest)
    if recomputed.accepted and not recomputed.enabled_action_codes:
        raise ValueError(
            "accepted manifest enables no code-changing action; a vacuous "
            "action policy cannot be certified accepted",
        )
    for field in _ACCEPTANCE_RECOMPUTED_FIELDS:
        if getattr(recomputed, field) != getattr(acceptance, field):
            raise ValueError(
                f"acceptance result field '{field}' disagrees with the cited "
                f"manifest",
            )


def _acceptance_admission(
    acceptance: DiagnosticAcceptanceResult | None,
    diagnostic: PerformanceDiagnosticSidecar,
    sources: _AcceptanceSources | None = None,
) -> tuple[PerformanceAcceptanceStatus, frozenset[str]]:
    if acceptance is None:
        return PerformanceAcceptanceStatus.OMITTED, frozenset()
    if sources is None:
        raise ValueError("accepted actions require complete source evidence")
    if acceptance.model_version != diagnostic.model_version:
        raise ValueError("diagnostic acceptance model mismatch")
    if acceptance.model_identity != diagnostic.model_identity:
        raise ValueError("diagnostic acceptance model identity mismatch")
    if (
        diagnostic.inference_profile_sha256 is None
        or acceptance.inference_profile_sha256
        != diagnostic.inference_profile_sha256
    ):
        raise ValueError("diagnostic acceptance inference profile mismatch")
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
    _verify_acceptance_against_manifest(acceptance, sources.manifest)
    verify_diagnostic_acceptance(
        acceptance=acceptance,
        manifest=sources.manifest,
        development_corpus_path=sources.development_corpus,
        held_out_corpus_path=sources.held_out_corpus,
        calibration_profile_path=sources.calibration_profile,
        inference_profile_path=sources.inference_profile,
        semantic_loader=load_manifest_semantic_characterization,
    )
    if not acceptance.accepted:
        return PerformanceAcceptanceStatus.FAILED, frozenset()
    return PerformanceAcceptanceStatus.ACCEPTED, frozenset(
        acceptance.enabled_action_codes
    )


__all__ = [
    "accept_performance_model_cli",
    "diagnostics_cli",
    "fit_performance_inference_cli",
    "performance_agent_feedback_cli",
    "performance_diagnostics_cli",
]
