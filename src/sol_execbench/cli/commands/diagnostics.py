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
from sol_execbench.core.bench.performance_model.lifecycle import (
    BlobStore,
    BlobStoreResolver,
    store_root,
)
from sol_execbench.core.bench.performance_model.models import (
    PerformanceDiagnosticSidecar,
)
from sol_execbench.core.bench.performance_model.publication import (
    DiagnosticPublicationProjection,
    build_diagnostic_publication_projection,
    verify_diagnostic_publication_projection,
)
from sol_execbench.core.bench.performance_model.release import (
    package_diagnostic_publication,
    verify_diagnostic_release_archive,
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
from sol_execbench.core.solar_bridge.publication import (
    project_solar_manifest,
    verify_projected_solar_manifest,
)

console = Console(stderr=True)
_FILE = click.Path(exists=True, dir_okay=False, path_type=Path)
_OUTPUT = click.Path(dir_okay=False, path_type=Path)
_OUTPUT_DIRECTORY = click.Path(file_okay=False, path_type=Path)


def _blob_resolver() -> BlobStoreResolver:
    """Return the lifecycle blob resolver bound to the configured store."""
    return BlobStoreResolver(BlobStore(store_root()))


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
            blob_resolver=_blob_resolver(),
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


@diagnostics_cli.command("build-publication-projection")
@click.option("--development-corpus", type=_FILE, required=True)
@click.option("--calibration-profile", type=_FILE, required=True)
@click.option("--source-inference-profile", type=_FILE, required=True)
@click.option("--output", type=_OUTPUT_DIRECTORY, required=True)
def build_publication_projection_cli(
    development_corpus: Path,
    calibration_profile: Path,
    source_inference_profile: Path,
    output: Path,
) -> CliResult:
    """Publish the compact, reproducible inputs of a frozen diagnostic."""
    try:
        manifest_path = build_diagnostic_publication_projection(
            development_corpus_path=development_corpus,
            calibration_profile_path=calibration_profile,
            source_inference_profile_path=source_inference_profile,
            output_root=output,
            semantic_loader=load_manifest_semantic_characterization,
            solar_projector=project_solar_manifest,
            solar_verifier=verify_projected_solar_manifest,
            blob_resolver=_blob_resolver(),
        )
        projection = load_json_file(
            DiagnosticPublicationProjection, manifest_path
        )
    except (OSError, ValueError) as error:
        raise CliFailure(
            str(error),
            code="diagnostic_publication_input_invalid",
            hint=(
                "Verify the frozen corpus, calibration audit, source "
                "inference, and every cited evidence artifact."
            ),
        ) from error
    return CliResult(
        data={
            "cases": projection.case_count,
            "uncompressed_size_bytes": projection.uncompressed_size_bytes,
            "diagnostic_only": True,
        },
        artifacts=(artifact(manifest_path, "diagnostic_publication_json"),),
    )


@diagnostics_cli.command("verify-publication-projection")
@click.option("--manifest", type=_FILE, required=True)
def verify_publication_projection_cli(manifest: Path) -> CliResult:
    """Verify a distributed compact diagnostic publication tree."""
    try:
        projection = verify_diagnostic_publication_projection(
            manifest,
            semantic_loader=load_manifest_semantic_characterization,
            solar_verifier=verify_projected_solar_manifest,
            blob_resolver=_blob_resolver(),
        )
    except (OSError, ValueError) as error:
        raise CliFailure(
            str(error),
            code="diagnostic_publication_invalid",
            hint="Restore the exact content-addressed publication tree.",
        ) from error
    return CliResult(
        data={
            "cases": projection.case_count,
            "uncompressed_size_bytes": projection.uncompressed_size_bytes,
            "verified": True,
            "diagnostic_only": True,
        },
        artifacts=(artifact(manifest, "diagnostic_publication_json"),),
    )


@diagnostics_cli.group(
    "release",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def release_cli() -> None:
    """Governed deterministic release archive packaging."""


@release_cli.command("package")
@click.option("--manifest", type=_FILE, required=True)
@click.option("--archive-output", type=_OUTPUT, required=True)
@click.option("--attestation-output", type=_OUTPUT, required=True)
@click.option(
    "--source-revision",
    required=True,
    help="Source revision the publication tree was built at.",
)
@click.option("--store-root", type=_OUTPUT_DIRECTORY)
def release_package_cli(
    manifest: Path,
    archive_output: Path,
    attestation_output: Path,
    source_revision: str,
    store_root: Path | None,
) -> CliResult:
    """Package one verified publication into a deterministic release object."""
    try:
        attestation = package_diagnostic_publication(
            manifest_path=manifest,
            archive_output=archive_output,
            attestation_output=attestation_output,
            source_revision=source_revision,
            semantic_loader=load_manifest_semantic_characterization,
            solar_verifier=verify_projected_solar_manifest,
            store_root_path=store_root,
        )
    except (OSError, ValueError) as error:
        raise CliFailure(
            str(error),
            code="diagnostic_release_input_invalid",
            hint=(
                "Verify the publication tree, its exact inventory, and that "
                "the archive output does not already exist."
            ),
        ) from error
    return CliResult(
        data={
            "release_id": attestation.release_id,
            "archive_sha256": attestation.archive.sha256,
            "archive_size_bytes": attestation.archive.size_bytes,
            "case_count": attestation.case_count,
            "diagnostic_only": True,
        },
        artifacts=(
            artifact(
                attestation_output,
                "diagnostic_release_attestation_json",
            ),
        ),
    )


@release_cli.command("verify")
@click.option("--archive", type=_FILE, required=True)
@click.option("--expected-sha256")
@click.option("--unpack-root", type=_OUTPUT_DIRECTORY)
def release_verify_cli(
    archive: Path,
    expected_sha256: str | None,
    unpack_root: Path | None,
) -> CliResult:
    """Verify a downloaded release archive against its publication contract."""
    try:
        projection = verify_diagnostic_release_archive(
            archive_path=archive,
            semantic_loader=load_manifest_semantic_characterization,
            solar_verifier=verify_projected_solar_manifest,
            expected_sha256=expected_sha256,
            unpack_root=unpack_root,
        )
    except (OSError, ValueError) as error:
        raise CliFailure(
            str(error),
            code="diagnostic_release_archive_invalid",
            hint="Restore the exact archive or supply the correct SHA-256.",
        ) from error
    return CliResult(
        data={
            "cases": projection.case_count,
            "uncompressed_size_bytes": projection.uncompressed_size_bytes,
            "verified": True,
            "diagnostic_only": True,
        },
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
            blob_resolver=_blob_resolver(),
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
        blob_resolver=_blob_resolver(),
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
    "release_cli",
    "release_package_cli",
    "release_verify_cli",
]
