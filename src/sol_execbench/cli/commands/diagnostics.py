# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic-only artifact commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import click
from rich.console import Console

from sol_execbench.cli.error_translation import (
    CliErrorRule,
    translate_cli_errors,
)
from sol_execbench.cli.protocol import (
    CliExitCode,
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
    DiagnosticEvidencePurpose,
    DiagnosticLifecyclePlan,
    DiagnosticRunManifest,
    DiagnosticStageStatus,
    GCPlan,
    apply_gc_plan,
    build_stage_handlers,
    diagnostic_lifecycle_status,
    plan_retirement,
    repo_root,
    resolved_retirement_targets,
    resume_diagnostic_lifecycle,
    run_diagnostic_lifecycle,
    run_gc,
    run_state_path,
    store_root,
)
from sol_execbench.core.bench.performance_model.lifecycle.planning import (
    LifecyclePlanInputs,
    author_lifecycle_plan,
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
    ingest_github_published_release,
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
from sol_execbench.core.platform.rdna4_validation import (
    verify_validation_receipt,
)
from sol_execbench.core.platform.source_state import verify_git_source_state
from sol_execbench.core.solar_bridge.performance import (
    load_manifest_semantic_characterization,
)
from sol_execbench.core.solar_bridge.publication import (
    project_solar_manifest,
    verify_projected_solar_manifest,
)

_PERFORMANCE_DIAGNOSTIC_INPUT_INVALID_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="performance_diagnostic_input_invalid",
        hint="Verify the performance/SOLAR manifests, hashes, candidate, "
        "GPU identity, timing evidence, and calibration compatibility.",
    ),
)

_PERFORMANCE_INFERENCE_INPUT_INVALID_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="performance_inference_input_invalid",
        hint="Verify development corpus hashes and calibration identity.",
    ),
)

_DIAGNOSTIC_PUBLICATION_INPUT_INVALID_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="diagnostic_publication_input_invalid",
        hint="Verify the frozen corpus, calibration audit, source "
        "inference, and every cited evidence artifact.",
    ),
)

_DIAGNOSTIC_PUBLICATION_INVALID_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="diagnostic_publication_invalid",
        hint="Restore the exact content-addressed publication tree.",
    ),
)

_DIAGNOSTIC_RELEASE_INPUT_INVALID_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="diagnostic_release_input_invalid",
        hint="Verify the publication tree, its exact inventory, and that "
        "the archive output does not already exist.",
    ),
)

_DIAGNOSTIC_RELEASE_ARCHIVE_INVALID_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="diagnostic_release_archive_invalid",
        hint="Restore the exact archive or supply the correct SHA-256.",
    ),
)

_DIAGNOSTIC_PUBLISHED_RELEASE_INVALID_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, RuntimeError, ValueError),
        code="diagnostic_published_release_invalid",
        hint="Verify the fixed tag, exact two assets, and public release state.",
    ),
)

_DIAGNOSTIC_LIFECYCLE_PLAN_INVALID_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, RuntimeError, ValueError),
        code="diagnostic_lifecycle_plan_invalid",
        hint="Verify registry identities, clean source, and exact inputs.",
    ),
)

_DIAGNOSTIC_LIFECYCLE_RUN_INVALID_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="diagnostic_lifecycle_run_invalid",
        hint="Verify the design manifest, the selected chain stages, and "
        "that every earlier stage is verified.",
    ),
)

_DIAGNOSTIC_LIFECYCLE_RESUME_INVALID_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="diagnostic_lifecycle_run_invalid",
        hint="Verify the run-state object and any drifted stage inputs.",
    ),
)

_DIAGNOSTIC_LIFECYCLE_INVALID_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="diagnostic_lifecycle_invalid",
        hint="Verify the run-state object path and store layout.",
    ),
)

_DIAGNOSTIC_GC_REFUSED_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="diagnostic_gc_refused",
        hint="A blob became reachable since planning; re-run and inspect "
        "the dry-run plan before deleting.",
    ),
)

_DIAGNOSTIC_GC_APPLY_REFUSED_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="diagnostic_gc_refused",
        hint="Create and review a fresh GC plan before applying it.",
    ),
)

_DIAGNOSTIC_RETIREMENT_INVALID_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="diagnostic_retirement_invalid",
        hint="Re-run after resolving the lifecycle store and target roots.",
    ),
)

_PERFORMANCE_ACCEPTANCE_INPUT_INVALID_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="performance_acceptance_input_invalid",
        hint="Verify disjoint corpora and all cited artifact hashes.",
    ),
)

_PERFORMANCE_AGENT_FEEDBACK_INPUT_INVALID_ERRORS = (
    CliErrorRule(
        exception_type=(OSError, ValueError),
        code="performance_agent_feedback_input_invalid",
        hint="Use a current diagnostic, its exact evidence manifest, and "
        "the complete frozen source bundle for an accepted report.",
    ),
)

console = Console(stderr=True)
_FILE = click.Path(exists=True, dir_okay=False, path_type=Path)
_DIRECTORY = click.Path(exists=True, file_okay=False, path_type=Path)
_OUTPUT = click.Path(dir_okay=False, path_type=Path)
_OUTPUT_DIRECTORY = click.Path(file_okay=False, path_type=Path)


def _blob_resolver(root: Path | None = None) -> BlobStoreResolver:
    """Return the lifecycle blob resolver bound to the configured store."""
    return BlobStoreResolver(BlobStore(root or store_root()))


def _store_root_path() -> Path:
    """Return the configured lifecycle store root."""
    return store_root()


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
    with translate_cli_errors(*_PERFORMANCE_DIAGNOSTIC_INPUT_INVALID_ERRORS):
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
    with translate_cli_errors(*_PERFORMANCE_INFERENCE_INPUT_INVALID_ERRORS):
        profile = fit_diagnostic_inference_profile(
            development_corpus_path=development_corpus,
            calibration_profile_path=calibration_profile,
            semantic_loader=load_manifest_semantic_characterization,
            blob_resolver=_blob_resolver(),
        )
        atomic_write_json_value(output, profile.model_dump(mode="json"))
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
    with translate_cli_errors(*_DIAGNOSTIC_PUBLICATION_INPUT_INVALID_ERRORS):
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
    with translate_cli_errors(*_DIAGNOSTIC_PUBLICATION_INVALID_ERRORS):
        projection = verify_diagnostic_publication_projection(
            manifest,
            semantic_loader=load_manifest_semantic_characterization,
            solar_verifier=verify_projected_solar_manifest,
            blob_resolver=_blob_resolver(),
        )
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
@click.option("--hardware-validation-receipt", type=_FILE, required=True)
@click.option("--hardware-evidence-dir", type=_DIRECTORY, required=True)
@click.option(
    "--source-revision",
    required=True,
    help="Source revision the publication tree was built at.",
)
@click.option("--store-root", type=_OUTPUT_DIRECTORY)
@click.option(
    "--purpose",
    type=click.Choice([purpose.value for purpose in DiagnosticEvidencePurpose]),
    default=DiagnosticEvidencePurpose.PRODUCTION.value,
    show_default=True,
)
def release_package_cli(
    manifest: Path,
    archive_output: Path,
    attestation_output: Path,
    hardware_validation_receipt: Path,
    hardware_evidence_dir: Path,
    source_revision: str,
    store_root: Path | None,
    purpose: str,
) -> CliResult:
    """Package one verified publication into a deterministic release object."""
    with translate_cli_errors(*_DIAGNOSTIC_RELEASE_INPUT_INVALID_ERRORS):
        hardware_validation = verify_validation_receipt(
            hardware_validation_receipt,
            hardware_evidence_dir,
            expected_source_revision=source_revision,
        )
        attestation = package_diagnostic_publication(
            manifest_path=manifest,
            archive_output=archive_output,
            attestation_output=attestation_output,
            source_revision=source_revision,
            hardware_validation=hardware_validation,
            semantic_loader=load_manifest_semantic_characterization,
            solar_verifier=verify_projected_solar_manifest,
            store_root_path=store_root,
            purpose=DiagnosticEvidencePurpose(purpose),
        )
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
    with translate_cli_errors(*_DIAGNOSTIC_RELEASE_ARCHIVE_INVALID_ERRORS):
        projection = verify_diagnostic_release_archive(
            archive_path=archive,
            semantic_loader=load_manifest_semantic_characterization,
            solar_verifier=verify_projected_solar_manifest,
            expected_sha256=expected_sha256,
            unpack_root=unpack_root,
        )
    return CliResult(
        data={
            "cases": projection.case_count,
            "uncompressed_size_bytes": projection.uncompressed_size_bytes,
            "verified": True,
            "diagnostic_only": True,
        },
    )


@release_cli.command("ingest-published")
@click.option("--repository", required=True)
@click.option("--tag", required=True)
@click.option(
    "--purpose",
    type=click.Choice([purpose.value for purpose in DiagnosticEvidencePurpose]),
    required=True,
)
@click.option("--store-root", type=_OUTPUT_DIRECTORY, required=True)
def release_ingest_published_cli(
    repository: str,
    tag: str,
    purpose: str,
    store_root: Path,
) -> CliResult:
    """Download and persist one verified published release receipt."""
    with translate_cli_errors(*_DIAGNOSTIC_PUBLISHED_RELEASE_INVALID_ERRORS):
        receipt = ingest_github_published_release(
            repository=repository,
            tag=tag,
            purpose=DiagnosticEvidencePurpose(purpose),
            store_root_path=store_root,
        )
    return CliResult(data=receipt.model_dump(mode="json"))


@diagnostics_cli.group(
    "lifecycle",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def lifecycle_cli() -> None:
    """Resumable diagnostic lifecycle run, status, and resume."""


@lifecycle_cli.command("plan")
@click.option("--design-id", required=True)
@click.option("--development-snapshot-id", required=True)
@click.option("--collection-root", type=_OUTPUT_DIRECTORY, required=True)
@click.option("--held-out-corpus", type=_FILE, required=True)
@click.option("--calibration-profile", type=_FILE, required=True)
@click.option("--calibration-audit", type=_FILE, required=True)
@click.option("--output-root", type=_OUTPUT_DIRECTORY, required=True)
@click.option("--model-version", required=True)
@click.option("--vram-policy", type=_FILE)
@click.option("--frozen-inference-profile", type=_FILE)
@click.option("--hardware-validation-receipt", type=_FILE, required=True)
@click.option("--hardware-evidence-dir", type=_DIRECTORY, required=True)
@click.option("--max-attempts", type=int, default=3, show_default=True)
@click.option("--store-root", type=_OUTPUT_DIRECTORY)
@click.option("--output", type=_OUTPUT, required=True)
def lifecycle_plan_cli(**options: object) -> CliResult:
    """Author one complete production plan from verified immutable inputs."""
    store_root_path = cast(Path | None, options["store_root"])
    output = cast(Path, options["output"])
    with translate_cli_errors(*_DIAGNOSTIC_LIFECYCLE_PLAN_INVALID_ERRORS):
        plan = author_lifecycle_plan(
            repository_root=repo_root(),
            store_root=(store_root_path or _store_root_path()),
            inputs=LifecyclePlanInputs(
                design_id=cast(str, options["design_id"]),
                development_snapshot_id=cast(
                    str, options["development_snapshot_id"]
                ),
                collection_root=cast(Path, options["collection_root"]),
                held_out_corpus_path=cast(Path, options["held_out_corpus"]),
                calibration_profile_path=cast(
                    Path, options["calibration_profile"]
                ),
                calibration_audit_path=cast(Path, options["calibration_audit"]),
                output_root=cast(Path, options["output_root"]),
                model_version=cast(str, options["model_version"]),
                max_attempts=cast(int, options["max_attempts"]),
                hardware_validation_receipt_path=cast(
                    Path,
                    options["hardware_validation_receipt"],
                ),
                hardware_validation_evidence_dir=cast(
                    Path,
                    options["hardware_evidence_dir"],
                ),
                vram_policy_path=cast(Path | None, options["vram_policy"]),
                frozen_inference_profile_path=cast(
                    Path | None, options["frozen_inference_profile"]
                ),
            ),
        )
        atomic_write_json_value(output, plan.model_dump(mode="json"))
    return CliResult(data=plan.model_dump(mode="json"))


@lifecycle_cli.command("run")
@click.option("--plan", "plan_path", type=_FILE, required=True)
@click.option("--store-root", type=_OUTPUT_DIRECTORY)
def lifecycle_run_cli(
    plan_path: Path,
    store_root: Path | None,
) -> CliResult:
    """Execute one reviewed lifecycle plan and persist its run-state."""
    with translate_cli_errors(*_DIAGNOSTIC_LIFECYCLE_RUN_INVALID_ERRORS):
        plan = load_json_file(DiagnosticLifecyclePlan, plan_path)
        verify_git_source_state(
            repo_root(),
            expected_revision=plan.source_revision,
            paths=("src", "scripts", "pyproject.toml", "uv.lock"),
        )
        root = store_root.resolve() if store_root else _store_root_path()
        run_state = run_diagnostic_lifecycle(
            plan_path=plan_path,
            store_root_path=root,
            handlers=build_stage_handlers(
                semantic_loader=load_manifest_semantic_characterization,
                solar_verifier=verify_projected_solar_manifest,
                solar_projector=project_solar_manifest,
                blob_resolver=_blob_resolver(root),
            ),
        )
        _require_lifecycle_success(run_state)
    return CliResult(
        data={
            "run_id": run_state.run_id,
            "collection_run_id": run_state.collection_run_id,
            "design_id": run_state.design_id,
            "generation": run_state.generation,
            "stages": [
                {
                    "stage": item.stage.value,
                    "status": item.status.value,
                    "attempts": item.attempts,
                }
                for item in run_state.stages
            ],
        },
    )


@lifecycle_cli.command("status")
@click.option("--run-id", required=True)
@click.option("--store-root", type=_OUTPUT_DIRECTORY)
def lifecycle_status_cli(run_id: str, store_root: Path | None) -> CliResult:
    """Re-verify the recorded run and report the current chain state."""
    with translate_cli_errors(*_DIAGNOSTIC_LIFECYCLE_INVALID_ERRORS):
        status = diagnostic_lifecycle_status(
            run_state_path=run_state_path(run_id, store_root),
            handlers=build_stage_handlers(
                semantic_loader=load_manifest_semantic_characterization,
                solar_verifier=verify_projected_solar_manifest,
                solar_projector=project_solar_manifest,
                blob_resolver=_blob_resolver(store_root),
            ),
        )
    return CliResult(data=status)


@lifecycle_cli.command("resume")
@click.option("--run-id", required=True)
@click.option("--store-root", type=_OUTPUT_DIRECTORY)
def lifecycle_resume_cli(run_id: str, store_root: Path | None) -> CliResult:
    """Re-verify and continue a previously interrupted lifecycle run."""
    with translate_cli_errors(*_DIAGNOSTIC_LIFECYCLE_RESUME_INVALID_ERRORS):
        run_state = resume_diagnostic_lifecycle(
            run_state_path=run_state_path(run_id, store_root),
            handlers=build_stage_handlers(
                semantic_loader=load_manifest_semantic_characterization,
                solar_verifier=verify_projected_solar_manifest,
                solar_projector=project_solar_manifest,
                blob_resolver=_blob_resolver(store_root),
            ),
        )
        _require_lifecycle_success(run_state)
    return CliResult(
        data={
            "run_id": run_state.run_id,
            "collection_run_id": run_state.collection_run_id,
            "stages": [
                {
                    "stage": item.stage.value,
                    "status": item.status.value,
                    "attempts": item.attempts,
                }
                for item in run_state.stages
            ],
        },
    )


def _require_lifecycle_success(run_state: DiagnosticRunManifest) -> None:
    failed = [
        item
        for item in run_state.stages
        if item.status is DiagnosticStageStatus.FAILED
    ]
    if failed:
        item = failed[0]
        raise ValueError(
            f"lifecycle stage {item.stage.value} exhausted "
            f"{item.attempts} attempts for run {run_state.run_id}; "
            "inspect the append-only attempt ledger"
        )


@lifecycle_cli.group("gc")
def lifecycle_gc_cli() -> None:
    """Plan and apply registry-bound garbage collection."""


@lifecycle_gc_cli.command("plan")
@click.option("--store-root", type=_OUTPUT_DIRECTORY)
@click.option("--output", type=_OUTPUT, required=True)
def lifecycle_gc_plan_cli(
    store_root: Path | None,
    output: Path,
) -> CliResult:
    """Write a 24-hour registry-bound GC plan without deleting data."""
    with translate_cli_errors(*_DIAGNOSTIC_GC_REFUSED_ERRORS):
        plan = run_gc(store_root_path=store_root, delete=False)
        atomic_write_json_value(output, plan.model_dump(mode="json"))
    return CliResult(data=plan.model_dump(mode="json"))


@lifecycle_gc_cli.command("apply")
@click.option("--plan", "plan_path", type=_FILE, required=True)
def lifecycle_gc_apply_cli(plan_path: Path) -> CliResult:
    """Apply exactly one reviewed, unexpired GC plan."""
    with translate_cli_errors(*_DIAGNOSTIC_GC_APPLY_REFUSED_ERRORS):
        plan = load_json_file(GCPlan, plan_path)
        applied = apply_gc_plan(plan)
    return CliResult(data=applied.model_dump(mode="json"))


@lifecycle_cli.command("retirement-plan")
@click.option("--store-root", type=_OUTPUT_DIRECTORY)
def lifecycle_retirement_plan_cli(
    store_root: Path | None,
) -> CliResult:
    """Print the audit-only dry-run plan for the resolved retirement targets."""
    with translate_cli_errors(*_DIAGNOSTIC_RETIREMENT_INVALID_ERRORS):
        plan = plan_retirement(
            store_root_path=(
                store_root.resolve() if store_root else _store_root_path()
            ),
            targets=resolved_retirement_targets(repo_root()),
            repo_root=repo_root(),
        )
    return CliResult(data=plan.model_dump(mode="json"))


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
    with translate_cli_errors(*_PERFORMANCE_ACCEPTANCE_INPUT_INVALID_ERRORS):
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
    with translate_cli_errors(
        *_PERFORMANCE_AGENT_FEEDBACK_INPUT_INVALID_ERRORS
    ):
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


@dataclass(frozen=True, slots=True, kw_only=True)
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
