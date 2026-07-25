# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Assembly of verified release statements, SOLAR indexes, and signed bundles."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.dataset.aka_corpus import AkaCorpusManifest
from sol_execbench.core.timestamps import utc_timestamp

from .release_authority import (
    ReleaseAuthority,
    load_release_authority,
    verify_signed_statement,
)
from .release_builders import artifact_reference, load_execution_plan
from .release_models import (
    ArtifactReference,
    AuthorityRole,
    BaselineStatement,
    CandidateStatement,
    ProblemRunEvidence,
    ReleaseBundle,
    ReleaseExecutionPlan,
    ReleaseRunStatement,
    RerunStatement,
    SignedStatement,
    SolarIndexStatement,
    SolarManifestEvidence,
    release_model_payload,
)
from .release_solar import verify_solar_index
from .release_traces import verify_release_run


def build_run_statement(
    plan_path: Path,
    *,
    corpus_manifest_path: Path,
    output_path: Path,
    baseline_payload_sha256: str | None = None,
) -> Path:
    """Verify one completed plan and write its unsigned authority payload."""
    output = output_path.resolve()
    _require_missing(output)
    workspace = plan_path.resolve().parents[1]
    plan = load_execution_plan(plan_path, workspace_root=workspace)
    corpus = AkaCorpusManifest.load(corpus_manifest_path)
    evidence = tuple(
        ProblemRunEvidence(
            problem_path=item.problem_path,
            definition_sha256=item.definition_sha256,
            workload_sha256=item.workload_sha256,
            implementation=item.implementation,
            trace=artifact_reference(workspace, workspace / item.trace_path),
        )
        for item in plan.problems
    )
    statement = _run_statement(
        plan,
        corpus_manifest=plan.corpus_manifest,
        environment=artifact_reference(workspace, workspace / plan.environment_path),
        problems=evidence,
        baseline_payload_sha256=baseline_payload_sha256,
    )
    verify_release_run(
        statement,
        bundle_root=workspace,
        corpus=corpus,
        require_passed=plan.role != AuthorityRole.CANDIDATE,
    )
    atomic_write_json_value(output, release_model_payload(statement))
    return output


def build_solar_index(
    workspace_root: Path,
    *,
    corpus_manifest_path: Path,
    source_revision: str,
    output_path: Path,
) -> Path:
    """Verify and index the exact per-workload formal manifest denominator."""
    workspace = workspace_root.resolve()
    output = output_path.resolve()
    _require_missing(output)
    corpus = AkaCorpusManifest.load(corpus_manifest_path)
    entries = tuple(
        SolarManifestEvidence(
            problem_path=entry.relative_problem_dir.as_posix(),
            workload_uuid=workload_uuid,
            manifest=artifact_reference(
                workspace,
                workspace
                / "solar"
                / "manifests"
                / entry.relative_problem_dir
                / workload_uuid
                / "manifest.yaml",
            ),
        )
        for entry in corpus.entries
        if entry.role == "scored"
        for workload_uuid in entry.workload_uuids
    )
    index = SolarIndexStatement(
        generated_at=utc_timestamp(),
        source_revision=source_revision,
        corpus_manifest=artifact_reference(
            workspace, workspace / "corpus" / "manifest.yaml"
        ),
        entries=entries,
    )
    verify_solar_index(index, bundle_root=workspace, corpus=corpus)
    atomic_write_json_value(output, release_model_payload(index))
    return output


def assemble_release_bundle(
    workspace_root: Path,
    *,
    corpus_manifest_path: Path,
    statement_paths: dict[AuthorityRole, Path],
    signature_paths: dict[AuthorityRole, Path],
    output_path: Path,
) -> Path:
    """Verify all four detached signatures and assemble the final bundle."""
    workspace = workspace_root.resolve()
    output = output_path.resolve()
    _require_missing(output)
    corpus = AkaCorpusManifest.load(corpus_manifest_path)
    authority = load_release_authority(corpus.path)
    signed = {
        role: _signed_statement(
            role,
            workspace=workspace,
            authority=authority,
            payload_path=statement_paths[role],
            signature_path=signature_paths[role],
        )
        for role in AuthorityRole
    }
    for statement in signed.values():
        verify_signed_statement(
            statement,
            bundle_root=workspace,
            corpus_root=corpus.authored_root,
            authority=authority,
        )
    bundle = ReleaseBundle(
        corpus_manifest=artifact_reference(
            workspace, workspace / "corpus" / "manifest.yaml"
        ),
        baseline=signed[AuthorityRole.BASELINE],
        rerun=signed[AuthorityRole.RERUN],
        candidate=signed[AuthorityRole.CANDIDATE],
        solar=signed[AuthorityRole.SOLAR],
    )
    atomic_write_json_value(output, release_model_payload(bundle))
    return output


def _run_statement(
    plan: ReleaseExecutionPlan,
    *,
    corpus_manifest: ArtifactReference,
    environment: ArtifactReference,
    problems: tuple[ProblemRunEvidence, ...],
    baseline_payload_sha256: str | None,
) -> ReleaseRunStatement:
    generated_at = utc_timestamp()
    if plan.role == AuthorityRole.BASELINE:
        return BaselineStatement(
            generated_at=generated_at,
            source_revision=plan.source_revision,
            corpus_manifest=corpus_manifest,
            environment=environment,
            problems=problems,
            baseline_id=plan.run_id,
        )
    if plan.role == AuthorityRole.RERUN:
        if baseline_payload_sha256 is None:
            raise ValueError("rerun statement requires the baseline payload digest")
        return RerunStatement(
            generated_at=generated_at,
            source_revision=plan.source_revision,
            corpus_manifest=corpus_manifest,
            environment=environment,
            problems=problems,
            baseline_payload_sha256=baseline_payload_sha256,
        )
    if plan.role == AuthorityRole.CANDIDATE:
        return CandidateStatement(
            generated_at=generated_at,
            source_revision=plan.source_revision,
            corpus_manifest=corpus_manifest,
            environment=environment,
            problems=problems,
            candidate_id=plan.run_id,
        )
    raise ValueError("SOLAR does not use a release execution plan")


def _signed_statement(
    role: AuthorityRole,
    *,
    workspace: Path,
    authority: ReleaseAuthority,
    payload_path: Path,
    signature_path: Path,
) -> SignedStatement:
    key = next(item for item in authority.keys if item.role == role)
    return SignedStatement(
        payload=artifact_reference(workspace, payload_path),
        signature=artifact_reference(workspace, signature_path),
        key_id=key.key_id,
        role=role,
    )


def _require_missing(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"release artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


__all__ = [
    "assemble_release_bundle",
    "build_run_statement",
    "build_solar_index",
]
