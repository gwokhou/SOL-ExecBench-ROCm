# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Assembly of verified statements, SOLAR indexes, and release bundles."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.dataset.aka_corpus import AkaCorpusManifest
from sol_execbench.core.integrity import verify_artifact_file
from sol_execbench.core.timestamps import utc_timestamp

from .release_builders import artifact_reference, load_execution_plan
from .release_models import (
    ArtifactReference,
    BaselineStatement,
    CandidateStatement,
    MAX_RELEASE_STATEMENT_BYTES,
    ProblemRunEvidence,
    ReleaseArtifactKind,
    ReleaseBundle,
    ReleaseExecutionPlan,
    ReleaseModel,
    ReleaseRunKind,
    ReleaseRunStatement,
    SolarIndexStatement,
    SolarManifestEvidence,
    release_model_payload,
)
from .release_solar import verify_solar_index
from .release_traces import verify_release_run

_ReleaseArtifact = TypeVar("_ReleaseArtifact", bound=ReleaseModel)


def build_run_statement(
    plan_path: Path,
    *,
    corpus_manifest_path: Path,
    output_path: Path,
) -> Path:
    """Verify one completed plan and write its content-addressed statement."""
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
    )
    verify_release_run(
        statement,
        bundle_root=workspace,
        corpus=corpus,
        require_passed=plan.role is ReleaseRunKind.BASELINE,
        require_reference_baseline=plan.role is ReleaseRunKind.BASELINE,
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
    statement_paths: dict[ReleaseArtifactKind, Path],
    output_path: Path,
) -> Path:
    """Verify publisher artifacts and assemble the content-addressed bundle."""
    workspace = workspace_root.resolve()
    output = output_path.resolve()
    _require_missing(output)
    corpus = AkaCorpusManifest.load(corpus_manifest_path)
    references = {
        kind: artifact_reference(workspace, statement_paths[kind])
        for kind in ReleaseArtifactKind
    }
    baseline = _load_statement(
        workspace,
        references[ReleaseArtifactKind.BASELINE],
        BaselineStatement,
    )
    candidate = _load_statement(
        workspace,
        references[ReleaseArtifactKind.CANDIDATE],
        CandidateStatement,
    )
    solar = _load_statement(
        workspace,
        references[ReleaseArtifactKind.SOLAR],
        SolarIndexStatement,
    )
    baseline_run = verify_release_run(
        baseline,
        bundle_root=workspace,
        corpus=corpus,
        require_passed=True,
        require_reference_baseline=True,
    )
    candidate_run = verify_release_run(
        candidate,
        bundle_root=workspace,
        corpus=corpus,
        require_passed=False,
    )
    verify_solar_index(solar, bundle_root=workspace, corpus=corpus)
    if (
        baseline.source_revision != candidate.source_revision
        or baseline.source_revision != solar.source_revision
    ):
        raise ValueError("release evidence source revisions do not match")
    if baseline_run.environment != candidate_run.environment:
        raise ValueError("release runs use different environment identities")
    scoring = corpus.official_scoring
    if (
        scoring.get("status") != "available"
        or scoring.get("baseline_id") != baseline.baseline_id
    ):
        raise ValueError("release baseline is not authorized by the corpus")
    bundle = ReleaseBundle(
        corpus_manifest=artifact_reference(
            workspace, workspace / "corpus" / "manifest.yaml"
        ),
        baseline=references[ReleaseArtifactKind.BASELINE],
        candidate=references[ReleaseArtifactKind.CANDIDATE],
        solar=references[ReleaseArtifactKind.SOLAR],
    )
    atomic_write_json_value(output, release_model_payload(bundle))
    return output


def _run_statement(
    plan: ReleaseExecutionPlan,
    *,
    corpus_manifest: ArtifactReference,
    environment: ArtifactReference,
    problems: tuple[ProblemRunEvidence, ...],
) -> ReleaseRunStatement:
    generated_at = utc_timestamp()
    if plan.role is ReleaseRunKind.BASELINE:
        return BaselineStatement(
            generated_at=generated_at,
            source_revision=plan.source_revision,
            corpus_manifest=corpus_manifest,
            environment=environment,
            problems=problems,
            baseline_id=plan.run_id,
        )
    if plan.role is ReleaseRunKind.CANDIDATE:
        return CandidateStatement(
            generated_at=generated_at,
            source_revision=plan.source_revision,
            corpus_manifest=corpus_manifest,
            environment=environment,
            problems=problems,
            candidate_id=plan.run_id,
        )
    raise ValueError("unknown release execution-plan kind")


def _load_statement(
    root: Path,
    reference: ArtifactReference,
    model: type[_ReleaseArtifact],
) -> _ReleaseArtifact:
    if reference.size_bytes > MAX_RELEASE_STATEMENT_BYTES:
        raise ValueError("release statement exceeds the size limit")
    path = verify_artifact_file(
        root,
        reference.path,
        expected_sha256=reference.sha256,
        expected_size_bytes=reference.size_bytes,
    )
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _require_missing(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"release artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


__all__ = [
    "assemble_release_bundle",
    "build_run_statement",
    "build_solar_index",
]
