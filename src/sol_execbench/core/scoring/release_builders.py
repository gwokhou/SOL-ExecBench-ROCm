# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Deterministic builders for release baseline and candidate evidence."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.data.solution_instance import Solution
from sol_execbench.core.data.solution_models import (
    BuildSpec,
    SourceFile,
    SupportedHardware,
    SupportedLanguages,
)
from sol_execbench.core.dataset.aka_contract import AkaCorpusRole
from sol_execbench.core.dataset.aka_corpus import AkaCorpusManifest
from sol_execbench.core.integrity import sha256_file, verify_artifact_file
from sol_execbench.core.timestamps import utc_timestamp

from .release_models import (
    ArtifactReference,
    ExecutionPlanProblem,
    ReleaseExecutionPlan,
    ReleaseRunKind,
    release_model_payload,
)


def materialize_release_baseline(
    manifest_path: Path,
    output_root: Path,
    *,
    baseline_id: str,
    source_revision: str,
) -> Path:
    """Create an immutable trusted-reference baseline execution plan."""
    corpus = AkaCorpusManifest.load(manifest_path)
    output = output_root.resolve()
    if output.exists():
        raise FileExistsError(f"release workspace already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        corpus_ref = _copy_corpus_manifest(corpus, staging)
        problems = _materialize_baseline_solutions(corpus, staging)
        _write_execution_plan(
            staging,
            corpus_ref=corpus_ref,
            problems=problems,
            baseline_id=baseline_id,
            source_revision=source_revision,
        )
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def materialize_release_candidate(
    manifest_path: Path,
    workspace_root: Path,
    candidate_root: Path,
    *,
    candidate_id: str,
    source_revision: str,
) -> Path:
    """Ingest one exact full-corpus candidate set into a release workspace."""
    corpus = AkaCorpusManifest.load(manifest_path)
    workspace = workspace_root.resolve()
    destination = workspace / "candidate"
    if destination.exists():
        raise FileExistsError(f"release candidate already exists: {destination}")
    _verify_workspace_corpus(corpus, workspace)
    staging = Path(tempfile.mkdtemp(prefix=".candidate.", dir=workspace))
    try:
        problems = _copy_candidate_solutions(
            corpus,
            workspace=workspace,
            staging=staging,
            candidate_root=candidate_root.resolve(),
        )
        plan = ReleaseExecutionPlan(
            generated_at=utc_timestamp(),
            source_revision=source_revision,
            run_id=candidate_id,
            role=ReleaseRunKind.CANDIDATE,
            corpus_manifest=artifact_reference(
                workspace, workspace / "corpus" / "manifest.yaml"
            ),
            environment_path="candidate/environment.json",
            problems=problems,
        )
        atomic_write_json_value(
            staging / "plan.json",
            release_model_payload(plan),
        )
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "plan.json"


def load_execution_plan(
    plan_path: Path,
    *,
    workspace_root: Path | None = None,
) -> ReleaseExecutionPlan:
    """Load a plan and verify all pre-existing content-addressed inputs."""
    path = plan_path.resolve()
    root = (workspace_root or path.parents[1]).resolve()
    plan = ReleaseExecutionPlan.model_validate_json(path.read_text(encoding="utf-8"))
    verify_artifact_file(
        root,
        plan.corpus_manifest.path,
        expected_sha256=plan.corpus_manifest.sha256,
        expected_size_bytes=plan.corpus_manifest.size_bytes,
    )
    for problem in plan.problems:
        verify_artifact_file(
            root,
            problem.implementation.path,
            expected_sha256=problem.implementation.sha256,
            expected_size_bytes=problem.implementation.size_bytes,
        )
    return plan


def artifact_reference(root: Path, path: Path) -> ArtifactReference:
    """Build a strict root-relative reference for one regular file."""
    base = root.resolve()
    resolved = path.resolve()
    if not resolved.is_file() or (
        resolved.parent != base and base not in resolved.parents
    ):
        raise ValueError(f"artifact is not a regular file below release root: {path}")
    relative = resolved.relative_to(base).as_posix()
    return ArtifactReference(
        path=relative,
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _copy_corpus_manifest(
    corpus: AkaCorpusManifest,
    workspace: Path,
) -> ArtifactReference:
    destination = workspace / "corpus" / "manifest.yaml"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(corpus.path, destination)
    return artifact_reference(workspace, destination)


def _verify_workspace_corpus(
    corpus: AkaCorpusManifest,
    workspace: Path,
) -> None:
    bundled = workspace / "corpus" / "manifest.yaml"
    if not bundled.is_file() or sha256_file(bundled) != sha256_file(corpus.path):
        raise ValueError("release workspace corpus manifest identity mismatch")


def _copy_candidate_solutions(
    corpus: AkaCorpusManifest,
    *,
    workspace: Path,
    staging: Path,
    candidate_root: Path,
) -> tuple[ExecutionPlanProblem, ...]:
    problems: list[ExecutionPlanProblem] = []
    for entry in corpus.entries:
        if entry.role is not AkaCorpusRole.SCORED:
            continue
        problem_path = entry.relative_problem_dir.as_posix()
        source = candidate_root / problem_path / "solution.json"
        solution = Solution.model_validate_json(source.read_text(encoding="utf-8"))
        definition = Definition.model_validate_json(
            (corpus.authored_root / problem_path / "definition.json").read_text(
                encoding="utf-8"
            )
        )
        if solution.definition != definition.name:
            raise ValueError(f"candidate targets wrong definition: {problem_path}")
        destination = staging / "implementations" / problem_path / "solution.json"
        atomic_write_json_value(destination, solution.model_dump(mode="json"))
        identity = corpus.materialized_problem_sha256[problem_path]
        final_path = (
            workspace / "candidate" / "implementations" / problem_path / "solution.json"
        )
        problems.append(
            ExecutionPlanProblem(
                problem_path=problem_path,
                definition_sha256=identity["definition_sha256"],
                workload_sha256=identity["workload_sha256"],
                implementation=ArtifactReference(
                    path=final_path.relative_to(workspace).as_posix(),
                    sha256=sha256_file(destination),
                    size_bytes=destination.stat().st_size,
                ),
                trace_path=f"candidate/traces/{problem_path}/trace.jsonl",
            )
        )
    return tuple(problems)


def _materialize_baseline_solutions(
    corpus: AkaCorpusManifest,
    workspace: Path,
) -> tuple[ExecutionPlanProblem, ...]:
    problems: list[ExecutionPlanProblem] = []
    for entry in corpus.entries:
        if entry.role is not AkaCorpusRole.SCORED:
            continue
        problem_path = entry.relative_problem_dir.as_posix()
        identity = corpus.materialized_problem_sha256[problem_path]
        definition = Definition.model_validate_json(
            (corpus.authored_root / problem_path / "definition.json").read_text(
                encoding="utf-8"
            )
        )
        destination = (
            workspace / "baseline" / "implementations" / problem_path / "solution.json"
        )
        atomic_write_json_value(
            destination,
            reference_baseline_solution(definition).model_dump(mode="json"),
        )
        problems.append(
            ExecutionPlanProblem(
                problem_path=problem_path,
                definition_sha256=identity["definition_sha256"],
                workload_sha256=identity["workload_sha256"],
                implementation=artifact_reference(workspace, destination),
                trace_path=(f"baseline/traces/{problem_path}/trace.jsonl"),
            )
        )
    if not problems:
        raise ValueError("release corpus contains no scored baseline problems")
    return tuple(problems)


def reference_baseline_solution(definition: Definition) -> Solution:
    """Build the canonical eager reference solution for one definition."""
    return Solution(
        name=f"release_trusted_reference_{definition.name}",
        definition=definition.name,
        author="SOL ExecBench ROCm release",
        spec=BuildSpec(
            languages=[SupportedLanguages.PYTORCH],
            target_hardware=[SupportedHardware.GFX1200],
            entry_point="kernel.py::run",
            dependencies=["torch"],
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="kernel.py", content=definition.reference)],
        description=(
            "Release-defined eager baseline using the exact corpus-pinned "
            "trusted PyTorch reference."
        ),
    )


def _write_execution_plan(
    workspace: Path,
    *,
    corpus_ref: ArtifactReference,
    problems: tuple[ExecutionPlanProblem, ...],
    baseline_id: str,
    source_revision: str,
) -> None:
    generated_at = utc_timestamp()
    baseline = ReleaseExecutionPlan(
        generated_at=generated_at,
        source_revision=source_revision,
        run_id=baseline_id,
        role=ReleaseRunKind.BASELINE,
        corpus_manifest=corpus_ref,
        environment_path="baseline/environment.json",
        problems=problems,
    )
    atomic_write_json_value(
        workspace / "baseline" / "plan.json",
        release_model_payload(baseline),
    )


__all__ = [
    "artifact_reference",
    "load_execution_plan",
    "materialize_release_baseline",
    "materialize_release_candidate",
    "reference_baseline_solution",
]
