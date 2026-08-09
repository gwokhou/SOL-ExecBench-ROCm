# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared immutable input verification for release execution tasks."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.dataset.aka_contract import AKACorpusRole
from sol_execbench.core.dataset.aka_corpus import AKACorpusManifest
from sol_execbench.core.integrity import sha256_file, verify_artifact_file
from sol_execbench.core.scoring.release_environment import (
    verify_release_source_state,
)
from sol_execbench.core.scoring.release_models import ReleaseExecutionPlan


def verify_release_plan_contract(
    plan: ReleaseExecutionPlan,
    workspace: Path,
    corpus: AKACorpusManifest,
) -> None:
    """Verify one plan's exact corpus denominator and clean source identity."""
    bundled = verify_artifact_file(
        workspace,
        plan.corpus_manifest.path,
        expected_sha256=plan.corpus_manifest.sha256,
        expected_size_bytes=plan.corpus_manifest.size_bytes,
    )
    if sha256_file(bundled) != sha256_file(corpus.path):
        raise ValueError("release execution plan corpus identity mismatch")
    expected = {
        entry.relative_problem_dir.as_posix(): entry
        for entry in corpus.entries
        if entry.role is AKACorpusRole.SCORED
    }
    observed = {item.problem_path: item for item in plan.problems}
    if set(observed) != set(expected):
        raise ValueError("release execution plan problem denominator mismatch")
    for path, item in observed.items():
        identity = corpus.materialized_problem_sha256[path]
        if (
            item.definition_sha256 != identity["definition_sha256"]
            or item.workload_sha256 != identity["workload_sha256"]
        ):
            raise ValueError(
                f"release execution plan identity mismatch: {path}"
            )
    verify_release_source_state(
        corpus.authored_root.parents[1],
        expected_revision=plan.source_revision,
    )


__all__ = ["verify_release_plan_contract"]
