from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.solution_instance import Solution
from sol_execbench.core.dataset.aka_corpus import AKACorpusManifest
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.scoring.release_builders import (
    load_execution_plan,
    materialize_release_baseline,
    materialize_release_candidate,
)
from sol_execbench.core.scoring.release_models import ReleaseRunKind

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST = REPO_ROOT / "problems" / "AMD_AKA" / "manifest.yaml"
SOURCE_REVISION = "a" * 40


def test_release_baseline_materializes_exact_scored_corpus(
    tmp_path: Path,
) -> None:
    workspace = materialize_release_baseline(
        MANIFEST,
        tmp_path / "release",
        baseline_id="rx9060xt-baseline-test",
        source_revision=SOURCE_REVISION,
    )
    corpus = AKACorpusManifest.load(MANIFEST)
    baseline = load_execution_plan(workspace / "baseline" / "plan.json")

    expected = {
        entry.relative_problem_dir.as_posix()
        for entry in corpus.entries
        if entry.role == "scored"
    }
    assert len(expected) == 43
    assert (
        sum(
            len(entry.workload_uuids)
            for entry in corpus.entries
            if entry.role == "scored"
        )
        == 163
    )
    assert {item.problem_path for item in baseline.problems} == expected
    assert baseline.role == ReleaseRunKind.BASELINE
    assert not (workspace / "rerun").exists()
    assert sha256_file(workspace / "corpus" / "manifest.yaml") == sha256_file(
        MANIFEST,
    )
    solution = Solution.model_validate_json(
        (workspace / baseline.problems[0].implementation.path).read_text(
            encoding="utf-8",
        ),
    )
    source = solution.sources[0].content
    definition = next(
        entry
        for entry in corpus.entries
        if entry.relative_problem_dir.as_posix()
        == baseline.problems[0].problem_path
    )
    definition_payload = Definition.model_validate_json(
        (
            corpus.authored_root
            / definition.relative_problem_dir
            / "definition.json"
        ).read_text(encoding="utf-8"),
    )
    assert source == definition_payload.reference
    assert "torch.compile" not in source


def test_release_baseline_refuses_to_overwrite_workspace(
    tmp_path: Path,
) -> None:
    output = tmp_path / "release"
    output.mkdir()

    with pytest.raises(FileExistsError):
        materialize_release_baseline(
            MANIFEST,
            output,
            baseline_id="test",
            source_revision=SOURCE_REVISION,
        )


def test_candidate_plan_requires_exact_full_corpus_solution_set(
    tmp_path: Path,
) -> None:
    workspace = materialize_release_baseline(
        MANIFEST,
        tmp_path / "release",
        baseline_id="rx9060xt-baseline-test",
        source_revision=SOURCE_REVISION,
    )
    baseline = load_execution_plan(workspace / "baseline" / "plan.json")
    candidate_root = tmp_path / "candidate-input"
    for item in baseline.problems:
        source = workspace / item.implementation.path
        destination = candidate_root / item.problem_path / "solution.json"
        destination.parent.mkdir(parents=True)
        shutil.copyfile(source, destination)

    plan_path = materialize_release_candidate(
        MANIFEST,
        workspace,
        candidate_root,
        candidate_id="candidate-test",
        source_revision=SOURCE_REVISION,
    )
    candidate = load_execution_plan(plan_path)

    assert candidate.role == ReleaseRunKind.CANDIDATE
    assert candidate.run_id == "candidate-test"
    assert len(candidate.problems) == 43
