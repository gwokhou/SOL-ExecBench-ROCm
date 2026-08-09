"""CPU-only orchestration checks for content-addressed release execution."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from sol_execbench.core.scoring import release_runner
from sol_execbench.core.scoring.release_models import ReleaseRunKind


class _TraceOutcome:
    def __init__(self, successful: bool) -> None:
        self._successful = successful

    def is_successful(self) -> bool:
        return self._successful


def _patch_release_plan(
    monkeypatch: pytest.MonkeyPatch,
    *,
    role: ReleaseRunKind,
    successful: bool,
) -> None:
    plan = SimpleNamespace(
        role=role,
        run_id="cpu-dry-run",
        problems=[SimpleNamespace(problem_path="synthetic")],
    )
    monkeypatch.setattr(
        release_runner,
        "require_release_qualification",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        release_runner, "load_execution_plan", lambda *a, **k: plan
    )
    monkeypatch.setattr(
        release_runner.AKACorpusManifest,
        "load",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr(
        release_runner, "_verify_plan_contract", lambda *a, **k: None
    )
    monkeypatch.setattr(
        release_runner,
        "_write_environment_evidence",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        release_runner,
        "_execute_problem",
        lambda *a, **k: [_TraceOutcome(successful)],
    )


def test_cpu_dry_run_counts_a_complete_baseline(monkeypatch) -> None:
    _patch_release_plan(
        monkeypatch,
        role=ReleaseRunKind.BASELINE,
        successful=True,
    )

    result = release_runner.execute_release_plan(
        Path("plans/baseline.json"),
        corpus_manifest_path=Path("corpus.json"),
        qualification_root=Path("qualification"),
        evaluator=lambda request: release_runner.ReleaseEvaluationResult(
            exit_code=0
        ),
    )

    assert result.problems == 1
    assert result.workloads == 1
    assert result.passed == 1


def test_cpu_dry_run_rejects_an_incomplete_baseline(monkeypatch) -> None:
    _patch_release_plan(
        monkeypatch,
        role=ReleaseRunKind.BASELINE,
        successful=False,
    )

    with pytest.raises(
        ValueError, match="baseline did not pass every workload"
    ):
        release_runner.execute_release_plan(
            Path("plans/baseline.json"),
            corpus_manifest_path=Path("corpus.json"),
            qualification_root=Path("qualification"),
            evaluator=lambda request: release_runner.ReleaseEvaluationResult(
                exit_code=0
            ),
        )
