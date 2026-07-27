from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from sol_execbench_type_helpers import (
    make_definition,
    make_solution,
    make_workload,
)

from sol_execbench.cli.protocol import CliFailure, CliResult
from sol_execbench.core.data.solution_instance import Solution
from sol_execbench.core.data.trace import EvaluationStatus, Trace
from sol_execbench.core.dataset.aka_contract import AkaCorpusRole
from sol_execbench.core.dataset.aka_corpus import AkaCorpusManifest
from sol_execbench.core.scoring import release_runner
from sol_execbench.core.scoring.release_models import (
    ExecutionPlanProblem,
    ReleaseExecutionPlan,
    ReleaseRunKind,
)


class _TraceResult:
    def __init__(self, successful: bool) -> None:
        self.successful = successful

    def is_successful(self) -> bool:
        return self.successful


def _plan(
    *,
    role: ReleaseRunKind = ReleaseRunKind.CANDIDATE,
    problems: tuple[object, ...] = (object(),),
    corpus_manifest: object | None = None,
) -> ReleaseExecutionPlan:
    return cast(
        ReleaseExecutionPlan,
        SimpleNamespace(
            role=role,
            run_id="run-1",
            problems=problems,
            source_revision="a" * 40,
            environment_path="environment.json",
            corpus_manifest=corpus_manifest,
        ),
    )


def test_execute_release_plan_summarizes_candidate_results(
    tmp_path,
    monkeypatch,
) -> None:
    plan = _plan(problems=(object(), object()))
    results = iter(
        [
            [_TraceResult(True), _TraceResult(False)],
            [_TraceResult(True)],
        ],
    )
    monkeypatch.setattr(
        release_runner,
        "load_execution_plan",
        lambda *_a, **_k: plan,
    )
    monkeypatch.setattr(
        release_runner.AkaCorpusManifest,
        "load",
        staticmethod(lambda _path: object()),
    )
    monkeypatch.setattr(
        release_runner,
        "_verify_plan_contract",
        lambda *_a: None,
    )
    monkeypatch.setattr(
        release_runner,
        "_write_environment_evidence",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        release_runner,
        "_execute_problem",
        lambda *_a, **_k: next(results),
    )

    result = release_runner.execute_release_plan(
        tmp_path / "workspace/plans/plan.json",
        corpus_manifest_path=tmp_path / "manifest.yaml",
        timeout_seconds=17,
        resume=True,
        device="cuda:1",
    )

    assert result.role is ReleaseRunKind.CANDIDATE
    assert result.run_id == "run-1"
    assert (result.problems, result.workloads, result.passed) == (2, 3, 2)


def test_execute_release_plan_rejects_incomplete_baseline(
    tmp_path,
    monkeypatch,
) -> None:
    plan = _plan(role=ReleaseRunKind.BASELINE)
    monkeypatch.setattr(
        release_runner,
        "load_execution_plan",
        lambda *_a, **_k: plan,
    )
    monkeypatch.setattr(
        release_runner.AkaCorpusManifest,
        "load",
        staticmethod(lambda _path: object()),
    )
    monkeypatch.setattr(
        release_runner,
        "_verify_plan_contract",
        lambda *_a: None,
    )
    monkeypatch.setattr(
        release_runner,
        "_write_environment_evidence",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        release_runner,
        "_execute_problem",
        lambda *_a, **_k: [_TraceResult(False)],
    )

    with pytest.raises(
        ValueError,
        match="baseline did not pass every workload",
    ):
        release_runner.execute_release_plan(
            tmp_path / "workspace/plans/plan.json",
            corpus_manifest_path=tmp_path / "manifest.yaml",
        )


def _contract_objects(
    tmp_path: Path,
) -> tuple[ReleaseExecutionPlan, AkaCorpusManifest]:
    problem = SimpleNamespace(
        problem_path="suite/problem",
        definition_sha256="d" * 64,
        workload_sha256="w" * 64,
    )
    plan = _plan(
        problems=(problem,),
        corpus_manifest=SimpleNamespace(
            path="corpus/manifest.yaml",
            sha256="c" * 64,
            size_bytes=10,
        ),
    )
    entry = SimpleNamespace(
        role=AkaCorpusRole.SCORED,
        relative_problem_dir=Path("suite/problem"),
    )
    corpus = SimpleNamespace(
        path=tmp_path / "authoritative.yaml",
        entries=(entry,),
        materialized_problem_sha256={
            "suite/problem": {
                "definition_sha256": "d" * 64,
                "workload_sha256": "w" * 64,
            },
        },
        authored_root=tmp_path / "source/problems",
    )
    return plan, cast(AkaCorpusManifest, corpus)


def _stub_contract_io(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        release_runner,
        "verify_artifact_file",
        lambda *_a, **_k: tmp_path / "bundled.yaml",
    )
    monkeypatch.setattr(release_runner, "sha256_file", lambda _path: "same")


def test_verify_plan_contract_accepts_exact_denominator(
    tmp_path,
    monkeypatch,
) -> None:
    plan, corpus = _contract_objects(tmp_path)
    verified: list[tuple[Path, str]] = []
    _stub_contract_io(monkeypatch, tmp_path)
    monkeypatch.setattr(
        release_runner,
        "verify_release_source_state",
        lambda root, expected_revision: verified.append(
            (root, expected_revision),
        ),
    )

    release_runner._verify_plan_contract(plan, tmp_path, corpus)

    assert verified == [(tmp_path, "a" * 40)]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda plan: setattr(
                plan.problems[0],
                "problem_path",
                "other/problem",
            ),
            "problem denominator mismatch",
        ),
        (
            lambda plan: setattr(plan.problems[0], "definition_sha256", "bad"),
            "identity mismatch",
        ),
    ],
)
def test_verify_plan_contract_rejects_drift(
    tmp_path,
    monkeypatch,
    mutation,
    message,
) -> None:
    plan, corpus = _contract_objects(tmp_path)
    mutation(plan)
    _stub_contract_io(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match=message):
        release_runner._verify_plan_contract(plan, tmp_path, corpus)


def test_verify_plan_contract_rejects_corpus_identity(
    tmp_path,
    monkeypatch,
) -> None:
    plan, corpus = _contract_objects(tmp_path)
    monkeypatch.setattr(
        release_runner,
        "verify_artifact_file",
        lambda *_a, **_k: tmp_path / "bundled.yaml",
    )
    monkeypatch.setattr(
        release_runner,
        "sha256_file",
        lambda path: (
            "bundled" if path.name == "bundled.yaml" else "authoritative"
        ),
    )

    with pytest.raises(ValueError, match="corpus identity mismatch"):
        release_runner._verify_plan_contract(plan, tmp_path, corpus)


def test_environment_evidence_is_written_and_resume_validated(
    tmp_path,
    monkeypatch,
) -> None:
    plan = _plan()
    validated: list[dict[str, Any]] = []
    monkeypatch.setattr(
        release_runner,
        "build_environment_diagnostics",
        lambda: SimpleNamespace(model_dump=lambda **_kwargs: {"status": "ok"}),
    )
    monkeypatch.setattr(
        release_runner,
        "validate_environment_payload",
        lambda payload: validated.append(payload),
    )
    monkeypatch.setattr(
        release_runner,
        "current_release_execution_identity",
        lambda _revision: SimpleNamespace(to_dict=lambda: {"source": "exact"}),
    )

    release_runner._write_environment_evidence(plan, tmp_path, resume=False)

    path = tmp_path / "environment.json"
    assert json.loads(path.read_text())["release_execution"] == {
        "source": "exact",
    }
    assert validated == [
        {"status": "ok", "release_execution": {"source": "exact"}},
    ]

    resumed: list[str] = []
    monkeypatch.setattr(
        release_runner,
        "release_execution_identity_from_payload",
        lambda _payload, expected_source_revision: resumed.append(
            expected_source_revision,
        ),
    )
    release_runner._write_environment_evidence(plan, tmp_path, resume=True)
    assert resumed == ["a" * 40]

    with pytest.raises(
        FileExistsError,
        match="environment evidence already exists",
    ):
        release_runner._write_environment_evidence(plan, tmp_path, resume=False)


def _problem(
    tmp_path: Path,
) -> tuple[ExecutionPlanProblem, AkaCorpusManifest, Path]:
    solution_path = tmp_path / "solution.json"
    solution_path.write_text("{}")
    problem = SimpleNamespace(
        trace_path="traces/problem.jsonl",
        implementation=SimpleNamespace(
            path="solution.json",
            sha256="s" * 64,
            size_bytes=2,
        ),
        problem_path="suite/problem",
    )
    corpus = SimpleNamespace(authored_root=tmp_path / "authored")
    return (
        cast(ExecutionPlanProblem, problem),
        cast(AkaCorpusManifest, corpus),
        solution_path,
    )


def _stub_solution(monkeypatch, solution_path: Path) -> Solution:
    solution = SimpleNamespace(name="candidate")
    monkeypatch.setattr(
        release_runner,
        "verify_artifact_file",
        lambda *_a, **_k: solution_path,
    )
    monkeypatch.setattr(
        release_runner.Solution,
        "model_validate_json",
        staticmethod(lambda _text: solution),
    )
    return cast(Solution, solution)


def test_execute_problem_covers_resume_and_missing_trace(
    tmp_path,
    monkeypatch,
) -> None:
    problem, corpus, solution_path = _problem(tmp_path)
    _stub_solution(monkeypatch, solution_path)
    expected = [cast(Trace, object())]
    monkeypatch.setattr(
        release_runner,
        "_validate_existing_trace",
        lambda *_a: expected,
    )
    trace_path = tmp_path / problem.trace_path
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text("{}")

    assert (
        release_runner._execute_problem(
            _plan(),
            problem,
            workspace=tmp_path,
            corpus=corpus,
            timeout_seconds=5,
            resume=True,
            device="cuda:0",
        )
        is expected
    )
    with pytest.raises(FileExistsError, match="release trace already exists"):
        release_runner._execute_problem(
            _plan(),
            problem,
            workspace=tmp_path,
            corpus=corpus,
            timeout_seconds=5,
            resume=False,
            device="cuda:0",
        )

    trace_path.unlink()
    monkeypatch.setattr(
        release_runner,
        "run_evaluation_cli",
        lambda **_kwargs: CliResult(exit_code=0),
    )
    with pytest.raises(ValueError, match="produced no trace"):
        release_runner._execute_problem(
            _plan(),
            problem,
            workspace=tmp_path,
            corpus=corpus,
            timeout_seconds=5,
            resume=False,
            device="cuda:0",
        )


def test_execute_problem_converts_candidate_cli_failure(
    tmp_path,
    monkeypatch,
) -> None:
    problem, corpus, solution_path = _problem(tmp_path)
    _stub_solution(monkeypatch, solution_path)
    expected = [cast(Trace, object())]
    monkeypatch.setattr(
        release_runner,
        "run_evaluation_cli",
        lambda **_kwargs: (_ for _ in ()).throw(
            CliFailure("timed out", code="evaluation_timeout", exit_code=4),
        ),
    )
    writes: list[Path] = []

    def write_failure(path, **_kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        writes.append(path)

    monkeypatch.setattr(
        release_runner,
        "_write_candidate_failure",
        write_failure,
    )
    monkeypatch.setattr(
        release_runner,
        "_validate_existing_trace",
        lambda *_a: expected,
    )

    result = release_runner._execute_problem(
        _plan(),
        problem,
        workspace=tmp_path,
        corpus=corpus,
        timeout_seconds=5,
        resume=False,
        device="cuda:0",
    )

    assert result is expected
    assert writes == [tmp_path / problem.trace_path]


def _problem_files(tmp_path: Path) -> tuple[Path, Solution]:
    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    definition = make_definition(
        name="vector_add",
        op_type="elementwise",
        axes={"n": {"type": "const", "value": 2}},
        inputs={
            "x": {"shape": ["n"], "dtype": "float32"},
            "y": {"shape": ["n"], "dtype": "float32"},
        },
        outputs={"z": {"shape": ["n"], "dtype": "float32"}},
        reference="def run(x, y):\n    return x + y",
    )
    workload = make_workload(
        uuid="workload-1",
        axes={},
        inputs={"x": {"type": "random"}, "y": {"type": "random"}},
    )
    solution = make_solution(
        name="candidate",
        definition="vector_add",
        author="test",
        spec={
            "languages": ["pytorch"],
            "target_hardware": ["LOCAL"],
            "entry_point": "kernel.py::run",
            "destination_passing_style": False,
        },
        sources=[
            {"path": "kernel.py", "content": "def run(x, y): return x + y"},
        ],
    )
    (problem_dir / "definition.json").write_text(definition.model_dump_json())
    (problem_dir / "workload.jsonl").write_text(
        workload.model_dump_json() + "\n",
    )
    return problem_dir, solution


def test_candidate_failure_writes_one_bounded_trace_per_workload(
    tmp_path,
) -> None:
    problem_dir, solution = _problem_files(tmp_path)
    trace_path = tmp_path / "trace.jsonl"

    release_runner._write_candidate_failure(
        trace_path,
        problem_dir=problem_dir,
        solution=solution,
        failure=CliFailure(
            "timeout " + "x" * 9000,
            code="evaluation_timeout",
            exit_code=4,
        ),
        device="cpu",
    )

    traces = release_runner._validate_existing_trace(
        trace_path,
        problem_dir,
        solution,
        ReleaseRunKind.CANDIDATE,
    )
    assert len(traces) == 1
    assert traces[0].evaluation is not None
    assert traces[0].evaluation.status is EvaluationStatus.TIMEOUT
    assert len(traces[0].evaluation.log) < 8300
    with pytest.raises(ValueError, match="baseline trace did not pass"):
        release_runner._validate_existing_trace(
            trace_path,
            problem_dir,
            solution,
            ReleaseRunKind.BASELINE,
        )


def test_evaluation_request_is_hardened_and_caps_compile_timeout(
    tmp_path,
) -> None:
    request = release_runner._evaluation_request(
        tmp_path / "problem",
        tmp_path / "solution.json",
        tmp_path / "trace.jsonl",
        timeout_seconds=900,
        device="cuda:1",
    )

    assert request.compile_timeout == 300
    assert request.timeout == 900
    assert request.device == "cuda:1"
    assert request.lock_clocks is True
    assert request.unsafe_local_execution is False
