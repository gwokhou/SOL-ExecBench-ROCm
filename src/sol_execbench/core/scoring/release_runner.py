# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Trusted full-corpus execution of content-addressed release plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sol_execbench.core.bench.evaluation import make_eval
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    atomic_write_jsonl_values,
    load_json_value,
    load_jsonl_file,
)
from sol_execbench.core.data.solution_instance import Solution
from sol_execbench.core.data.trace import EvaluationStatus, Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.aka_corpus import AKACorpusManifest
from sol_execbench.core.integrity import verify_artifact_file
from sol_execbench.core.platform.environment_diagnostics import (
    build_environment_diagnostics,
)
from sol_execbench.core.platform.rdna4_validation import (
    validate_environment_payload,
)
from sol_execbench.core.scoring.release_builders import load_execution_plan
from sol_execbench.core.scoring.release_contract import (
    verify_release_plan_contract,
)
from sol_execbench.core.scoring.release_environment import (
    current_release_execution_identity,
    release_execution_identity_from_payload,
)
from sol_execbench.core.scoring.release_models import (
    ExecutionPlanProblem,
    ReleaseExecutionPlan,
    ReleaseRunKind,
)
from sol_execbench.core.scoring.release_qualification import (
    require_release_qualification,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseRunResult:
    """Summary of one completed full-corpus release execution."""

    role: ReleaseRunKind
    run_id: str
    problems: int
    workloads: int
    passed: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseEvaluationRequest:
    """Inputs required from an evaluation application adapter."""

    problem_dir: Path
    solution_path: Path
    trace_path: Path
    timeout_seconds: int
    device: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseEvaluationResult:
    """Minimal evaluator outcome consumed by release execution."""

    exit_code: int


class ReleaseEvaluationError(RuntimeError):
    """Classified evaluator failure without a CLI dependency."""

    def __init__(self, message: str, *, code: str, exit_code: int) -> None:
        """Initialize a classified application-level evaluation error."""
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code


class ReleaseEvaluator(Protocol):
    """Application adapter capable of producing one canonical trace."""

    def __call__(
        self,
        request: ReleaseEvaluationRequest,
    ) -> ReleaseEvaluationResult:
        """Evaluate one release problem and publish its trace."""
        ...


def execute_release_plan(
    plan_path: Path,
    *,
    corpus_manifest_path: Path,
    qualification_root: Path,
    evaluator: ReleaseEvaluator,
    timeout_seconds: int = 900,
    resume: bool = False,
    device: str = "cuda:0",
) -> ReleaseRunResult:
    """Execute one exact plan through the normal hardened evaluator."""
    plan_file = plan_path.resolve()
    require_release_qualification(
        plan_file,
        corpus_manifest_path=corpus_manifest_path,
        qualification_root=qualification_root,
        timeout_seconds=timeout_seconds,
        device=device,
    )
    workspace = plan_file.parents[1]
    plan = load_execution_plan(plan_file, workspace_root=workspace)
    corpus = AKACorpusManifest.load(corpus_manifest_path)
    _verify_plan_contract(plan, workspace, corpus)
    _write_environment_evidence(plan, workspace, resume=resume)
    passed = 0
    workloads = 0
    for problem in plan.problems:
        result = _execute_problem(
            plan,
            problem,
            workspace=workspace,
            corpus=corpus,
            timeout_seconds=timeout_seconds,
            resume=resume,
            device=device,
            evaluator=evaluator,
        )
        workloads += len(result)
        passed += sum(trace.is_successful() for trace in result)
    if plan.role is ReleaseRunKind.BASELINE and passed != workloads:
        raise ValueError("release baseline did not pass every workload")
    return ReleaseRunResult(
        role=plan.role,
        run_id=plan.run_id,
        problems=len(plan.problems),
        workloads=workloads,
        passed=passed,
    )


def _verify_plan_contract(
    plan: ReleaseExecutionPlan,
    workspace: Path,
    corpus: AKACorpusManifest,
) -> None:
    verify_release_plan_contract(plan, workspace, corpus)


def _write_environment_evidence(
    plan: ReleaseExecutionPlan,
    workspace: Path,
    *,
    resume: bool,
) -> None:
    path = workspace / plan.environment_path
    if path.exists():
        if not resume:
            raise FileExistsError(
                f"release environment evidence already exists: {path}",
            )
        payload = load_json_value(path)
        validate_environment_payload(payload)
        release_execution_identity_from_payload(
            payload,
            expected_source_revision=plan.source_revision,
        )
        return
    payload = build_environment_diagnostics().model_dump(mode="json")
    validate_environment_payload(payload)
    payload["release_execution"] = current_release_execution_identity(
        plan.source_revision,
    ).to_dict()
    atomic_write_json_value(path, payload)


def _execute_problem(
    plan: ReleaseExecutionPlan,
    problem: ExecutionPlanProblem,
    *,
    workspace: Path,
    corpus: AKACorpusManifest,
    timeout_seconds: int,
    resume: bool,
    device: str,
    evaluator: ReleaseEvaluator,
) -> list[Trace]:
    trace_path = workspace / problem.trace_path
    solution_path = verify_artifact_file(
        workspace,
        problem.implementation.path,
        expected_sha256=problem.implementation.sha256,
        expected_size_bytes=problem.implementation.size_bytes,
    )
    problem_dir = corpus.authored_root / problem.problem_path
    solution = Solution.model_validate_json(
        solution_path.read_text(encoding="utf-8"),
    )
    if trace_path.exists():
        if not resume:
            raise FileExistsError(f"release trace already exists: {trace_path}")
        return _validate_existing_trace(
            trace_path,
            problem_dir,
            solution,
            plan.role,
        )
    try:
        result = evaluator(
            ReleaseEvaluationRequest(
                problem_dir=problem_dir,
                solution_path=solution_path,
                trace_path=trace_path,
                timeout_seconds=timeout_seconds,
                device=device,
            ),
        )
    except ReleaseEvaluationError as exc:
        if plan.role is not ReleaseRunKind.CANDIDATE:
            raise
        _write_candidate_failure(
            trace_path,
            problem_dir=problem_dir,
            solution=solution,
            failure=exc,
            device=device,
        )
    else:
        if result.exit_code not in {0, 1} or not trace_path.is_file():
            raise ValueError(
                f"release evaluator produced no trace: {problem.problem_path}",
            )
    return _validate_existing_trace(
        trace_path,
        problem_dir,
        solution,
        plan.role,
    )


def _write_candidate_failure(
    trace_path: Path,
    *,
    problem_dir: Path,
    solution: Solution,
    failure: ReleaseEvaluationError,
    device: str,
) -> None:
    status = {
        "compilation_failed": EvaluationStatus.COMPILE_ERROR,
        "evaluation_timeout": EvaluationStatus.TIMEOUT,
    }.get(failure.code, EvaluationStatus.RUNTIME_ERROR)
    if failure.exit_code < 4:
        raise failure
    definition = Definition.model_validate_json(
        (problem_dir / "definition.json").read_text(encoding="utf-8"),
    )
    workloads = load_jsonl_file(Workload, problem_dir / "workload.jsonl")
    message = str(failure)[:8192]
    traces = [
        Trace(
            definition=definition.name,
            workload=workload,
            solution=solution.name,
            evaluation=make_eval(
                status,
                device,
                None,
                extra_msg=message,
                clocks_locked=None,
                timing_protocol=None,
            ),
        )
        for workload in workloads
    ]
    atomic_write_jsonl_values(trace_path, traces)


def _validate_existing_trace(
    trace_path: Path,
    problem_dir: Path,
    solution: Solution,
    role: ReleaseRunKind,
) -> list[Trace]:
    traces = load_jsonl_file(Trace, trace_path)
    workloads = load_jsonl_file(Workload, problem_dir / "workload.jsonl")
    expected = {item.uuid for item in workloads}
    observed = {item.workload.uuid for item in traces}
    if (
        len(traces) != len(expected)
        or observed != expected
        or any(item.solution != solution.name for item in traces)
    ):
        raise ValueError(f"release trace identity mismatch: {trace_path}")
    if role is ReleaseRunKind.BASELINE and not all(
        item.is_successful() for item in traces
    ):
        raise ValueError(f"release baseline trace did not pass: {trace_path}")
    return traces


__all__ = [
    "ReleaseEvaluationError",
    "ReleaseEvaluationRequest",
    "ReleaseEvaluationResult",
    "ReleaseEvaluator",
    "ReleaseRunResult",
    "execute_release_plan",
]
