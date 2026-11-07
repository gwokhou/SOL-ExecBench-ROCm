# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Mandatory three-stage qualification for full-corpus release execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sol_execbench.core.bench.batch_gpu_qualification import (
    BatchGPUQualificationGate,
    BatchGPUQualificationReceipt,
    BatchGPUQualificationStage,
    LargeBatchGPUTask,
    qualification_artifact,
    qualification_gate_path,
    qualification_parent_stage,
    require_isolated_qualification_root,
    verify_qualification_artifact,
)
from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    atomic_write_jsonl_values,
    load_json_file,
    load_jsonl_file,
)
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.aka_corpus import AKACorpusManifest
from sol_execbench.core.integrity import (
    sha256_file,
    stable_json_checksum,
    verify_artifact_file,
)
from sol_execbench.core.scoring.release_builders import load_execution_plan
from sol_execbench.core.scoring.release_contract import (
    verify_release_plan_contract,
)
from sol_execbench.core.scoring.release_models import (
    ExecutionPlanProblem,
    ReleaseExecutionPlan,
)
from sol_execbench.core.timestamps import utc_timestamp

_QUALIFICATION_CONFIG = BenchmarkConfig(
    warmup_runs=0,
    iterations=1,
    trials=1,
    min_measurement_time_seconds=None,
    lock_clocks=False,
    benchmark_reference=True,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseQualificationRequest:
    """One problem partition selected for minimal evaluator execution."""

    problem_dir: Path
    solution_path: Path
    workload_path: Path
    config_path: Path
    trace_path: Path
    timeout_seconds: int
    device: str


class ReleaseQualificationEvaluator(Protocol):
    """Application adapter for the canonical isolated evaluator."""

    def __call__(self, request: ReleaseQualificationRequest) -> int:
        """Evaluate the selected workloads and return the CLI exit code."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class _ReleaseQualificationContext:
    plan_path: Path
    workspace: Path
    plan: ReleaseExecutionPlan
    corpus: AKACorpusManifest
    root: Path
    timeout_seconds: int
    device: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _ProblemPartition:
    problem: ExecutionPlanProblem
    definition: Definition
    workloads: tuple[Workload, ...]
    risk: int


def run_release_qualification(
    plan_path: Path,
    *,
    corpus_manifest_path: Path,
    qualification_root: Path,
    stage: BatchGPUQualificationStage,
    evaluator: ReleaseQualificationEvaluator,
    timeout_seconds: int = 900,
    device: str = "cuda:0",
) -> BatchGPUQualificationGate:
    """Run or verify one mandatory release qualification stage."""
    context = _context(
        plan_path,
        corpus_manifest_path=corpus_manifest_path,
        qualification_root=qualification_root,
        timeout_seconds=timeout_seconds,
        device=device,
    )
    gate_path = qualification_gate_path(context.root, stage)
    if gate_path.is_file():
        return _verify_gate(context, stage)
    parent_hash = _require_parent(context, stage)
    selected = _selected_partitions(context, stage)
    receipts = (
        (_run_static(context, selected),)
        if stage is BatchGPUQualificationStage.STATIC
        else tuple(
            _run_gpu_partition(context, stage, item, evaluator)
            for item in selected
        )
    )
    gate = _build_gate(context, stage, parent_hash, receipts)
    atomic_write_json_value(gate_path, gate.model_dump(mode="json"))
    return _verify_gate(context, stage)


def require_release_qualification(
    plan_path: Path,
    *,
    corpus_manifest_path: Path,
    qualification_root: Path,
    timeout_seconds: int = 900,
    device: str = "cuda:0",
) -> BatchGPUQualificationGate:
    """Verify the complete chain before formal release timing starts."""
    context = _context(
        plan_path,
        corpus_manifest_path=corpus_manifest_path,
        qualification_root=qualification_root,
        timeout_seconds=timeout_seconds,
        device=device,
    )
    return _verify_gate(context, BatchGPUQualificationStage.FULL)


def _context(
    plan_path: Path,
    *,
    corpus_manifest_path: Path,
    qualification_root: Path,
    timeout_seconds: int,
    device: str,
) -> _ReleaseQualificationContext:
    if timeout_seconds <= 0:
        raise ValueError("qualification timeout must be positive")
    plan_file = plan_path.resolve()
    workspace = plan_file.parents[1]
    plan = load_execution_plan(plan_file, workspace_root=workspace)
    corpus = AKACorpusManifest.load(corpus_manifest_path)
    verify_release_plan_contract(plan, workspace, corpus)
    root = require_isolated_qualification_root(qualification_root, workspace)
    return _ReleaseQualificationContext(
        plan_path=plan_file,
        workspace=workspace,
        plan=plan,
        corpus=corpus,
        root=root,
        timeout_seconds=timeout_seconds,
        device=device,
    )


def _partitions(
    context: _ReleaseQualificationContext,
) -> tuple[_ProblemPartition, ...]:
    partitions: list[_ProblemPartition] = []
    for problem in context.plan.problems:
        problem_dir = context.corpus.authored_root / problem.problem_path
        definition = load_json_file(Definition, problem_dir / "definition.json")
        workloads = tuple(
            load_jsonl_file(Workload, problem_dir / "workload.jsonl")
        )
        risk = max(
            (_workload_risk(definition, item) for item in workloads), default=0
        )
        partitions.append(
            _ProblemPartition(
                problem=problem,
                definition=definition,
                workloads=workloads,
                risk=risk,
            )
        )
    return tuple(partitions)


def _workload_risk(definition: Definition, workload: Workload) -> int:
    axes = definition.get_resolved_axes_values(workload.axes)
    risk = 1
    for value in axes.values():
        risk *= max(1, value)
    return risk


def _canary_workloads(partition: _ProblemPartition) -> tuple[Workload, ...]:
    resolved = {
        item.uuid: partition.definition.get_resolved_axes_values(item.axes)
        for item in partition.workloads
    }
    selected: set[str] = set()
    axis_names = sorted({name for axes in resolved.values() for name in axes})
    for name in axis_names:
        ordered = sorted(
            partition.workloads,
            key=lambda item: (resolved[item.uuid].get(name, 0), item.uuid),
        )
        selected.update((ordered[0].uuid, ordered[-1].uuid))
    if not selected and partition.workloads:
        selected.add(partition.workloads[0].uuid)
    return tuple(
        sorted(
            (item for item in partition.workloads if item.uuid in selected),
            key=lambda item: (
                -_workload_risk(partition.definition, item),
                item.uuid,
            ),
        )
    )


def _selected_partitions(
    context: _ReleaseQualificationContext,
    stage: BatchGPUQualificationStage,
) -> tuple[_ProblemPartition, ...]:
    partitions = _partitions(context)
    if stage is BatchGPUQualificationStage.CANARY:
        selected = tuple(
            _ProblemPartition(
                problem=item.problem,
                definition=item.definition,
                workloads=_canary_workloads(item),
                risk=item.risk,
            )
            for item in partitions
        )
        return tuple(
            sorted(
                selected,
                key=lambda item: (-item.risk, item.problem.problem_path),
            )
        )
    return partitions


def _item_ids(partitions: tuple[_ProblemPartition, ...]) -> tuple[str, ...]:
    return tuple(
        f"{partition.problem.problem_path}/{workload.uuid}"
        for partition in partitions
        for workload in partition.workloads
    )


def _subject_sha256(context: _ReleaseQualificationContext) -> str:
    implementations = []
    for problem in context.plan.problems:
        path = verify_artifact_file(
            context.workspace,
            problem.implementation.path,
            expected_sha256=problem.implementation.sha256,
            expected_size_bytes=problem.implementation.size_bytes,
        )
        implementations.append((problem.problem_path, sha256_file(path)))
    return stable_json_checksum(
        {
            "plan_sha256": sha256_file(context.plan_path),
            "corpus_sha256": sha256_file(context.corpus.path),
            "implementations": implementations,
        }
    )


def _runner_sha256() -> str:
    return stable_json_checksum(
        {"release_qualification": sha256_file(Path(__file__))}
    )


def _configuration_sha256(context: _ReleaseQualificationContext) -> str:
    return stable_json_checksum(
        {
            "device": context.device,
            "timeout_seconds": context.timeout_seconds,
            "benchmark_config": _QUALIFICATION_CONFIG.model_dump(mode="json"),
        }
    )


def _run_static(
    context: _ReleaseQualificationContext,
    partitions: tuple[_ProblemPartition, ...],
) -> BatchGPUQualificationReceipt:
    path = context.root / "static" / "preflight.json"
    item_ids = _item_ids(partitions)
    payload = {
        "task": LargeBatchGPUTask.RELEASE_EVALUATION,
        "scope_id": context.plan.run_id,
        "subject_sha256": _subject_sha256(context),
        "item_ids": item_ids,
        "all_inputs_valid": True,
    }
    atomic_write_json_value(path, payload)
    return BatchGPUQualificationReceipt(
        stage=BatchGPUQualificationStage.STATIC,
        partition="release-plan",
        item_ids=item_ids,
        input_sha256=stable_json_checksum(payload),
        artifacts=(qualification_artifact(context.root, path),),
    )


def _run_gpu_partition(
    context: _ReleaseQualificationContext,
    stage: BatchGPUQualificationStage,
    partition: _ProblemPartition,
    evaluator: ReleaseQualificationEvaluator,
) -> BatchGPUQualificationReceipt:
    directory = context.root / stage.value / partition.problem.problem_path
    workload_path = directory / "workload.jsonl"
    config_path = context.root / "config.json"
    trace_path = directory / "trace.jsonl"
    atomic_write_jsonl_values(workload_path, list(partition.workloads))
    atomic_write_json_value(
        config_path, _QUALIFICATION_CONFIG.model_dump(mode="json")
    )
    solution_path = verify_artifact_file(
        context.workspace,
        partition.problem.implementation.path,
        expected_sha256=partition.problem.implementation.sha256,
        expected_size_bytes=partition.problem.implementation.size_bytes,
    )
    exit_code = evaluator(
        ReleaseQualificationRequest(
            problem_dir=context.corpus.authored_root
            / partition.problem.problem_path,
            solution_path=solution_path,
            workload_path=workload_path,
            config_path=config_path,
            trace_path=trace_path,
            timeout_seconds=context.timeout_seconds,
            device=context.device,
        )
    )
    _verify_trace(trace_path, partition, exit_code)
    item_ids = _item_ids((partition,))
    return BatchGPUQualificationReceipt(
        stage=stage,
        partition=partition.problem.problem_path,
        item_ids=item_ids,
        input_sha256=_partition_input_sha256(partition, solution_path),
        artifacts=(
            qualification_artifact(context.root, workload_path),
            qualification_artifact(context.root, trace_path),
        ),
    )


def _partition_input_sha256(
    partition: _ProblemPartition, solution: Path
) -> str:
    return stable_json_checksum(
        {
            "definition_sha256": partition.problem.definition_sha256,
            "workload_sha256": partition.problem.workload_sha256,
            "solution_sha256": sha256_file(solution),
            "selected": [item.uuid for item in partition.workloads],
        }
    )


def _verify_trace(
    trace_path: Path, partition: _ProblemPartition, exit_code: int
) -> None:
    if exit_code != 0:
        raise ValueError(
            f"release qualification failed: {partition.problem.problem_path}"
        )
    traces = load_jsonl_file(Trace, trace_path)
    expected = {item.uuid for item in partition.workloads}
    observed = {item.workload.uuid for item in traces}
    if len(traces) != len(expected) or observed != expected:
        raise ValueError("release qualification trace denominator mismatch")
    if not all(item.is_successful() for item in traces):
        raise ValueError("release qualification trace contains a failure")


def _build_gate(
    context: _ReleaseQualificationContext,
    stage: BatchGPUQualificationStage,
    parent_hash: str | None,
    receipts: tuple[BatchGPUQualificationReceipt, ...],
) -> BatchGPUQualificationGate:
    return BatchGPUQualificationGate(
        task=LargeBatchGPUTask.RELEASE_EVALUATION,
        stage=stage,
        scope_id=context.plan.run_id,
        subject_sha256=_subject_sha256(context),
        runner_sha256=_runner_sha256(),
        configuration_sha256=_configuration_sha256(context),
        source_revision=context.plan.source_revision,
        parent_gate_sha256=parent_hash,
        item_ids=tuple(
            item for receipt in receipts for item in receipt.item_ids
        ),
        receipts=receipts,
        created_at=utc_timestamp(),
    )


def _require_parent(
    context: _ReleaseQualificationContext,
    stage: BatchGPUQualificationStage,
) -> str | None:
    parent = qualification_parent_stage(stage)
    if parent is None:
        return None
    _verify_gate(context, parent)
    return sha256_file(qualification_gate_path(context.root, parent))


def _verify_gate(
    context: _ReleaseQualificationContext,
    stage: BatchGPUQualificationStage,
) -> BatchGPUQualificationGate:
    parent = qualification_parent_stage(stage)
    parent_hash = None
    if parent is not None:
        _verify_gate(context, parent)
        parent_hash = sha256_file(qualification_gate_path(context.root, parent))
    gate = load_json_file(
        BatchGPUQualificationGate,
        qualification_gate_path(context.root, stage),
    )
    expected_items = _item_ids(_selected_partitions(context, stage))
    expected = (
        gate.task is LargeBatchGPUTask.RELEASE_EVALUATION
        and gate.stage is stage
        and gate.scope_id == context.plan.run_id
        and gate.subject_sha256 == _subject_sha256(context)
        and gate.runner_sha256 == _runner_sha256()
        and gate.configuration_sha256 == _configuration_sha256(context)
        and gate.source_revision == context.plan.source_revision
        and gate.parent_gate_sha256 == parent_hash
        and set(gate.item_ids) == set(expected_items)
    )
    if not expected:
        raise ValueError(f"release qualification identity drift: {stage}")
    _verify_receipts(context, gate, _selected_partitions(context, stage))
    return gate


def _verify_receipts(
    context: _ReleaseQualificationContext,
    gate: BatchGPUQualificationGate,
    partitions: tuple[_ProblemPartition, ...],
) -> None:
    by_partition = {item.problem.problem_path: item for item in partitions}
    for receipt in gate.receipts:
        for artifact in receipt.artifacts:
            verify_qualification_artifact(context.root, artifact)
        if gate.stage is BatchGPUQualificationStage.STATIC:
            continue
        partition = by_partition.get(receipt.partition)
        if partition is None:
            raise ValueError("release qualification partition drift")
        solution = context.workspace / partition.problem.implementation.path
        if receipt.input_sha256 != _partition_input_sha256(partition, solution):
            raise ValueError("release qualification input drift")
        trace = next(
            item
            for item in receipt.artifacts
            if item.path.endswith("trace.jsonl")
        )
        _verify_trace(context.root / trace.path, partition, 0)


__all__ = [
    "ReleaseQualificationEvaluator",
    "ReleaseQualificationRequest",
    "require_release_qualification",
    "run_release_qualification",
]
