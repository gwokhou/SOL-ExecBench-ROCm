# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Mandatory three-stage qualification for full-corpus SOLAR release builds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sol_execbench.core.bench.batch_gpu_qualification import (
    BatchGPUQualificationGate,
    BatchGPUQualificationReceipt,
    BatchGPUQualificationStage,
    LargeBatchGPUTask,
    qualification_artifact,
    qualification_gate_path,
    qualification_parent_stage,
    require_isolated_qualification_root,
    select_risk_first_axis_extrema,
    verify_qualification_artifact,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
    load_jsonl_file,
)
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.aka_contract import AKACorpusRole
from sol_execbench.core.dataset.aka_corpus import AKACorpusManifest
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.scoring.release_builders import load_execution_plan
from sol_execbench.core.scoring.release_environment import (
    verify_release_source_state,
)
from sol_execbench.core.solar_bridge.corpus_readiness import (
    CorpusReadinessRecord,
    CorpusReadinessStatus,
    CorpusReadinessSummary,
    audit_corpus_stage_readiness,
)
from sol_execbench.core.solar_bridge.formal_device import (
    formal_architecture_profile_hash,
)
from sol_execbench.core.solar_bridge.models import (
    DEFAULT_IR_PATH,
    IRPath,
    normalize_ir_path,
)
from sol_execbench.core.timestamps import utc_timestamp


@dataclass(frozen=True, slots=True, kw_only=True)
class _SolarQualificationContext:
    workspace: Path
    corpus: AKACorpusManifest
    source_revision: str
    root: Path
    orojenesis_home: Path
    device: str
    ir_path: IRPath
    timeout_seconds: float
    scope_id: str
    jobs: int


def run_solar_release_qualification(
    workspace: Path,
    *,
    corpus_manifest_path: Path,
    orojenesis_home: Path,
    qualification_root: Path,
    stage: BatchGPUQualificationStage,
    timeout_seconds: float = 14_400,
    device: str = "cuda:0",
    ir_path: IRPath | str = DEFAULT_IR_PATH,
    jobs: int = 1,
) -> BatchGPUQualificationGate:
    """Run or verify one SOLAR release qualification stage."""
    context = _context(
        workspace,
        corpus_manifest_path=corpus_manifest_path,
        orojenesis_home=orojenesis_home,
        qualification_root=qualification_root,
        timeout_seconds=timeout_seconds,
        device=device,
        ir_path=ir_path,
        jobs=jobs,
    )
    path = qualification_gate_path(context.root, stage)
    if path.is_file():
        return _verify_gate(context, stage)
    parent_hash = _require_parent(context, stage)
    item_ids = _selected_item_ids(context, stage)
    receipt = (
        _run_static(context, item_ids)
        if stage is BatchGPUQualificationStage.STATIC
        else _run_gpu_stage(context, stage, item_ids)
    )
    gate = _build_gate(context, stage, parent_hash, receipt)
    atomic_write_json_value(path, gate.model_dump(mode="json"))
    return _verify_gate(context, stage)


def require_solar_release_qualification(
    workspace: Path,
    *,
    corpus_manifest_path: Path,
    orojenesis_home: Path,
    qualification_root: Path,
    timeout_seconds: float = 14_400,
    device: str = "cuda:0",
    ir_path: IRPath | str = DEFAULT_IR_PATH,
    jobs: int = 1,
) -> BatchGPUQualificationGate:
    """Verify the complete chain before the formal SOLAR batch starts."""
    context = _context(
        workspace,
        corpus_manifest_path=corpus_manifest_path,
        orojenesis_home=orojenesis_home,
        qualification_root=qualification_root,
        timeout_seconds=timeout_seconds,
        device=device,
        ir_path=ir_path,
        jobs=jobs,
    )
    return _verify_gate(context, BatchGPUQualificationStage.FULL)


def _context(
    workspace: Path,
    *,
    corpus_manifest_path: Path,
    orojenesis_home: Path,
    qualification_root: Path,
    timeout_seconds: float,
    device: str,
    ir_path: IRPath | str,
    jobs: int,
) -> _SolarQualificationContext:
    if timeout_seconds <= 0:
        raise ValueError("qualification timeout must be positive")
    if jobs <= 0:
        raise ValueError("qualification jobs must be positive")
    root = workspace.resolve()
    corpus = AKACorpusManifest.load(corpus_manifest_path)
    plan = load_execution_plan(
        root / "baseline" / "plan.json", workspace_root=root
    )
    verify_release_source_state(
        corpus.authored_root.parents[1], expected_revision=plan.source_revision
    )
    selected_path = normalize_ir_path(ir_path)
    qualification = require_isolated_qualification_root(
        qualification_root, root
    )
    return _SolarQualificationContext(
        workspace=root,
        corpus=corpus,
        source_revision=plan.source_revision,
        root=qualification,
        orojenesis_home=orojenesis_home.resolve(),
        device=device,
        ir_path=selected_path,
        timeout_seconds=timeout_seconds,
        scope_id=f"{plan.run_id}:{selected_path.value}",
        jobs=jobs,
    )


def _all_workloads(
    context: _SolarQualificationContext,
) -> dict[str, tuple[Definition, Workload]]:
    result: dict[str, tuple[Definition, Workload]] = {}
    for entry in context.corpus.entries:
        if entry.role is not AKACorpusRole.SCORED:
            continue
        problem_path = entry.relative_problem_dir.as_posix()
        problem = context.corpus.authored_root / entry.relative_problem_dir
        definition = load_json_file(Definition, problem / "definition.json")
        workloads = {
            item.uuid: item
            for item in load_jsonl_file(Workload, problem / "workload.jsonl")
        }
        for workload_uuid in entry.workload_uuids:
            result[f"{problem_path}/{workload_uuid}"] = (
                definition,
                workloads[workload_uuid],
            )
    return result


def _selected_item_ids(
    context: _SolarQualificationContext,
    stage: BatchGPUQualificationStage,
) -> tuple[str, ...]:
    all_items = _all_workloads(context)
    if stage is not BatchGPUQualificationStage.CANARY:
        return tuple(all_items)
    by_problem: dict[str, list[tuple[str, Definition, Workload]]] = {}
    for item_id, (definition, workload) in all_items.items():
        problem_path = item_id.rsplit("/", maxsplit=1)[0]
        by_problem.setdefault(problem_path, []).append(
            (item_id, definition, workload)
        )
    selected = []
    for problem_path in sorted(by_problem):
        items = by_problem[problem_path]
        selected.extend(
            select_risk_first_axis_extrema(
                items,
                item_id=lambda item: item[0],
                axes=lambda item: item[1].get_resolved_axes_values(
                    item[2].axes
                ),
            )
        )
    return tuple(item[0] for item in selected)


def _subject_sha256(context: _SolarQualificationContext) -> str:
    return stable_json_checksum(
        {
            "corpus_sha256": sha256_file(context.corpus.path),
            "architecture_sha256": formal_architecture_profile_hash(),
            "source_revision": context.source_revision,
        }
    )


def _configuration_sha256(context: _SolarQualificationContext) -> str:
    return stable_json_checksum(
        {
            "device": context.device,
            "ir_path": context.ir_path,
            "orojenesis_home": str(context.orojenesis_home),
            "timeout_seconds": context.timeout_seconds,
            "jobs": context.jobs,
        }
    )


def _runner_sha256() -> str:
    return stable_json_checksum(
        {"solar_release_qualification": sha256_file(Path(__file__))}
    )


def _run_static(
    context: _SolarQualificationContext,
    item_ids: tuple[str, ...],
) -> BatchGPUQualificationReceipt:
    path = context.root / "static" / "preflight.json"
    payload = {
        "task": LargeBatchGPUTask.SOLAR_RELEASE_BUILD,
        "scope_id": context.scope_id,
        "subject_sha256": _subject_sha256(context),
        "item_ids": item_ids,
        "all_inputs_valid": True,
    }
    atomic_write_json_value(path, payload)
    return BatchGPUQualificationReceipt(
        stage=BatchGPUQualificationStage.STATIC,
        partition="solar-corpus",
        item_ids=item_ids,
        input_sha256=stable_json_checksum(payload),
        artifacts=(qualification_artifact(context.root, path),),
    )


def _run_gpu_stage(
    context: _SolarQualificationContext,
    stage: BatchGPUQualificationStage,
    item_ids: tuple[str, ...],
) -> BatchGPUQualificationReceipt:
    result = audit_corpus_stage_readiness(
        context.corpus.path,
        context.root / stage.value / "readiness",
        device=context.device,
        timeout_seconds=context.timeout_seconds,
        resume=True,
        ir_path=context.ir_path,
        selected_item_ids=frozenset(item_ids),
    )
    if result.status is not CorpusReadinessStatus.READY:
        raise ValueError(f"SOLAR {stage} qualification is incomplete")
    return BatchGPUQualificationReceipt(
        stage=stage,
        partition="solar-corpus",
        item_ids=item_ids,
        input_sha256=_subject_sha256(context),
        artifacts=(
            qualification_artifact(context.root, result.matrix_path),
            qualification_artifact(context.root, result.summary_path),
        ),
    )


def _build_gate(
    context: _SolarQualificationContext,
    stage: BatchGPUQualificationStage,
    parent_hash: str | None,
    receipt: BatchGPUQualificationReceipt,
) -> BatchGPUQualificationGate:
    return BatchGPUQualificationGate(
        task=LargeBatchGPUTask.SOLAR_RELEASE_BUILD,
        stage=stage,
        scope_id=context.scope_id,
        subject_sha256=_subject_sha256(context),
        runner_sha256=_runner_sha256(),
        configuration_sha256=_configuration_sha256(context),
        source_revision=context.source_revision,
        parent_gate_sha256=parent_hash,
        item_ids=receipt.item_ids,
        receipts=(receipt,),
        created_at=utc_timestamp(),
    )


def _require_parent(
    context: _SolarQualificationContext,
    stage: BatchGPUQualificationStage,
) -> str | None:
    parent = qualification_parent_stage(stage)
    if parent is None:
        return None
    _verify_gate(context, parent)
    return sha256_file(qualification_gate_path(context.root, parent))


def _verify_gate(
    context: _SolarQualificationContext,
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
    if not (
        gate.task is LargeBatchGPUTask.SOLAR_RELEASE_BUILD
        and gate.stage is stage
        and gate.scope_id == context.scope_id
        and gate.subject_sha256 == _subject_sha256(context)
        and gate.runner_sha256 == _runner_sha256()
        and gate.configuration_sha256 == _configuration_sha256(context)
        and gate.source_revision == context.source_revision
        and gate.parent_gate_sha256 == parent_hash
        and gate.item_ids == _selected_item_ids(context, stage)
    ):
        raise ValueError(f"SOLAR qualification identity drift: {stage}")
    for artifact in gate.receipts[0].artifacts:
        verify_qualification_artifact(context.root, artifact)
    if stage is not BatchGPUQualificationStage.STATIC:
        _verify_readiness(context, gate)
    return gate


def _verify_readiness(
    context: _SolarQualificationContext,
    gate: BatchGPUQualificationGate,
) -> None:
    receipt = gate.receipts[0]
    matrix_artifact = next(
        item for item in receipt.artifacts if item.path.endswith("matrix.jsonl")
    )
    summary_artifact = next(
        item for item in receipt.artifacts if item.path.endswith("summary.json")
    )
    records = load_jsonl_file(
        CorpusReadinessRecord, context.root / matrix_artifact.path
    )
    summary = load_json_file(
        CorpusReadinessSummary, context.root / summary_artifact.path
    )
    observed = tuple(
        f"{item.problem_path}/{item.workload_uuid}" for item in records
    )
    if set(observed) != set(gate.item_ids) or len(observed) != len(
        gate.item_ids
    ):
        raise ValueError("SOLAR qualification readiness denominator drift")
    if summary.status is not CorpusReadinessStatus.READY:
        raise ValueError("SOLAR qualification readiness is not complete")


__all__ = [
    "require_solar_release_qualification",
    "run_solar_release_qualification",
]
