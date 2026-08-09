#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Validate or execute the governed 41-workload SOLAR cross-path focus."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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
from sol_execbench.core.dataset.aka_corpus import (
    AKA_REVISION,
    AKACorpusEntry,
    AKACorpusManifest,
)
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.solar_bridge.corpus_readiness import (
    CorpusReadinessStatus,
    audit_corpus_stage_readiness,
)
from sol_execbench.core.solar_bridge.models import IRPath, SolarWorkerRequest
from sol_execbench.core.solar_bridge.path_comparison import (
    compare_solar_ir_paths,
)
from sol_execbench.core.solar_bridge.runner import run_solar_worker
from sol_execbench.core.timestamps import utc_timestamp

FOCUS_WORKLOAD_COUNTS = {
    "torch2hip/l1n95_cross_entropy": 5,
    "torch2hip/l2n52_conv_activation_batchnorm": 5,
    "torch2hip/14007_kd_loss": 4,
    "torch2flydsl/fused_add_rmsnorm_bf16": 6,
    "torch2flydsl/per_token_i8_quant": 5,
    "torch2flydsl/rope_thd_fwd_bf16": 4,
    "torch2flydsl/dynamic_mxfp8_quant": 8,
    "torch2flydsl/moe_topk_softmax": 4,
}
FOCUS_WORKLOADS = sum(FOCUS_WORKLOAD_COUNTS.values())
IR_PATHS = (IRPath.MAKE_FX_ATEN, IRPath.TORCHVIEW_EXTENDED_EINSUM)


@dataclass(frozen=True, slots=True)
class FocusRunResult:
    """Summary of one complete focused dual-path execution."""

    problems: int
    workloads: int
    path_workloads: int
    generated: int
    resumed: int
    comparison_path: Path


def _focus_entries(corpus: AKACorpusManifest) -> tuple[AKACorpusEntry, ...]:
    by_path = {
        entry.relative_problem_dir.as_posix(): entry for entry in corpus.entries
    }
    missing = set(FOCUS_WORKLOAD_COUNTS) - set(by_path)
    if missing:
        raise ValueError(
            f"AKA manifest lacks focused problems: {sorted(missing)}"
        )
    selected = tuple(by_path[path] for path in FOCUS_WORKLOAD_COUNTS)
    observed_uuids: list[str] = []
    for entry in selected:
        path = entry.relative_problem_dir.as_posix()
        if entry.role is not AKACorpusRole.SCORED:
            raise ValueError(f"focused problem is not scored: {path}")
        expected = FOCUS_WORKLOAD_COUNTS[path]
        if len(entry.workload_uuids) != expected:
            raise ValueError(
                f"focused workload count mismatch for {path}: "
                f"expected {expected}, observed {len(entry.workload_uuids)}"
            )
        observed_uuids.extend(entry.workload_uuids)
    if len(observed_uuids) != FOCUS_WORKLOADS:
        raise ValueError("focused workload denominator mismatch")
    if len(observed_uuids) != len(set(observed_uuids)):
        raise ValueError("focused workloads repeat UUIDs")
    return selected


def focus_summary(corpus: AKACorpusManifest) -> dict[str, object]:
    """Return the CPU-only validated focus inventory."""
    entries = _focus_entries(corpus)
    return {
        "ok": True,
        "problems": len(entries),
        "workloads": sum(len(entry.workload_uuids) for entry in entries),
        "ir_paths": [path.value for path in IR_PATHS],
        "path_workloads": len(IR_PATHS) * FOCUS_WORKLOADS,
        "selection": [
            {
                "problem": entry.relative_problem_dir.as_posix(),
                "workloads": len(entry.workload_uuids),
            }
            for entry in entries
        ],
    }


def _run_entry(
    corpus: AKACorpusManifest,
    entry: AKACorpusEntry,
    *,
    ir_path: IRPath,
    output_root: Path,
    orojenesis_home: Path,
    device: str,
    timeout_seconds: float,
    resume: bool,
) -> tuple[int, int]:
    generated = resumed = 0
    for workload_uuid in entry.workload_uuids:
        output = (
            output_root
            / ir_path.value
            / entry.relative_problem_dir
            / workload_uuid
        )
        if output.exists():
            if not resume:
                raise FileExistsError(
                    f"focused SOLAR output already exists: {output}"
                )
            resumed += 1
            continue
        outcome = run_solar_worker(
            SolarWorkerRequest(
                problem_dir=str(
                    (
                        corpus.authored_root / entry.relative_problem_dir
                    ).resolve()
                ),
                workload_uuid=workload_uuid,
                output_dir=str(output),
                device=device,
                orojenesis_home=str(orojenesis_home),
                ir_path=ir_path,
            ),
            timeout_seconds=timeout_seconds,
        )
        if not outcome.is_formal_publication:
            raise RuntimeError(
                f"focused SOLAR failed for {ir_path.value}/"
                f"{entry.relative_problem_dir}/{workload_uuid}: "
                f"{outcome.stage}/{outcome.reason_code}: {outcome.message}"
            )
        generated += 1
    return generated, resumed


def run_focus(
    corpus: AKACorpusManifest,
    *,
    output_root: Path,
    orojenesis_home: Path,
    device: str = "cuda:0",
    timeout_seconds: float = 14_400,
    resume: bool = False,
) -> FocusRunResult:
    """Execute both fixed paths, then compare only after all 82 succeed."""
    entries = _focus_entries(corpus)
    output_root = output_root.resolve()
    orojenesis_home = orojenesis_home.resolve()
    generated = resumed = 0
    for ir_path in IR_PATHS:
        for entry in entries:
            new, existing = _run_entry(
                corpus,
                entry,
                ir_path=ir_path,
                output_root=output_root,
                orojenesis_home=orojenesis_home,
                device=device,
                timeout_seconds=timeout_seconds,
                resume=resume,
            )
            generated += new
            resumed += existing
    comparison_path = output_root / "path-comparison.json"
    compare_solar_ir_paths(
        output_root / IRPath.MAKE_FX_ATEN.value,
        output_root / IRPath.TORCHVIEW_EXTENDED_EINSUM.value,
        comparison_path,
    )
    return FocusRunResult(
        problems=len(entries),
        workloads=FOCUS_WORKLOADS,
        path_workloads=len(IR_PATHS) * FOCUS_WORKLOADS,
        generated=generated,
        resumed=resumed,
        comparison_path=comparison_path,
    )


def _qualification_root(arguments: argparse.Namespace) -> Path:
    return require_isolated_qualification_root(
        arguments.qualification_root,
        arguments.output,
    )


def _focus_workloads(
    corpus: AKACorpusManifest,
) -> dict[str, tuple[Definition, Workload]]:
    result: dict[str, tuple[Definition, Workload]] = {}
    for entry in _focus_entries(corpus):
        problem_path = entry.relative_problem_dir.as_posix()
        problem = corpus.authored_root / entry.relative_problem_dir
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


def _selected_focus_ids(
    corpus: AKACorpusManifest,
    stage: BatchGPUQualificationStage,
) -> tuple[str, ...]:
    items = _focus_workloads(corpus)
    if stage is not BatchGPUQualificationStage.CANARY:
        selected = tuple(items)
    else:
        by_problem: dict[str, list[tuple[str, Definition, Workload]]] = {}
        for item_id, (definition, workload) in items.items():
            problem_path = item_id.rsplit("/", maxsplit=1)[0]
            by_problem.setdefault(problem_path, []).append(
                (item_id, definition, workload)
            )
        canaries: list[tuple[str, Definition, Workload]] = []
        for problem_path in sorted(by_problem):
            canaries.extend(
                select_risk_first_axis_extrema(
                    by_problem[problem_path],
                    item_id=lambda item: item[0],
                    axes=lambda item: item[1].get_resolved_axes_values(
                        item[2].axes
                    ),
                )
            )
        selected = tuple(item[0] for item in canaries)
    return tuple(
        f"{ir_path.value}/{item_id}"
        for ir_path in IR_PATHS
        for item_id in selected
    )


def _qualification_subject(corpus: AKACorpusManifest) -> str:
    return stable_json_checksum(
        {
            "manifest_sha256": sha256_file(corpus.path),
            "focus": focus_summary(corpus),
        }
    )


def _qualification_configuration(arguments: argparse.Namespace) -> str:
    return stable_json_checksum(
        {
            "device": arguments.device,
            "orojenesis_home": str(arguments.orojenesis_home.resolve()),
            "timeout_seconds": arguments.timeout,
            "ir_paths": [path.value for path in IR_PATHS],
        }
    )


def _run_static_qualification(
    corpus: AKACorpusManifest,
    arguments: argparse.Namespace,
    item_ids: tuple[str, ...],
) -> BatchGPUQualificationReceipt:
    root = _qualification_root(arguments)
    path = root / "static" / "focus.json"
    payload = {
        **focus_summary(corpus),
        "task": LargeBatchGPUTask.SOLAR_CROSS_PATH_FOCUS,
        "subject_sha256": _qualification_subject(corpus),
        "item_ids": item_ids,
    }
    atomic_write_json_value(path, payload)
    return BatchGPUQualificationReceipt(
        stage=BatchGPUQualificationStage.STATIC,
        partition="cross-path-focus",
        item_ids=item_ids,
        input_sha256=stable_json_checksum(payload),
        artifacts=(qualification_artifact(root, path),),
    )


def _run_path_qualification(
    corpus: AKACorpusManifest,
    arguments: argparse.Namespace,
    stage: BatchGPUQualificationStage,
    ir_path: IRPath,
    item_ids: tuple[str, ...],
) -> BatchGPUQualificationReceipt:
    root = _qualification_root(arguments)
    prefix = f"{ir_path.value}/"
    path_items = tuple(item for item in item_ids if item.startswith(prefix))
    corpus_items = frozenset(item.removeprefix(prefix) for item in path_items)
    result = audit_corpus_stage_readiness(
        corpus.path,
        root / stage.value / ir_path.value / "readiness",
        device=arguments.device,
        timeout_seconds=arguments.timeout,
        resume=True,
        ir_path=ir_path,
        selected_item_ids=corpus_items,
    )
    if result.status is not CorpusReadinessStatus.READY:
        raise ValueError(f"cross-path {ir_path.value} qualification failed")
    return BatchGPUQualificationReceipt(
        stage=stage,
        partition=ir_path.value,
        item_ids=path_items,
        input_sha256=_qualification_subject(corpus),
        artifacts=(
            qualification_artifact(root, result.matrix_path),
            qualification_artifact(root, result.summary_path),
        ),
    )


def _run_qualification(
    corpus: AKACorpusManifest,
    arguments: argparse.Namespace,
    stage: BatchGPUQualificationStage,
) -> BatchGPUQualificationGate:
    root = _qualification_root(arguments)
    path = qualification_gate_path(root, stage)
    if path.is_file():
        return _verify_qualification(corpus, arguments, stage)
    parent = qualification_parent_stage(stage)
    parent_hash = None
    if parent is not None:
        _verify_qualification(corpus, arguments, parent)
        parent_hash = sha256_file(qualification_gate_path(root, parent))
    item_ids = _selected_focus_ids(corpus, stage)
    receipts = (
        (_run_static_qualification(corpus, arguments, item_ids),)
        if stage is BatchGPUQualificationStage.STATIC
        else tuple(
            _run_path_qualification(corpus, arguments, stage, ir_path, item_ids)
            for ir_path in IR_PATHS
        )
    )
    gate = BatchGPUQualificationGate(
        task=LargeBatchGPUTask.SOLAR_CROSS_PATH_FOCUS,
        stage=stage,
        scope_id=f"{AKA_REVISION}:cross-path-focus",
        subject_sha256=_qualification_subject(corpus),
        runner_sha256=sha256_file(Path(__file__)),
        configuration_sha256=_qualification_configuration(arguments),
        source_revision=AKA_REVISION,
        parent_gate_sha256=parent_hash,
        item_ids=tuple(
            item for receipt in receipts for item in receipt.item_ids
        ),
        receipts=receipts,
        created_at=utc_timestamp(),
    )
    atomic_write_json_value(path, gate.model_dump(mode="json"))
    return _verify_qualification(corpus, arguments, stage)


def _verify_qualification(
    corpus: AKACorpusManifest,
    arguments: argparse.Namespace,
    stage: BatchGPUQualificationStage,
) -> BatchGPUQualificationGate:
    root = _qualification_root(arguments)
    parent = qualification_parent_stage(stage)
    parent_hash = None
    if parent is not None:
        _verify_qualification(corpus, arguments, parent)
        parent_hash = sha256_file(qualification_gate_path(root, parent))
    gate = load_json_file(
        BatchGPUQualificationGate, qualification_gate_path(root, stage)
    )
    if not (
        gate.task is LargeBatchGPUTask.SOLAR_CROSS_PATH_FOCUS
        and gate.stage is stage
        and gate.scope_id == f"{AKA_REVISION}:cross-path-focus"
        and gate.subject_sha256 == _qualification_subject(corpus)
        and gate.runner_sha256 == sha256_file(Path(__file__))
        and gate.configuration_sha256 == _qualification_configuration(arguments)
        and gate.parent_gate_sha256 == parent_hash
        and set(gate.item_ids) == set(_selected_focus_ids(corpus, stage))
    ):
        raise ValueError(f"cross-path qualification identity drift: {stage}")
    for receipt in gate.receipts:
        for artifact in receipt.artifacts:
            verify_qualification_artifact(root, artifact)
    return gate


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            *(stage.command for stage in BatchGPUQualificationStage),
            "run",
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("problems/AMD_AKA/manifest.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orojenesis-home", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout", type=float, default=14_400)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate the focus or execute and compare both formal IR paths."""
    arguments = _parse_args(argv)
    corpus = AKACorpusManifest.load(arguments.manifest)
    if arguments.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if arguments.stage != "run":
        stage = BatchGPUQualificationStage(
            arguments.stage.removeprefix("qualify-")
        )
        gate = _run_qualification(corpus, arguments, stage)
        print(gate.model_dump_json(indent=2))
        return 0
    _verify_qualification(corpus, arguments, BatchGPUQualificationStage.FULL)
    result = run_focus(
        corpus,
        output_root=arguments.output,
        orojenesis_home=arguments.orojenesis_home,
        device=arguments.device,
        timeout_seconds=arguments.timeout,
        resume=arguments.resume,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
