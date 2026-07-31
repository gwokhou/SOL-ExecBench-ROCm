#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Prepare, collect, and freeze the governed gfx1200 diagnostic corpora."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.evidence_manifest import (
    load_and_verify_performance_evidence_manifest,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.bench.performance_model.validation_corpus import (
    DiagnosticValidationCase,
    DiagnosticValidationCorpus,
    ValidationArtifactReference,
    validation_pair_id,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    atomic_write_jsonl_values,
    load_json_file,
    load_json_value,
)
from sol_execbench.core.integrity import sha256_file

Role = Literal["development", "held_out"]
Phase = Literal["point_fit", "conformal", "held_out"]
FAMILIES = (
    WorkloadKind.ELEMENTWISE,
    WorkloadKind.TRANSPOSE,
    WorkloadKind.REDUCTION,
    WorkloadKind.MATMUL,
    WorkloadKind.SOFTMAX,
    WorkloadKind.CROSS_ENTROPY,
    WorkloadKind.INDEXED_READ,
    WorkloadKind.INDEXED_UPDATE,
    WorkloadKind.COMPOSITE,
    WorkloadKind.TRANSFORMER,
    WorkloadKind.CONCURRENT,
)
CASES_PER_PHASE = 20
UNIVERSE_START = 100
UNIVERSE_CASES_PER_FAMILY = 3 * CASES_PER_PHASE
CORPUS_DESIGN_SCHEMA_VERSION = "rdna4_diagnostic_corpus_design.v1"
_PHASE_ROTATIONS: tuple[tuple[Phase, Phase, Phase], ...] = (
    ("point_fit", "conformal", "held_out"),
    ("conformal", "held_out", "point_fit"),
    ("held_out", "point_fit", "conformal"),
)
SMOKE_DIR_NAMES = {
    WorkloadKind.ELEMENTWISE: "elementwise",
    WorkloadKind.TRANSPOSE: "transpose",
    WorkloadKind.REDUCTION: "reduction",
    WorkloadKind.MATMUL: "matmul",
    WorkloadKind.SOFTMAX: "softmax",
    WorkloadKind.CROSS_ENTROPY: "cross_entropy",
    WorkloadKind.INDEXED_READ: "indexed_read",
    WorkloadKind.INDEXED_UPDATE: "indexed_update",
    WorkloadKind.COMPOSITE: "composite",
    WorkloadKind.TRANSFORMER: "transformer",
    WorkloadKind.CONCURRENT: "concurrent",
}


@dataclass(frozen=True, slots=True)
class CaseSpec:
    """One workload in the preregistered three-way stratified split."""

    phase: Phase
    family: WorkloadKind
    index: int
    global_index: int
    axes: dict[str, int]

    @property
    def role(self) -> Role:
        """Return the governed corpus role for this phase."""
        return "held_out" if self.phase == "held_out" else "development"

    @property
    def case_id(self) -> str:
        """Return the stable corpus-local case identifier."""
        return f"{self.phase}-{self.family.value}-{self.index:02d}"

    @property
    def workload_uuid(self) -> str:
        """Return the stable shape-bearing workload identifier."""
        dimensions = "x".join(str(value) for value in self.axes.values())
        return f"diagnostic-{self.family.value}-{self.phase}-{dimensions}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "preregister",
            "prepare",
            "solar",
            "collect",
            "freeze",
            "promote",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument(
        "--role",
        choices=("development", "held_out"),
    )
    parser.add_argument(
        "--family",
        choices=tuple(family.value for family in FAMILIES),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--source-corpus",
        type=Path,
        action="append",
        default=[],
    )
    return parser.parse_args()


def _shape(family: WorkloadKind, global_index: int) -> dict[str, int]:
    if family is WorkloadKind.ELEMENTWISE:
        return {
            "M": 512 + 64 * global_index,
            "N": 768 + 32 * ((17 * global_index) % 40),
        }
    if family is WorkloadKind.TRANSPOSE:
        return {
            "M": 513 + 53 * global_index,
            "N": 769 + 37 * ((13 * global_index) % 40),
        }
    if family is WorkloadKind.REDUCTION:
        calibrated_widths = (32, 64, 128, 256, 512, 1024)
        return {
            "M": 512 + 96 * global_index,
            "N": calibrated_widths[global_index % len(calibrated_widths)],
        }
    if family in {WorkloadKind.SOFTMAX, WorkloadKind.CROSS_ENTROPY}:
        return {
            "M": 128 + 16 * global_index,
            "N": 64 + 32 * (global_index % 16),
        }
    if family in {WorkloadKind.INDEXED_READ, WorkloadKind.INDEXED_UPDATE}:
        return {
            "M": 1024 + 128 * global_index,
            "N": 256 + 32 * (global_index % 16),
        }
    if family in {
        WorkloadKind.COMPOSITE,
        WorkloadKind.TRANSFORMER,
        WorkloadKind.CONCURRENT,
    }:
        return {
            "M": 32 + 8 * (global_index % 16),
            "N": 768,
        }
    return {
        "M": 64 + 16 * (global_index % 10),
        "N": 80 + 16 * ((7 * global_index) % 10),
        "K": 64 + 16 * ((3 * global_index) % 13),
    }


def _cases(role: Role) -> list[CaseSpec]:
    phases: tuple[Phase, ...] = (
        ("point_fit", "conformal") if role == "development" else ("held_out",)
    )
    cases = []
    for family in FAMILIES:
        for phase in phases:
            selected = [
                global_index
                for global_index in _universe_indices()
                if _phase(global_index) == phase
            ]
            cases.extend(
                CaseSpec(
                    phase=phase,
                    family=family,
                    index=index,
                    global_index=global_index,
                    axes=_shape(family, global_index),
                )
                for index, global_index in enumerate(selected)
            )
    return cases


def _universe_indices() -> range:
    return range(UNIVERSE_START, UNIVERSE_START + UNIVERSE_CASES_PER_FAMILY)


def _phase(global_index: int) -> Phase:
    offset = global_index - UNIVERSE_START
    if not 0 <= offset < UNIVERSE_CASES_PER_FAMILY:
        raise ValueError("global index is outside preregistered universe")
    block_index, position = divmod(offset, 3)
    rotation = _PHASE_ROTATIONS[(block_index // 2) % len(_PHASE_ROTATIONS)]
    return rotation[position]


def _design_payload() -> dict[str, object]:
    cases = [*_cases("development"), *_cases("held_out")]
    return {
        "schema_version": CORPUS_DESIGN_SCHEMA_VERSION,
        "design": "adjacent_shape_stratified_three_way_rotation",
        "universe_start": UNIVERSE_START,
        "universe_cases_per_family": UNIVERSE_CASES_PER_FAMILY,
        "cases_per_family": {
            "point_fit": CASES_PER_PHASE,
            "conformal": CASES_PER_PHASE,
            "held_out": CASES_PER_PHASE,
        },
        "configuration_frozen_before_collection": True,
        "cases": [
            {
                "case_id": case.case_id,
                "family": case.family.value,
                "phase": case.phase,
                "role": case.role,
                "global_index": case.global_index,
                "axes": case.axes,
                "workload_uuid": case.workload_uuid,
            }
            for case in cases
        ],
    }


def _definition_template(
    repo_root: Path,
    root: Path,
    family: WorkloadKind,
) -> tuple[dict[str, object], dict[str, object]]:
    if family is WorkloadKind.ELEMENTWISE:
        definition_path = (
            repo_root
            / "problems/AMD_AKA/torch2hip/gpumode_sigmoid/definition.json"
        )
    else:
        definition_path = (
            root.parent
            / "smoke"
            / SMOKE_DIR_NAMES[family]
            / "problem/definition.json"
        )
    solution_path = (
        root.parent / "smoke" / SMOKE_DIR_NAMES[family] / "solution.json"
    )
    definition = load_json_value(definition_path)
    solution = load_json_value(solution_path)
    for name in definition["axes"]:
        definition["axes"][name] = {
            "type": "var",
            "description": f"Governed {name} extent.",
        }
    return definition, solution


def _workload(case: CaseSpec) -> dict[str, object]:
    if case.family is WorkloadKind.ELEMENTWISE:
        inputs = {
            "a": {"type": "scalar", "value": 1.0},
            "max": {"type": "scalar", "value": 10.0},
            "v": {"type": "random"},
        }
        tolerance = 1.25e-5
    elif case.family is WorkloadKind.MATMUL:
        inputs = {"a": {"type": "random"}, "b": {"type": "random"}}
        tolerance = 0.02
    elif case.family is WorkloadKind.CROSS_ENTROPY:
        inputs = {
            "logits": {"type": "random"},
            "target": {
                "type": "generated",
                "generator": {"type": "integer", "low": 0, "high": "N"},
            },
        }
        tolerance = 0.01
    elif case.family is WorkloadKind.INDEXED_READ:
        inputs = {
            "source": {"type": "random"},
            "indices": {
                "type": "generated",
                "generator": {"type": "integer", "low": 0, "high": "M"},
            },
        }
        tolerance = 0.01
    elif case.family is WorkloadKind.INDEXED_UPDATE:
        inputs = {
            "output": {"type": "random"},
            "indices": {
                "type": "generated",
                "generator": {"type": "integer", "low": 0, "high": "M"},
            },
            "updates": {"type": "random"},
        }
        tolerance = 0.01
    elif case.family is WorkloadKind.TRANSFORMER:
        inputs = {
            "input": {"type": "random"},
            "weight": {"type": "random"},
        }
        tolerance = 0.003
    else:
        inputs = {"input": {"type": "random"}}
        tolerance = 0.01 if case.family is WorkloadKind.REDUCTION else 1e-5
    return {
        "schema_version": "sol_execbench.workload.v1",
        "axes": case.axes,
        "inputs": inputs,
        "uuid": case.workload_uuid,
        "checks": [
            {
                "type": "numeric",
                "output": "output",
                "max_atol": tolerance,
                "max_rtol": tolerance,
                "required_matched_ratio": 1.0,
            }
        ],
    }


def _prepare(
    root: Path,
    repo_root: Path,
    family: WorkloadKind | None = None,
) -> None:
    _require_frozen_design(root)
    selected_families = FAMILIES if family is None else (family,)
    all_cases = [
        case
        for case in [*_cases("development"), *_cases("held_out")]
        if family is None or case.family is family
    ]
    for selected_family in selected_families:
        problem = root / "problems" / selected_family.value
        definition, solution = _definition_template(
            repo_root,
            root,
            selected_family,
        )
        atomic_write_json_value(
            problem / "definition.json",
            definition,
            sort_keys=False,
        )
        atomic_write_json_value(problem / "solution.json", solution)
        atomic_write_jsonl_values(
            problem / "workload.jsonl",
            [
                _workload(case)
                for case in all_cases
                if case.family is selected_family
            ],
        )
    print(
        f"prepared {len(all_cases)} disjoint workloads under {root}", flush=True
    )


def _preregister(root: Path) -> None:
    design_path = root / "design.json"
    design = _design_payload()
    if design_path.exists():
        if load_json_value(design_path) != design:
            raise ValueError("existing corpus design differs from current plan")
        print(f"verified frozen design at {design_path}", flush=True)
        return
    atomic_write_json_value(design_path, design)
    print(f"froze {len(design['cases'])} cases at {design_path}", flush=True)


def _case_dir(root: Path, case: CaseSpec) -> Path:
    return root / "cases" / case.phase / case.family.value / case.case_id


def _run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=14_400,
            check=False,
        )
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8")[-4096:]
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{tail}"
        )


def _solar_case(root: Path, case: CaseSpec, *, force: bool) -> None:
    case_dir = _case_dir(root, case)
    solar_dir = case_dir / "solar"
    manifest = solar_dir / "manifest.yaml"
    if manifest.is_file() and not force:
        return
    if force and solar_dir.exists():
        shutil.rmtree(solar_dir)
    command = [
        "sol-execbench",
        "--format",
        "json",
        "solar",
        "analyze",
        str(root / "problems" / case.family.value),
        "--workload",
        case.workload_uuid,
        "--output",
        str(case_dir / "solar"),
        "--backend",
        "make_fx_aten",
    ]
    _run_logged(command, case_dir / "solar.log")


def _collect_case(root: Path, case: CaseSpec, *, force: bool) -> None:
    case_dir = _case_dir(root, case)
    trace = case_dir / "trace.jsonl"
    evidence = trace.with_name(f"{trace.name}.performance-evidence.json")
    if evidence.is_file() and not force:
        return
    if force:
        _remove_trace_artifacts(trace)
    problem = root / "problems" / case.family.value
    command = [
        "sol-execbench",
        "--format",
        "json",
        "evaluate",
        str(problem),
        "--solution",
        str(problem / "solution.json"),
        "--trace-output",
        str(trace),
        "--workload-uuid",
        case.workload_uuid,
        "--lock-clocks",
        "--profile",
        "rocprofv3-counters",
        "--static-evidence",
        "auto",
    ]
    _run_logged(command, case_dir / "collect.log")


def _remove_trace_artifacts(trace: Path) -> None:
    """Remove only artifacts rooted at one explicitly forced trace path."""
    for artifact in trace.parent.glob(f"{trace.name}*"):
        if artifact.is_dir():
            shutil.rmtree(artifact)
        else:
            artifact.unlink()


def _execute_cases(arguments: argparse.Namespace) -> None:
    if arguments.role is None:
        raise ValueError(f"{arguments.stage} requires --role")
    _require_frozen_design(arguments.root)
    selected = _cases(arguments.role)
    if arguments.family is not None:
        selected = [
            case for case in selected if case.family.value == arguments.family
        ]
    if arguments.limit is not None:
        selected = [case for case in selected if case.index < arguments.limit]
    operation = _solar_case if arguments.stage == "solar" else _collect_case
    for position, case in enumerate(selected, start=1):
        print(
            f"[{position}/{len(selected)}] {arguments.stage} {case.case_id}",
            flush=True,
        )
        operation(arguments.root, case, force=arguments.force)


def _validation_case(root: Path, case: CaseSpec) -> DiagnosticValidationCase:
    case_dir = _case_dir(root, case)
    evidence_path = case_dir / "trace.jsonl.performance-evidence.json"
    solar_path = case_dir / "solar/manifest.yaml"
    manifest = load_and_verify_performance_evidence_manifest(evidence_path)
    if manifest.status is not DiagnosticSidecarStatus.AVAILABLE:
        raise ValueError(
            f"{case.case_id} evidence is {manifest.status}: "
            f"{manifest.reason_codes}"
        )
    if manifest.identity.workload_uuid != case.workload_uuid:
        raise ValueError(f"{case.case_id} workload identity mismatch")
    relative_evidence = evidence_path.relative_to(root).as_posix()
    relative_solar = solar_path.relative_to(root).as_posix()
    return DiagnosticValidationCase(
        case_id=case.case_id,
        pair_id=validation_pair_id(
            workload_sha256=manifest.identity.workload_sha256,
            candidate_sha256=manifest.identity.candidate_sha256,
        ),
        workload_kind=case.family,
        evidence_manifest=ValidationArtifactReference(
            path=relative_evidence,
            sha256=sha256_file(evidence_path),
        ),
        solar_manifest=ValidationArtifactReference(
            path=relative_solar,
            sha256=sha256_file(solar_path),
        ),
        gold_action_codes=(
            ["restore_wmma_path"] if case.family is WorkloadKind.MATMUL else []
        ),
    )


def _freeze(root: Path, role: Role) -> None:
    _require_frozen_design(root)
    corpus = DiagnosticValidationCorpus(
        role=role,
        cases=[_validation_case(root, case) for case in _cases(role)],
    )
    destination = root / f"{role}.json"
    atomic_write_json_value(destination, corpus.model_dump(mode="json"))
    print(
        f"froze {len(corpus.cases)} cases at {destination}",
        flush=True,
    )


def _require_frozen_design(root: Path) -> None:
    design_path = root / "design.json"
    if load_json_value(design_path) != _design_payload():
        raise ValueError("corpus design does not match frozen preregistration")


def _promote_development(
    root: Path,
    source_paths: list[Path],
) -> None:
    """Combine prior governed corpora into the next development corpus."""
    if len(source_paths) < 2:
        raise ValueError("promote requires at least two --source-corpus inputs")
    cases = []
    for source_index, source_path in enumerate(source_paths):
        corpus = load_json_file(
            DiagnosticValidationCorpus,
            source_path.resolve(),
        )
        cases.extend(
            case.model_copy(
                update={
                    "case_id": (f"promoted-{source_index:02d}-{case.case_id}")
                }
            )
            for case in corpus.cases
        )
    promoted = DiagnosticValidationCorpus(role="development", cases=cases)
    destination = root / "development.json"
    atomic_write_json_value(destination, promoted.model_dump(mode="json"))
    print(
        f"promoted {len(promoted.cases)} cases into {destination}",
        flush=True,
    )


def main() -> int:
    """Run one resumable authoring stage."""
    arguments = _parse_args()
    root = arguments.root.resolve()
    if arguments.stage == "preregister":
        _preregister(root)
    elif arguments.stage == "prepare":
        _prepare(
            root,
            arguments.repo_root.resolve(),
            (
                WorkloadKind(arguments.family)
                if arguments.family is not None
                else None
            ),
        )
    elif arguments.stage in {"solar", "collect"}:
        _execute_cases(arguments)
    elif arguments.stage == "freeze":
        if arguments.role is None:
            raise ValueError("freeze requires --role")
        _freeze(root, arguments.role)
    else:
        _promote_development(root, arguments.source_corpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
