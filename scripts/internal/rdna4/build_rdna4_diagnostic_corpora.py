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
from tempfile import TemporaryDirectory
from typing import Literal

from sol_execbench.cli.evaluation.compilation import _compiler_provenance
from sol_execbench.cli.evaluation.problem_io import (
    load_problem_inputs,
    resolve_problem_inputs,
)
from sol_execbench.cli.evaluation.profile_mode import ProfileMode
from sol_execbench.cli.sidecars.static_evidence import _static_evidence_payload
from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.evidence_manifest import (
    PerformanceEvidenceArtifact,
    PerformanceEvidenceArtifactKind,
    PerformanceEvidenceManifest,
    candidate_sha256,
    load_and_verify_performance_evidence_manifest,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.bench.performance_model.replay_evidence import (
    PerformanceReplayEvidenceSidecar,
)
from sol_execbench.core.bench.performance_model.validation_corpus import (
    DiagnosticValidationCase,
    DiagnosticValidationCorpus,
    ValidationArtifactReference,
    validation_pair_id,
)
from sol_execbench.core.bench.static_kernel.evidence import (
    StaticKernelEvidenceSidecar,
)
from sol_execbench.core.bench.static_kernel.isa_analysis import (
    collect_static_isa_analyses,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    atomic_write_jsonl_values,
    load_json_file,
    load_json_value,
)
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.solar_bridge.performance import (
    load_manifest_semantic_characterization,
)
from sol_execbench.driver.problem_packager import ProblemPackager

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
            "repair-static-identity",
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
        "--template-root",
        type=Path,
        help="Explicit eleven-family smoke template directory.",
    )
    parser.add_argument(
        "--role",
        choices=("development", "held_out"),
    )
    parser.add_argument(
        "--family",
        choices=tuple(family.value for family in FAMILIES),
    )
    parser.add_argument(
        "--case-id",
        help="Select one exact preregistered case ID.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--confirm-recollect-held-out",
        action="store_true",
        help=(
            "explicitly allow --force to overwrite a frozen held-out corpus; "
            "any prior acceptance artifact citing it is thereby invalidated"
        ),
    )
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
        # Each universe case needs a distinct M so no two cases within a phase
        # share a shape-bearing workload_uuid. The prior ``global_index % 16``
        # universe yielded only 16 distinct M values for 20 cases per phase.
        return {
            "M": 32 + 8 * (global_index - UNIVERSE_START),
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
    template_root: Path,
    family: WorkloadKind,
) -> tuple[dict[str, object], dict[str, object]]:
    if family is WorkloadKind.ELEMENTWISE:
        definition_path = (
            repo_root
            / "problems/AMD_AKA/torch2hip/gpumode_sigmoid/definition.json"
        )
    else:
        definition_path = (
            template_root / SMOKE_DIR_NAMES[family] / "problem/definition.json"
        )
    solution_path = template_root / SMOKE_DIR_NAMES[family] / "solution.json"
    definition = load_json_value(definition_path)
    solution = load_json_value(solution_path)
    solution["definition"] = definition["name"]
    for name in definition["axes"]:
        definition["axes"][name] = {
            "type": "var",
            "description": f"Governed {name} extent.",
        }
    return definition, solution


def _workload(case: CaseSpec, output_name: str) -> dict[str, object]:
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
            "predictions": {"type": "random"},
            "targets": {
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
        "schema_version": "sol_execbench.workload.v2",
        "axes": case.axes,
        "inputs": inputs,
        "uuid": case.workload_uuid,
        "checks": [
            {
                "type": "numeric",
                "output": output_name,
                "max_atol": tolerance,
                "max_rtol": tolerance,
                "required_matched_ratio": 1.0,
            }
        ],
    }


def _prepare(
    root: Path,
    repo_root: Path,
    template_root: Path,
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
            template_root,
            selected_family,
        )
        atomic_write_json_value(
            problem / "definition.json",
            definition,
            sort_keys=False,
        )
        atomic_write_json_value(problem / "solution.json", solution)
        output_name = next(iter(definition["outputs"]))
        atomic_write_jsonl_values(
            problem / "workload.jsonl",
            [
                _workload(case, output_name)
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


def _solar_case(
    root: Path,
    case: CaseSpec,
    *,
    force: bool,
    source_corpora: list[Path] | None = None,
) -> None:
    case_dir = _case_dir(root, case)
    solar_dir = case_dir / "solar"
    manifest = solar_dir / "manifest.yaml"
    if manifest.is_file() and not force:
        return
    if force and solar_dir.exists():
        shutil.rmtree(solar_dir)
    if _reuse_solar_case(
        root,
        case,
        source_corpora or [],
    ):
        return
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


def _reuse_solar_case(
    root: Path,
    case: CaseSpec,
    source_corpora: list[Path],
) -> bool:
    """Reuse formal evidence only from an identical frozen case design."""
    target = _case_dir(root, case) / "solar"
    target.parent.mkdir(parents=True, exist_ok=True)
    for source in source_corpora:
        source_root = source.resolve()
        _require_frozen_design(source_root)
        if load_json_value(source_root / "design.json") != load_json_value(
            root / "design.json"
        ):
            raise ValueError("source SOLAR corpus design does not match")
        candidate = _case_dir(source_root, case) / "solar"
        manifest = candidate / "manifest.yaml"
        if not manifest.is_file():
            continue
        definition = load_json_value(
            root / "problems" / case.family.value / "definition.json"
        )
        load_manifest_semantic_characterization(
            manifest,
            workload_uuid=case.workload_uuid,
            definition=str(definition["name"]),
        )
        with TemporaryDirectory(dir=target.parent) as temporary:
            staged = Path(temporary) / "solar"
            shutil.copytree(candidate, staged)
            staged.rename(target)
        return True
    return False


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
        str(ProfileMode.ROCPROFV3_COUNTERS),
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


def _refuse_frozen_held_out_recollect(arguments: argparse.Namespace) -> None:
    """Refuse to overwrite a frozen held-out corpus unless explicitly confirmed.

    Guards rerun-until-acceptance (audit ``build_rdna4_diagnostic_corpora:446``):
    once held-out evidence is frozen it must not be silently deleted and
    re-collected, because any prior acceptance artifact cited the frozen
    evidence and its ``held_out_corpus_sha256`` would silently drift.
    """
    frozen_held_out = arguments.root / "held_out.json"
    if (
        arguments.stage == "collect"
        and arguments.role == "held_out"
        and arguments.force
        and frozen_held_out.exists()
        and not arguments.confirm_recollect_held_out
    ):
        raise ValueError(
            "refusing --force re-collection of a frozen held-out corpus "
            f"({frozen_held_out}); delete it first to explicitly invalidate "
            "prior acceptance, or pass --confirm-recollect-held-out",
        )


def _execute_cases(arguments: argparse.Namespace) -> None:
    if arguments.role is None:
        raise ValueError(f"{arguments.stage} requires --role")
    _require_frozen_design(arguments.root)
    _refuse_frozen_held_out_recollect(arguments)
    selected = _cases(arguments.role)
    if arguments.family is not None:
        selected = [
            case for case in selected if case.family.value == arguments.family
        ]
    if arguments.limit is not None:
        selected = [case for case in selected if case.index < arguments.limit]
    if arguments.case_id is not None:
        selected = [
            case for case in selected if case.case_id == arguments.case_id
        ]
        if not selected:
            raise ValueError(
                f"case ID is not in the selected role/family: {arguments.case_id}"
            )
    for position, case in enumerate(selected, start=1):
        print(
            f"[{position}/{len(selected)}] {arguments.stage} {case.case_id}",
            flush=True,
        )
        if arguments.stage == "solar":
            _solar_case(
                arguments.root,
                case,
                force=arguments.force,
                source_corpora=arguments.source_corpus,
            )
        else:
            _collect_case(arguments.root, case, force=arguments.force)


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


def _compile_identity(root: Path) -> tuple[tuple[str, ...], str]:
    """Reconstruct the deterministic native compile identity in this image."""
    problem = root / "problems" / WorkloadKind.ELEMENTWISE.value
    loaded = load_problem_inputs(
        resolve_problem_inputs(
            problem_dir=problem,
            definition_file=None,
            workload_file=None,
            solution_file=None,
            config_file=None,
        )
    )
    with (
        TemporaryDirectory(dir=root) as temporary,
        ProblemPackager(
            definition=loaded.definition,
            workloads=[loaded.workloads[0]],
            solution=loaded.solution,
            config=loaded.config,
            output_dir=Path(temporary),
            keep_output_dir=True,
        ) as packager,
    ):
        command, _ = packager.compile()
    executable = Path(command[0])
    aliases = {
        str(executable),
        str(executable.resolve()),
        *(str(path) for path in executable.parent.glob("python*")),
    }
    command_hashes = tuple(
        sorted(
            {stable_json_checksum([alias, *command[1:]]) for alias in aliases}
        )
    )
    _, compiler_hash, _ = _compiler_provenance()
    if compiler_hash is None:
        raise ValueError("cannot reconstruct compiler identity")
    return command_hashes, compiler_hash


def _load_static_sidecar(path: Path) -> StaticKernelEvidenceSidecar:
    payload = load_json_value(path)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid static evidence object: {path}")
    payload.pop("summary", None)
    return StaticKernelEvidenceSidecar.model_validate(payload)


def _repair_static_sidecar(path: Path) -> StaticKernelEvidenceSidecar:
    sidecar = _load_static_sidecar(path)
    evidence_dir = path.with_suffix("")
    sources = [
        artifact
        for artifact in sidecar.artifacts
        if artifact.inspectable and artifact.source_path is not None
    ]
    analyses, tool_runs, generated = collect_static_isa_analyses(
        artifacts=sources,
        evidence_root=evidence_dir,
        sidecar_base=evidence_dir,
        timeout_seconds=30.0,
        allow_spec_download=False,
    )
    repaired = sidecar.model_copy(
        update={
            "artifacts": [
                artifact
                for artifact in sidecar.artifacts
                if artifact.producer != "amd-isa"
            ]
            + generated,
            "tool_runs": [
                run for run in sidecar.tool_runs if run.tool_id != "amd-isa"
            ]
            + tool_runs,
            "isa_analyses": analyses,
        }
    )
    atomic_write_json_value(path, _static_evidence_payload(repaired))
    return repaired


def _repair_replay_identity(
    path: Path,
    *,
    candidate_rebindings: dict[str, str],
) -> str:
    replay = load_json_file(PerformanceReplayEvidenceSidecar, path)
    new_candidate_sha256 = candidate_rebindings.get(replay.candidate_sha256)
    if new_candidate_sha256 is None:
        raise ValueError("replay evidence has unrelated candidate identity")
    repaired = replay.model_copy(
        update={"candidate_sha256": new_candidate_sha256}
    )
    atomic_write_json_value(path, repaired.to_dict())
    return new_candidate_sha256


def _repaired_artifact_reference(
    artifact: PerformanceEvidenceArtifact,
    *,
    static_path: Path,
    replay_path: Path,
) -> PerformanceEvidenceArtifact:
    path = {
        PerformanceEvidenceArtifactKind.STATIC_EVIDENCE: static_path,
        PerformanceEvidenceArtifactKind.REPLAY_EVIDENCE: replay_path,
    }.get(artifact.kind)
    if path is None:
        return artifact
    return artifact.model_copy(
        update={
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    )


def _candidate_rebindings(
    manifest: PerformanceEvidenceManifest,
    code_hashes: list[str],
    compile_identity: tuple[tuple[str, ...], str],
) -> dict[str, str]:
    command_hashes, compiler_hash = compile_identity
    result = {
        candidate_sha256(
            solution_sha256=manifest.identity.solution_sha256,
            compile_command_sha256=command_hash,
            compiler_sha256=compiler_hash,
            code_object_sha256=[],
        ): candidate_sha256(
            solution_sha256=manifest.identity.solution_sha256,
            compile_command_sha256=command_hash,
            compiler_sha256=compiler_hash,
            code_object_sha256=code_hashes,
        )
        for command_hash in command_hashes
    }
    result.update({candidate: candidate for candidate in result.values()})
    return result


def _write_repaired_manifest(
    manifest: PerformanceEvidenceManifest,
    *,
    evidence_path: Path,
    static_path: Path,
    replay_path: Path,
    code_hashes: list[str],
    candidate: str,
) -> None:
    artifacts = [
        _repaired_artifact_reference(
            artifact,
            static_path=static_path,
            replay_path=replay_path,
        )
        for artifact in manifest.artifacts
    ]
    repaired = manifest.model_copy(
        update={
            "status": DiagnosticSidecarStatus.AVAILABLE,
            "identity": manifest.identity.model_copy(
                update={"candidate_sha256": candidate}
            ),
            "artifacts": artifacts,
            "code_object_sha256": code_hashes,
            "reason_codes": [],
        }
    )
    atomic_write_json_value(evidence_path, repaired.to_dict())
    load_and_verify_performance_evidence_manifest(evidence_path)


def _repair_case_identity(
    root: Path,
    case: CaseSpec,
    compile_identity: tuple[tuple[str, ...], str],
) -> None:
    case_dir = _case_dir(root, case)
    evidence_path = case_dir / "trace.jsonl.performance-evidence.json"
    static_path = case_dir / "trace.jsonl.static-evidence.json"
    replay_path = case_dir / "trace.jsonl.performance-replay.json"
    manifest = load_and_verify_performance_evidence_manifest(evidence_path)
    unexpected = set(manifest.reason_codes) - {
        "inspectable_code_object_missing"
    }
    if unexpected:
        raise ValueError(
            f"{case.case_id} has unrelated reasons: {sorted(unexpected)}"
        )
    repaired_static = _repair_static_sidecar(static_path)
    code_hashes = sorted(
        {
            analysis.code_object_sha256
            for analysis in repaired_static.isa_analyses
            if analysis.code_object_sha256 is not None
        }
    )
    if not code_hashes:
        raise ValueError(f"{case.case_id} code-object recovery failed")
    static_artifact = manifest.artifact(
        PerformanceEvidenceArtifactKind.STATIC_EVIDENCE
    )
    if static_artifact is None:
        raise ValueError(f"{case.case_id} static evidence reference missing")
    new_candidate = _repair_replay_identity(
        replay_path,
        candidate_rebindings=_candidate_rebindings(
            manifest,
            code_hashes,
            compile_identity,
        ),
    )
    _write_repaired_manifest(
        manifest,
        evidence_path=evidence_path,
        static_path=static_path,
        replay_path=replay_path,
        code_hashes=code_hashes,
        candidate=new_candidate,
    )


def _repair_role(root: Path, role: Role) -> None:
    compile_identity = _compile_identity(root)
    cases = _cases(role)
    for position, case in enumerate(cases, start=1):
        print(f"[{position}/{len(cases)}] repair {case.case_id}", flush=True)
        _repair_case_identity(root, case, compile_identity)


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
                arguments.template_root.resolve()
                if arguments.template_root is not None
                else root.parent / "smoke"
            ),
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
    elif arguments.stage == "repair-static-identity":
        if arguments.role is None:
            raise ValueError("repair-static-identity requires --role")
        _repair_role(root, arguments.role)
    else:
        _promote_development(root, arguments.source_corpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
