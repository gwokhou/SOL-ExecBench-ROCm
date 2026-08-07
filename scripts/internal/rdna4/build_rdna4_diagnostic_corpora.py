#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Prepare, collect, and freeze the governed gfx1200 diagnostic corpora."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from sol_execbench.cli.evaluation.compilation import _compiler_provenance
from sol_execbench.cli.evaluation.problem_io import (
    load_problem_inputs,
    resolve_problem_inputs,
)
from sol_execbench.cli.evaluation.profile_mode import ProfileMode
from sol_execbench.cli.sidecars.static_evidence import _static_evidence_payload
from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.corpus_preflight import (
    DiagnosticCorpusDesign,
)
from sol_execbench.core.bench.performance_model.evidence_manifest import (
    PerformanceEvidenceArtifact,
    PerformanceEvidenceArtifactKind,
    PerformanceEvidenceManifest,
    candidate_sha256,
    load_and_verify_performance_evidence_manifest,
)
from sol_execbench.core.bench.performance_model.lifecycle import (
    BlobStore,
    DiagnosticCollectionRunManifest,
    DiagnosticCorpusSnapshotManifest,
    DiagnosticDesignManifest,
    DiagnosticLifecycleStage,
    DiagnosticRetentionClass,
    DiagnosticStageStatus,
    collection_run_id,
    corpus_snapshot_id,
    design_id,
    designs_dir,
    snapshots_dir,
    store_root,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.bench.performance_model.replay_evidence import (
    PerformanceReplayEvidenceSidecar,
)
from sol_execbench.core.bench.performance_model.validation_corpus import (
    BlobArtifactReference,
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
from sol_execbench.core.integrity.schema_versions import SchemaVersion
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
HISTORICAL_UNIVERSE_START = 100
UNIVERSE_CASES_PER_FAMILY = 3 * CASES_PER_PHASE
_PHASE_ROTATIONS: tuple[tuple[Phase, Phase, Phase], ...] = (
    ("point_fit", "conformal", "held_out"),
    ("conformal", "held_out", "point_fit"),
    ("held_out", "point_fit", "conformal"),
)
_TEMPLATE_PACKAGE = "sol_execbench.data.rdna4_diagnostic_templates"


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
            "new-run",
            "promote",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--universe-start", type=int)
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
        "--generation",
        type=int,
        help=(
            "generation counter for `new-run`; defaults to one past the "
            "highest generation already recorded for the frozen design"
        ),
    )
    parser.add_argument(
        "--source-corpus",
        type=Path,
        action="append",
        default=[],
        help=(
            "development then held-out corpora beneath the common --root; "
            "promotion verifies and rebases their artifact references"
        ),
    )
    parser.add_argument("--output", type=Path)
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
            "M": 32 + 8 * (global_index - HISTORICAL_UNIVERSE_START),
            "N": 768,
        }
    return {
        "M": 64 + 16 * (global_index % 10),
        "N": 80 + 16 * ((7 * global_index) % 10),
        "K": 64 + 16 * ((3 * global_index) % 13),
    }


def _cases(role: Role, universe_start: int) -> list[CaseSpec]:
    phases: tuple[Phase, ...] = (
        ("point_fit", "conformal") if role == "development" else ("held_out",)
    )
    cases = []
    for family in FAMILIES:
        for phase in phases:
            selected = [
                global_index
                for global_index in _universe_indices(universe_start)
                if _phase(global_index, universe_start) == phase
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


def _universe_indices(universe_start: int) -> range:
    return range(
        universe_start,
        universe_start + UNIVERSE_CASES_PER_FAMILY,
    )


def _phase(global_index: int, universe_start: int) -> Phase:
    offset = global_index - universe_start
    if not 0 <= offset < UNIVERSE_CASES_PER_FAMILY:
        raise ValueError("global index is outside preregistered universe")
    block_index, position = divmod(offset, 3)
    rotation = _PHASE_ROTATIONS[(block_index // 2) % len(_PHASE_ROTATIONS)]
    return rotation[position]


def _design_payload(universe_start: int) -> dict[str, Any]:
    cases = [
        *_cases("development", universe_start),
        *_cases("held_out", universe_start),
    ]
    return {
        "schema_version": SchemaVersion.RDNA4_DIAGNOSTIC_CORPUS_DESIGN.value,
        "design": "adjacent_shape_stratified_three_way_rotation",
        "universe_start": universe_start,
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
    family: WorkloadKind,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resource_root = files(_TEMPLATE_PACKAGE).joinpath(family.value)
    definition = json.loads(
        resource_root.joinpath("definition.json").read_text(encoding="utf-8")
    )
    solution = json.loads(
        resource_root.joinpath("solution.json").read_text(encoding="utf-8")
    )
    source = solution["sources"][0]
    source["content"] = resource_root.joinpath("kernel.hip").read_text(
        encoding="utf-8"
    )
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
        "schema_version": SchemaVersion.WORKLOAD,
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
    family: WorkloadKind | None = None,
) -> None:
    design = _require_frozen_design(root)
    universe_start = design.universe_start
    selected_families = FAMILIES if family is None else (family,)
    all_cases = [
        case
        for case in [
            *_cases("development", universe_start),
            *_cases("held_out", universe_start),
        ]
        if family is None or case.family is family
    ]
    for selected_family in selected_families:
        problem = root / "problems" / selected_family.value
        definition, solution = _definition_template(selected_family)
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


def _preregister(root: Path, universe_start: int) -> None:
    design_path = root / "design.json"
    design = _design_payload(universe_start)
    if design_path.exists():
        if load_json_value(design_path) != design:
            raise ValueError("existing corpus design differs from current plan")
        print(f"verified frozen design at {design_path}", flush=True)
        return
    atomic_write_json_value(design_path, design)
    design_digest = sha256_file(design_path)
    did = design_id(
        universe_start=universe_start,
        design_payload_sha256=design_digest,
    )
    _write_design_manifest(
        root=root,
        universe_start=universe_start,
        design_payload_sha256=design_digest,
        did=did,
    )
    print(
        f"froze {len(design['cases'])} cases at {design_path} "
        f"(design_id={did})",
        flush=True,
    )


def _source_revision() -> str:
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if git.returncode != 0:
        return "unknown"
    return git.stdout.strip()


def _write_design_manifest(
    *,
    root: Path,
    universe_start: int,
    design_payload_sha256: str,
    did: str,
) -> None:
    directory = designs_dir() / did
    if directory.exists():
        return
    directory.mkdir(parents=True)
    manifest = DiagnosticDesignManifest(
        stage=DiagnosticLifecycleStage.DESIGN,
        stage_id=did,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        source_revision=_source_revision(),
        policy_hashes={"root": str(root.resolve())},
        created_at=datetime.now(UTC).isoformat(),
        universe_start=universe_start,
        design_payload_sha256=design_payload_sha256,
    )
    atomic_write_json_value(
        directory / "manifest.json",
        manifest.model_dump(mode="json"),
    )


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
    """Refuse any same-generation mutation of a frozen held-out corpus.

    Guards rerun-until-acceptance (audit ``build_rdna4_diagnostic_corpora:446``):
    once held-out evidence is frozen it must never be deleted and re-collected,
    because any prior acceptance artifact cited the frozen evidence and its
    ``held_out_corpus_sha256`` would silently drift. Recollection or repair of
    a frozen held-out generation requires a new ``collection_run_id`` via the
    ``new-run`` stage; there is no confirmation escape.
    """
    frozen_held_out = arguments.root / "held_out.json"
    if (
        arguments.stage in {"collect", "solar"}
        and arguments.role == "held_out"
        and arguments.force
        and frozen_held_out.exists()
    ):
        raise ValueError(
            "refusing --force mutation of a frozen held-out corpus "
            f"({frozen_held_out}); recollection or repair requires a new "
            "collection_run_id via the `new-run` stage",
        )


def _execute_cases(arguments: argparse.Namespace) -> None:
    if arguments.role is None:
        raise ValueError(f"{arguments.stage} requires --role")
    design = _require_frozen_design(arguments.root)
    _refuse_frozen_held_out_recollect(arguments)
    selected = _cases(arguments.role, design.universe_start)
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
    design = _require_frozen_design(root)
    cases = _cases(role, design.universe_start)
    for position, case in enumerate(cases, start=1):
        print(f"[{position}/{len(cases)}] repair {case.case_id}", flush=True)
        _repair_case_identity(root, case, compile_identity)


def _freeze(root: Path, role: Role) -> None:
    destination = root / f"{role}.json"
    if destination.exists():
        raise ValueError(
            f"refusing to overwrite the frozen {role} corpus: {destination}",
        )
    design = _require_frozen_design(root)
    corpus = DiagnosticValidationCorpus(
        role=role,
        cases=[
            _validation_case(root, case)
            for case in _cases(role, design.universe_start)
        ],
    )
    atomic_write_json_value(destination, corpus.model_dump(mode="json"))
    did = _frozen_design_id(root, design)
    run_id = _current_collection_run_id(root, design, did)
    snapshot = corpus_snapshot_id(
        collection_run_id=run_id,
        role=role,
        corpus_sha256=sha256_file(destination),
    )
    _write_corpus_snapshot_manifest(
        root=root,
        role=role,
        corpus_sha256=sha256_file(destination),
        case_count=len(corpus.cases),
        run_id=run_id,
        snapshot=snapshot,
    )
    print(
        f"froze {len(corpus.cases)} cases at {destination} "
        f"(corpus_snapshot_id={snapshot})",
        flush=True,
    )


def _frozen_design_id(
    root: Path,
    design: DiagnosticCorpusDesign,
) -> str:
    return design_id(
        universe_start=design.universe_start,
        design_payload_sha256=sha256_file(root / "design.json"),
    )


def _current_collection_run_id(
    root: Path,
    design: DiagnosticCorpusDesign,
    did: str,
) -> str:
    """Return the current collection run identity for a design generation.

    The corpus authoring script records its run identity deterministically
    from the design and a generation counter; generation one is the default
    for a fresh preregistered design.
    """
    del root, design
    return collection_run_id(design_id=did, generation=1)


def _write_corpus_snapshot_manifest(
    *,
    root: Path,
    role: Role,
    corpus_sha256: str,
    case_count: int,
    run_id: str,
    snapshot: str,
) -> None:
    directory = snapshots_dir() / snapshot
    if directory.exists():
        return
    directory.mkdir(parents=True)
    manifest = DiagnosticCorpusSnapshotManifest(
        stage=DiagnosticLifecycleStage.CORPUS_SNAPSHOT,
        stage_id=snapshot,
        status=DiagnosticStageStatus.VERIFIED,
        retention_class=DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        source_revision=_source_revision(),
        parents=(),
        policy_hashes={
            "root": str(root.resolve()),
            "collection_run_id": run_id,
        },
        created_at=datetime.now(UTC).isoformat(),
        role=role,
        corpus_file_sha256=corpus_sha256,
        case_count=case_count,
    )
    atomic_write_json_value(
        directory / "manifest.json",
        manifest.model_dump(mode="json"),
    )


def _require_frozen_design(root: Path) -> DiagnosticCorpusDesign:
    design_path = root / "design.json"
    design = DiagnosticCorpusDesign.model_validate_json(
        design_path.read_text(encoding="utf-8")
    )
    if design.model_dump(mode="json") != _design_payload(design.universe_start):
        raise ValueError("corpus design does not match frozen preregistration")
    return design


def _validate_promoted_reference(
    source_root: Path,
    reference: ValidationArtifactReference,
) -> BlobArtifactReference:
    """Import one source artifact into the blob store and emit its key.

    Promotion targets the content-addressed blob store so the promoted corpus
    does not extend the lifetime of the historical path trees.
    """
    artifact = (source_root / reference.path).resolve()
    if not artifact.is_relative_to(source_root):
        raise ValueError("promoted corpus reference escapes its corpus root")
    if not artifact.is_file():
        raise ValueError(
            f"promoted corpus artifact is missing: {reference.path}"
        )
    if sha256_file(artifact) != reference.sha256:
        raise ValueError(f"promoted corpus hash mismatch: {reference.path}")
    digest = BlobStore(store_root()).put_file(
        artifact,
        expected_sha256=reference.sha256,
    )
    return BlobArtifactReference(
        sha256=digest,
        size_bytes=artifact.stat().st_size,
    )


def _promoted_case(
    case: DiagnosticValidationCase,
    *,
    source_index: int,
    source_root: Path,
) -> DiagnosticValidationCase:
    return case.model_copy(
        update={
            "case_id": f"promoted-{source_index:02d}-{case.case_id}",
            "evidence_manifest": _validate_promoted_reference(
                source_root,
                case.evidence_manifest,
            ),
            "solar_manifest": _validate_promoted_reference(
                source_root,
                case.solar_manifest,
            ),
        }
    )


def _promote_development(
    root: Path,
    source_paths: list[Path],
    output: Path,
) -> None:
    """Combine governed corpora beneath one root into the next development set.

    Source artifacts are imported into the content-addressed blob store, so the
    promoted corpus depends on no historical physical path tree.
    """
    if len(source_paths) != 2:
        raise ValueError("promote requires development then held-out corpora")
    root = root.resolve()
    output = output.resolve()
    if output.parent != root:
        raise ValueError("promoted output must be directly under --root")
    if output.exists():
        raise ValueError("promoted output already exists")
    cases = []
    for source_index, (provided_source, role) in enumerate(
        zip(source_paths, ("development", "held_out"), strict=True)
    ):
        source_path = provided_source.resolve()
        if not source_path.is_relative_to(root):
            raise ValueError("source corpora must remain under --root")
        corpus = load_json_file(
            DiagnosticValidationCorpus,
            source_path,
        )
        if corpus.role != role:
            raise ValueError(
                "source corpus order must be development then held_out"
            )
        cases.extend(
            _promoted_case(
                case,
                source_index=source_index,
                source_root=source_path.parent,
            )
            for case in corpus.cases
        )
    promoted = DiagnosticValidationCorpus(role="development", cases=cases)
    atomic_write_json_value(output, promoted.model_dump(mode="json"))
    print(
        f"promoted {len(promoted.cases)} cases into {output}",
        flush=True,
    )


def _new_run(
    root: Path,
    role: Role,
    generation: int | None,
) -> None:
    """Open a new immutable collection generation for a frozen design.

    A frozen held-out generation is never mutated in place. Recollection or
    repair opens a new ``collection_run_id`` that supersedes the prior
    generation; the operator then collects into a fresh root beneath the same
    design.
    """
    design = _require_frozen_design(root)
    if not (root / f"{role}.json").is_file():
        raise ValueError(
            f"cannot open a new run before freezing {role} evidence",
        )
    did = _frozen_design_id(root, design)
    next_generation = 2 if generation is None else generation
    new_id = collection_run_id(design_id=did, generation=next_generation)
    directory = store_root() / "runs" / new_id
    if directory.exists():
        raise FileExistsError(f"collection run already exists: {directory}")
    prior_id = collection_run_id(design_id=did, generation=next_generation - 1)
    directory.mkdir(parents=True)
    manifest = DiagnosticCollectionRunManifest(
        stage=DiagnosticLifecycleStage.COLLECTION_RUN,
        stage_id=new_id,
        status=DiagnosticStageStatus.RUNNING,
        retention_class=DiagnosticRetentionClass.PROCESS_EVIDENCE,
        source_revision=_source_revision(),
        policy_hashes={
            "design_id": did,
            "generation": str(next_generation),
            "root": str(root.resolve()),
        },
        roles=(role,),
        supersedes=prior_id,
        created_at=datetime.now(UTC).isoformat(),
    )
    atomic_write_json_value(
        directory / "manifest.json",
        manifest.model_dump(mode="json"),
    )
    _mark_run_superseded(prior_id)
    print(
        f"opened collection run {new_id} (supersedes {prior_id})",
        flush=True,
    )


def _mark_run_superseded(run_id: str) -> None:
    manifest_path = store_root() / "runs" / run_id / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = DiagnosticCollectionRunManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8"),
    )
    updated = manifest.model_copy(
        update={"status": DiagnosticStageStatus.SUPERSEDED}
    )
    atomic_write_json_value(
        manifest_path,
        updated.model_dump(mode="json"),
    )


def main() -> int:
    """Run one resumable authoring stage."""
    arguments = _parse_args()
    root = arguments.root.resolve()
    if arguments.stage == "preregister":
        if arguments.universe_start is None:
            raise ValueError("preregister requires --universe-start")
        _preregister(root, arguments.universe_start)
    elif arguments.stage == "prepare":
        _prepare(
            root,
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
    elif arguments.stage == "new-run":
        if arguments.role is None:
            raise ValueError("new-run requires --role")
        _new_run(root, arguments.role, arguments.generation)
    else:
        if arguments.output is None:
            raise ValueError("promote requires --output")
        _promote_development(
            root,
            arguments.source_corpus,
            arguments.output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
