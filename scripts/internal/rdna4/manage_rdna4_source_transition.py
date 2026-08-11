#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Prove stage-scoped source reuse and rebind verified raw development cases.

This control-plane tool is intentionally separate from the GPU collector. Its
own evolution therefore cannot alter the collector bytes cited by qualification
gates or the raw collection implementation whose AST projection it verifies.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

from sol_execbench.core.bench.batch_gpu_qualification import (
    BatchGPUQualificationStage,
)
from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarStatus
from sol_execbench.core.bench.performance_model.builder import _load_calibration
from sol_execbench.core.bench.performance_model.corpus_preflight import (
    DiagnosticCorpusCase,
    DiagnosticCorpusDesign,
)
from sol_execbench.core.bench.performance_model.evidence_manifest import (
    load_and_verify_performance_evidence_manifest,
)
from sol_execbench.core.bench.performance_model.lifecycle.calibration_identity import (
    load_calibration_gpu_identity,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
)
from sol_execbench.core.bench.performance_model.lifecycle.inventory import (
    inventory_regular_tree,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
)
from sol_execbench.core.bench.performance_model.lifecycle.source_review import (
    DiagnosticSourceReview,
    load_and_verify_source_review,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.bench.performance_model.source_transition import (
    DevelopmentCaseRebind,
    DiagnosticDevelopmentCaseRebindReceipt,
    DiagnosticSourceTransitionAttestation,
    QualificationTimeoutTransition,
    SemanticProjectionPair,
    SourcePathStageImpact,
    SourceStageDecision,
    SourceTransitionDisposition,
    SourceTransitionStage,
)
from sol_execbench.core.bench.performance_model.vram_policy import (
    DiagnosticVRAMWorkingSetPolicy,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
    load_json_value,
)
from sol_execbench.core.integrity import (
    sha256_bytes,
    sha256_file,
    stable_json_checksum,
)
from sol_execbench.core.solar_bridge.publication import (
    verified_solar_artifact_paths,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_COLLECTOR_PATH = "scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py"
_RAW_COLLECTION_NODES = (
    "CaseSpec",
    "_case_dir",
    "_run_logged",
    "_expected_case_workload",
    "_verify_resumable_evidence",
    "_collect_case",
    "_remove_trace_artifacts",
)
_REUSABLE_PATHS = (
    "calibration/cal.audit.json",
    "calibration/cal.json",
    "corpus/design.json",
    "vram-policy.json",
)


def _load_collector() -> ModuleType:
    path = _REPOSITORY_ROOT / _COLLECTOR_PATH
    name = "_sol_execbench_rdna4_source_transition_collector"
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load collector module: {path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


collector = _load_collector()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "author",
            "verify",
            "rebind-development-cases",
            "rebind-equal-development-cases",
        ),
    )
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--impact-review", type=Path)
    parser.add_argument("--source-review", type=Path)
    parser.add_argument("--control-plane-review", type=Path)
    parser.add_argument("--base-source-revision")
    parser.add_argument("--target-source-revision")
    parser.add_argument("--base-policy", type=Path)
    parser.add_argument("--comparison-policy", type=Path)
    parser.add_argument("--base-calibration", type=Path)
    parser.add_argument("--base-calibration-audit", type=Path)
    parser.add_argument("--base-design", type=Path)
    parser.add_argument("--comparison-design", type=Path)
    parser.add_argument("--base-problems", type=Path)
    parser.add_argument("--comparison-problems", type=Path)
    parser.add_argument("--old-qualification-timeout", type=int)
    parser.add_argument("--new-qualification-timeout", type=int)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--qualification-root", type=Path)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _run_git(arguments: list[str], *, text: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=text,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        stderr = (
            result.stderr if text else result.stderr.decode(errors="replace")
        )
        raise ValueError(f"git command failed: {stderr.strip()}")
    return result.stdout


def _exact_revision(revision: str) -> str:
    observed = str(
        _run_git(["rev-parse", "--verify", f"{revision}^{{commit}}"], text=True)
    ).strip()
    if observed != revision:
        raise ValueError(f"source revision is not exact: {revision}")
    return observed


def _git_change_identity(line: str) -> tuple[str, str | None, str]:
    fields = line.split("\t")
    status = fields[0][0]
    change = {
        "A": "added",
        "D": "deleted",
        "M": "modified",
        "R": "renamed",
    }.get(status)
    if change is None:
        raise ValueError(f"unsupported git change status: {fields[0]}")
    if status == "R" and len(fields) == 3:
        return change, fields[1], fields[2]
    if len(fields) != 2:
        raise ValueError(f"malformed git change record: {line}")
    return change, None, fields[1]


def _load_impact_review(path: Path) -> tuple[SourcePathStageImpact, ...]:
    value = load_json_value(path)
    if not isinstance(value, list):
        raise ValueError("impact review must be a JSON list")
    return tuple(SourcePathStageImpact.model_validate(item) for item in value)


def _verify_source_diff(
    base: str, target: str, review: tuple[SourcePathStageImpact, ...]
) -> str:
    _exact_revision(base)
    _exact_revision(target)
    output = str(
        _run_git(
            ["diff", "--name-status", "--find-renames", base, target, "--"],
            text=True,
        )
    )
    actual = tuple(
        sorted(
            (_git_change_identity(line) for line in output.splitlines()),
            key=lambda item: item[2],
        )
    )
    declared = tuple(
        (item.change, item.previous_path, item.path) for item in review
    )
    if declared != actual:
        raise ValueError(
            "impact review does not exactly cover the version diff"
        )
    patch = _run_git(["diff", "--binary", "--full-index", base, target, "--"])
    if not isinstance(patch, bytes):
        raise TypeError("binary Git diff unexpectedly returned text")
    return sha256_bytes(patch)


def _policy_projection(
    path: Path,
) -> tuple[str, DiagnosticVRAMWorkingSetPolicy]:
    policy = load_json_file(DiagnosticVRAMWorkingSetPolicy, path)
    payload = policy.model_dump(mode="json")
    payload.pop("created_at")
    payload.pop("source_revision")
    return stable_json_checksum(payload), policy


def _design_projection(path: Path) -> tuple[str, DiagnosticCorpusDesign]:
    design = load_json_file(DiagnosticCorpusDesign, path)
    payload = design.model_dump(mode="json")
    payload.pop("vram_policy_sha256", None)
    return stable_json_checksum(payload), design


def _inventory_projection(
    root: Path,
) -> tuple[str, tuple[DiagnosticLifecycleArtifact, ...]]:
    inventory = inventory_regular_tree(root)
    payload = [item.model_dump(mode="json") for item in inventory]
    return stable_json_checksum(payload), inventory


def _raw_collection_projection(revision: str) -> str:
    source = str(_run_git(["show", f"{revision}:{_COLLECTOR_PATH}"], text=True))
    tree = ast.parse(source)
    selected = {
        node.name: ast.dump(
            node, annotate_fields=True, include_attributes=False
        )
        for node in tree.body
        if isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.name in _RAW_COLLECTION_NODES
    }
    if set(selected) != set(_RAW_COLLECTION_NODES):
        missing = sorted(set(_RAW_COLLECTION_NODES) - set(selected))
        raise ValueError(f"raw collection projection nodes missing: {missing}")
    return stable_json_checksum(selected)


def _artifact(path: Path, relative_path: str) -> DiagnosticLifecycleArtifact:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"reusable artifact is not a regular file: {path}")
    return DiagnosticLifecycleArtifact(
        relative_path=relative_path,
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _validate_calibration(profile: Path, audit: Path) -> None:
    expected_audit = profile.with_name(f"{profile.stem}.audit.json")
    if audit.resolve() != expected_audit.resolve():
        raise ValueError(
            "calibration audit path does not match production loader"
        )
    if _load_calibration(profile) is None:
        raise ValueError("calibration profile is unavailable")
    load_calibration_gpu_identity(
        profile,
        audit,
        expected_purpose=DiagnosticEvidencePurpose.PRODUCTION,
        require_pcie_topology=True,
    )


def _reusable_artifacts(
    arguments: argparse.Namespace,
) -> tuple[DiagnosticLifecycleArtifact, ...]:
    paths = (
        arguments.base_calibration_audit,
        arguments.base_calibration,
        arguments.base_design,
        arguments.base_policy,
    )
    return tuple(
        sorted(
            (
                _artifact(path, relative)
                for path, relative in zip(paths, _REUSABLE_PATHS, strict=True)
            ),
            key=lambda item: item.relative_path,
        )
    )


def _stage_decisions(
    review: tuple[SourcePathStageImpact, ...],
) -> tuple[SourceStageDecision, ...]:
    by_stage = {
        stage: sorted(
            item.path for item in review if stage in item.affected_stages
        )
        for stage in SourceTransitionStage
    }
    return tuple(
        SourceStageDecision(
            stage=stage,
            disposition=(
                SourceTransitionDisposition.CHANGED
                if paths
                else SourceTransitionDisposition.UNCHANGED
            ),
            rationale=(
                f"affected by exact reviewed paths: {','.join(paths)}"
                if paths
                else "no exact reviewed path affects this stage"
            ),
        )
        for stage, paths in by_stage.items()
    )


def _required_author_arguments(arguments: argparse.Namespace) -> None:
    required = (
        "impact_review",
        "base_source_revision",
        "target_source_revision",
        "base_policy",
        "comparison_policy",
        "base_calibration",
        "base_calibration_audit",
        "base_design",
        "comparison_design",
        "base_problems",
        "comparison_problems",
        "old_qualification_timeout",
        "new_qualification_timeout",
    )
    missing = [
        f"--{name.replace('_', '-')}"
        for name in required
        if getattr(arguments, name) is None
    ]
    if missing:
        raise ValueError(f"author requires {', '.join(missing)}")


def _build_attestation(
    arguments: argparse.Namespace,
    *,
    created_at: str,
    review: tuple[SourcePathStageImpact, ...] | None = None,
) -> DiagnosticSourceTransitionAttestation:
    _required_author_arguments(arguments)
    if review is None:
        review = _load_impact_review(arguments.impact_review)
    patch_sha = _verify_source_diff(
        arguments.base_source_revision, arguments.target_source_revision, review
    )
    base_policy_sha, base_policy = _policy_projection(arguments.base_policy)
    comparison_policy_sha, comparison_policy = _policy_projection(
        arguments.comparison_policy
    )
    base_design_sha, _ = _design_projection(arguments.base_design)
    comparison_design_sha, _ = _design_projection(arguments.comparison_design)
    base_problems_sha, base_inventory = _inventory_projection(
        arguments.base_problems
    )
    comparison_problems_sha, comparison_inventory = _inventory_projection(
        arguments.comparison_problems
    )
    _validate_calibration(
        arguments.base_calibration, arguments.base_calibration_audit
    )
    if base_policy.source_revision != arguments.base_source_revision:
        raise ValueError("base policy source revision mismatch")
    return DiagnosticSourceTransitionAttestation(
        base_source_revision=arguments.base_source_revision,
        comparison_artifact_source_revision=comparison_policy.source_revision,
        target_source_revision=arguments.target_source_revision,
        base_policy_path=str(arguments.base_policy.resolve()),
        comparison_policy_path=str(arguments.comparison_policy.resolve()),
        base_calibration_profile_path=str(arguments.base_calibration.resolve()),
        base_calibration_audit_path=str(
            arguments.base_calibration_audit.resolve()
        ),
        base_design_path=str(arguments.base_design.resolve()),
        comparison_design_path=str(arguments.comparison_design.resolve()),
        base_problems_root=str(arguments.base_problems.resolve()),
        comparison_problems_root=str(arguments.comparison_problems.resolve()),
        git_patch_sha256=patch_sha,
        source_changes=review,
        stage_decisions=_stage_decisions(review),
        policy_behavior_projection=SemanticProjectionPair(
            base_sha256=base_policy_sha,
            comparison_sha256=comparison_policy_sha,
        ),
        design_behavior_projection=SemanticProjectionPair(
            base_sha256=base_design_sha,
            comparison_sha256=comparison_design_sha,
        ),
        problems_tree_projection=SemanticProjectionPair(
            base_sha256=base_problems_sha,
            comparison_sha256=comparison_problems_sha,
        ),
        raw_collection_projection=SemanticProjectionPair(
            base_sha256=_raw_collection_projection(
                arguments.base_source_revision
            ),
            comparison_sha256=_raw_collection_projection(
                arguments.target_source_revision
            ),
        ),
        qualification_timeout=QualificationTimeoutTransition(
            old_seconds=arguments.old_qualification_timeout,
            new_seconds=arguments.new_qualification_timeout,
        ),
        reusable_artifacts=_reusable_artifacts(arguments),
        base_problems_inventory=base_inventory,
        comparison_problems_inventory=comparison_inventory,
        created_at=created_at,
    )


def _arguments_from_attestation(
    attestation: DiagnosticSourceTransitionAttestation,
) -> argparse.Namespace:
    return argparse.Namespace(
        impact_review=None,
        base_source_revision=attestation.base_source_revision,
        target_source_revision=attestation.target_source_revision,
        base_policy=Path(attestation.base_policy_path),
        comparison_policy=Path(attestation.comparison_policy_path),
        base_calibration=Path(attestation.base_calibration_profile_path),
        base_calibration_audit=Path(attestation.base_calibration_audit_path),
        base_design=Path(attestation.base_design_path),
        comparison_design=Path(attestation.comparison_design_path),
        base_problems=Path(attestation.base_problems_root),
        comparison_problems=Path(attestation.comparison_problems_root),
        old_qualification_timeout=attestation.qualification_timeout.old_seconds,
        new_qualification_timeout=attestation.qualification_timeout.new_seconds,
    )


def _verify_attestation(path: Path) -> DiagnosticSourceTransitionAttestation:
    attestation = load_json_file(DiagnosticSourceTransitionAttestation, path)
    arguments = _arguments_from_attestation(attestation)
    arguments.impact_review = path
    rebuilt = _build_attestation(
        arguments,
        created_at=attestation.created_at,
        review=attestation.source_changes,
    )
    if rebuilt != attestation:
        raise ValueError("source transition attestation no longer verifies")
    return attestation


def _author(arguments: argparse.Namespace) -> None:
    if arguments.attestation.exists():
        raise ValueError(
            f"refusing to overwrite attestation: {arguments.attestation}"
        )
    attestation = _build_attestation(
        arguments, created_at=datetime.now(UTC).isoformat()
    )
    atomic_write_json_value(
        arguments.attestation, attestation.model_dump(mode="json")
    )
    _verify_attestation(arguments.attestation)


def _case_spec(case: DiagnosticCorpusCase) -> object:
    index = int(case.case_id.rsplit("-", 1)[1])
    family = WorkloadKind(case.family.value)
    axes = collector._shape(family, case.global_index)
    if axes != case.axes:
        raise ValueError(
            f"design axes differ from collector identity: {case.case_id}"
        )
    spec = collector.CaseSpec(
        phase=case.phase.value,
        family=family,
        index=index,
        global_index=case.global_index,
        axes=axes,
    )
    if spec.case_id != case.case_id or spec.workload_uuid != case.workload_uuid:
        raise ValueError(
            f"design case cannot reconstruct collector identity: {case.case_id}"
        )
    return spec


def _selected_design_cases(
    source: DiagnosticCorpusDesign,
    target: DiagnosticCorpusDesign,
    case_ids: list[str],
) -> tuple[DiagnosticCorpusCase, ...]:
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("rebind requires unique --case-id values")
    source_by_id = {case.case_id: case for case in source.cases}
    target_by_id = {case.case_id: case for case in target.cases}
    selected = []
    for case_id in sorted(case_ids):
        if case_id not in source_by_id or source_by_id[
            case_id
        ] != target_by_id.get(case_id):
            raise ValueError(f"source/target design case mismatch: {case_id}")
        case = source_by_id[case_id]
        if case.phase.value == "held_out":
            raise ValueError("held-out evidence cannot use development rebind")
        selected.append(case)
    return tuple(selected)


def _prepare_rebind_case(
    source_root: Path,
    target_root: Path,
    case: DiagnosticCorpusCase,
) -> tuple[DevelopmentCaseRebind, Path, Path]:
    spec = _case_spec(case)
    source_dir = collector._case_dir(source_root, spec)
    target_dir = collector._case_dir(target_root, spec)
    if target_dir.exists() or target_dir.is_symlink():
        raise ValueError(f"refusing to overwrite target case: {target_dir}")
    if case.phase.value not in {"point_fit", "conformal"}:
        raise ValueError("held-out evidence cannot use development rebind")
    inventory = inventory_regular_tree(source_dir)
    evidence = source_dir / "trace.jsonl.performance-evidence.json"
    collector._verify_resumable_evidence(source_root, spec, evidence)
    manifest = load_and_verify_performance_evidence_manifest(
        evidence, require_complete=True
    )
    if manifest.status is not DiagnosticSidecarStatus.AVAILABLE:
        raise ValueError(f"source case evidence is unavailable: {case.case_id}")
    return (
        DevelopmentCaseRebind(
            case_id=case.case_id,
            workload_kind=WorkloadKind(case.family.value),
            phase=case.phase.value,
            workload_uuid=case.workload_uuid,
            evidence_manifest_sha256=sha256_file(evidence),
            inventory=inventory,
        ),
        source_dir,
        target_dir,
    )


def _verify_solar_case_tree(root: Path, case: DiagnosticCorpusCase) -> None:
    spec = _case_spec(case)
    manifest = collector._case_dir(root, spec) / "solar/manifest.yaml"
    if not manifest.is_file():
        raise ValueError(f"SOLAR manifest is missing: {case.case_id}")
    verified_solar_artifact_paths(manifest)


def _copy_to_staging(
    prepared: tuple[tuple[DevelopmentCaseRebind, Path, Path], ...],
    staging: Path,
) -> dict[str, Path]:
    staged: dict[str, Path] = {}
    for record, source, _target in prepared:
        destination = staging / record.case_id
        shutil.copytree(source, destination, copy_function=shutil.copy2)
        if inventory_regular_tree(destination) != record.inventory:
            raise ValueError(f"staged case inventory drift: {record.case_id}")
        staged[record.case_id] = destination
    return staged


def _qualification_gate_artifacts(
    qualification_root: Path,
) -> tuple[DiagnosticLifecycleArtifact, ...]:
    stages = (
        (BatchGPUQualificationStage.STATIC, None),
        (BatchGPUQualificationStage.CANARY, "development"),
        (BatchGPUQualificationStage.FULL, "development"),
    )
    paths = tuple(
        collector._qualification_gate_path(qualification_root, stage, role)
        for stage, role in stages
    )
    return tuple(
        sorted(
            (
                _artifact(
                    path,
                    path.relative_to(qualification_root).as_posix(),
                )
                for path in paths
            ),
            key=lambda item: item.relative_path,
        )
    )


def _require_rebindable_transition(
    attestation: DiagnosticSourceTransitionAttestation,
) -> None:
    decisions = {
        item.stage: item.disposition for item in attestation.stage_decisions
    }
    required = (
        SourceTransitionStage.VRAM_POLICY,
        SourceTransitionStage.CALIBRATION,
        SourceTransitionStage.DESIGN,
        SourceTransitionStage.RAW_COLLECTION,
    )
    changed = [
        stage.value
        for stage in required
        if decisions[stage] is not SourceTransitionDisposition.UNCHANGED
    ]
    if changed:
        raise ValueError(
            "development evidence cannot cross changed stages: "
            + ",".join(changed)
        )


def _rebind(arguments: argparse.Namespace) -> None:
    required = (
        arguments.source_root,
        arguments.target_root,
        arguments.qualification_root,
        arguments.output,
    )
    if any(item is None for item in required):
        raise ValueError(
            "rebind requires --source-root, --target-root, "
            "--qualification-root, and --output"
        )
    if arguments.output.exists():
        raise ValueError(
            f"refusing to overwrite rebind receipt: {arguments.output}"
        )
    attestation = _verify_attestation(arguments.attestation)
    _require_rebindable_transition(attestation)
    source_design_path = arguments.source_root / "design.json"
    target_design_path = arguments.target_root / "design.json"
    source_design = load_json_file(DiagnosticCorpusDesign, source_design_path)
    target_design = load_json_file(DiagnosticCorpusDesign, target_design_path)
    qualification_root = collector._require_qualification_root(
        arguments.target_root, arguments.qualification_root
    )
    collector._require_collection_qualification(
        arguments.target_root, qualification_root, "development"
    )
    qualification_gates = _qualification_gate_artifacts(qualification_root)
    selected = _selected_design_cases(
        source_design, target_design, arguments.case_id
    )
    prepared = tuple(
        _prepare_rebind_case(arguments.source_root, arguments.target_root, case)
        for case in selected
    )
    receipt = DiagnosticDevelopmentCaseRebindReceipt(
        transition_attestation_sha256=sha256_file(arguments.attestation),
        base_source_revision=attestation.base_source_revision,
        target_source_revision=attestation.target_source_revision,
        source_design_sha256=sha256_file(source_design_path),
        target_design_sha256=sha256_file(target_design_path),
        source_root=str(arguments.source_root.resolve()),
        target_root=str(arguments.target_root.resolve()),
        qualification_root=str(qualification_root),
        qualification_gates=qualification_gates,
        cases=tuple(item[0] for item in prepared),
        created_at=datetime.now(UTC).isoformat(),
    )
    with TemporaryDirectory(
        prefix=".source-transition-", dir=arguments.target_root
    ) as name:
        staging = Path(name)
        staged = _copy_to_staging(prepared, staging)
        _commit_staged_cases_with_design(
            prepared,
            staged,
            arguments.target_root,
            selected,
            arguments.output,
            receipt,
        )


def _commit_staged_cases_with_design(
    prepared: tuple[tuple[DevelopmentCaseRebind, Path, Path], ...],
    staged: dict[str, Path],
    target_root: Path,
    cases: tuple[DiagnosticCorpusCase, ...],
    receipt_path: Path,
    receipt: DiagnosticDevelopmentCaseRebindReceipt,
    *,
    verify_solar: bool = False,
) -> None:
    case_by_id = {case.case_id: case for case in cases}
    moved: list[tuple[Path, Path]] = []
    try:
        if any(
            target.exists() or target.is_symlink() for _, _, target in prepared
        ):
            raise ValueError("target case appeared during rebind")
        for record, _source, target in prepared:
            target.parent.mkdir(parents=True, exist_ok=True)
            staged[record.case_id].rename(target)
            moved.append((target, staged[record.case_id]))
        for record, _source, target in prepared:
            if inventory_regular_tree(target) != record.inventory:
                raise ValueError(
                    f"target case inventory drift: {record.case_id}"
                )
            collector._verify_resumable_evidence(
                target_root,
                _case_spec(case_by_id[record.case_id]),
                target / "trace.jsonl.performance-evidence.json",
            )
            if verify_solar:
                _verify_solar_case_tree(target_root, case_by_id[record.case_id])
        atomic_write_json_value(receipt_path, receipt.model_dump(mode="json"))
    except BaseException:
        for target, original in reversed(moved):
            if target.exists() and not original.exists():
                target.rename(original)
        raise


def _equal_development_cases(
    source: DiagnosticCorpusDesign,
    target: DiagnosticCorpusDesign,
) -> tuple[DiagnosticCorpusCase, ...]:
    target_by_id = {case.case_id: case for case in target.cases}
    source_development = tuple(
        case for case in source.cases if case.phase.value != "held_out"
    )
    selected = tuple(
        case
        for case in source_development
        if target_by_id.get(case.case_id) == case
    )
    if not selected or len(selected) == len(source_development):
        raise ValueError(
            "case-granular rebind requires both equal and changed cases"
        )
    return selected


def _verify_equal_rebind_source_chain(
    arguments: argparse.Namespace,
) -> DiagnosticSourceReview:
    prior = _verify_attestation(arguments.attestation)
    review = load_and_verify_source_review(
        arguments.source_review,
        repository_root=_REPOSITORY_ROOT,
    )
    head = str(_run_git(["rev-parse", "HEAD"], text=True)).strip()
    if prior.target_source_revision != review.base_source_revision:
        raise ValueError("equal-case rebind source-transition chain is invalid")
    _verify_control_plane_successor(
        arguments.control_plane_review,
        experiment_revision=review.target_source_revision,
        head=head,
    )
    solar_paths = (
        "src/solar/",
        "src/sol_execbench/core/solar_bridge/",
        "src/sol_execbench/cli/commands/solar.py",
    )
    if any(
        item.path == solar_paths[-1] or item.path.startswith(solar_paths[:2])
        for item in review.source_changes
    ):
        raise ValueError("equal-case SOLAR reuse crosses changed SOLAR source")
    return review


def _verify_control_plane_successor(
    path: Path | None,
    *,
    experiment_revision: str,
    head: str,
) -> None:
    if head == experiment_revision:
        if path is not None:
            raise ValueError(
                "control-plane review is unnecessary at experiment HEAD"
            )
        return
    if path is None:
        raise ValueError(
            "control-plane review is required after experiment HEAD"
        )
    review = load_and_verify_source_review(
        path,
        repository_root=_REPOSITORY_ROOT,
    )
    allowed_paths = {
        "scripts/internal/rdna4/manage_rdna4_source_transition.py",
        "tests/scripts/test_rdna4_source_transition_control_plane.py",
    }
    if (
        review.base_source_revision != experiment_revision
        or review.target_source_revision != head
        or any(item.path not in allowed_paths for item in review.source_changes)
        or any(
            stage is not SourceTransitionStage.GOVERNANCE_CONTROL_PLANE
            for item in review.source_changes
            for stage in item.affected_stages
        )
    ):
        raise ValueError("successor contains non-control-plane source changes")


def _require_reviewed_collection_qualification(
    root: Path,
    qualification_root: Path,
    source_revision: str,
) -> None:
    namespace = vars(collector)
    current_source_revision = namespace["_source_revision"]
    try:
        namespace["_source_revision"] = lambda: source_revision
        collector._require_collection_qualification(
            root,
            qualification_root,
            "development",
        )
    finally:
        namespace["_source_revision"] = current_source_revision


def _rebind_equal(arguments: argparse.Namespace) -> None:
    required = (
        arguments.source_review,
        arguments.source_root,
        arguments.target_root,
        arguments.qualification_root,
        arguments.output,
    )
    if any(item is None for item in required):
        raise ValueError(
            "equal-case rebind requires --source-review, --source-root, "
            "--target-root, --qualification-root, and --output"
        )
    if arguments.output.exists():
        raise ValueError(
            f"refusing to overwrite rebind receipt: {arguments.output}"
        )
    review = _verify_equal_rebind_source_chain(arguments)
    source_design_path = arguments.source_root / "design.json"
    target_design_path = arguments.target_root / "design.json"
    source_design = load_json_file(DiagnosticCorpusDesign, source_design_path)
    target_design = load_json_file(DiagnosticCorpusDesign, target_design_path)
    qualification_root = collector._require_qualification_root(
        arguments.target_root, arguments.qualification_root
    )
    _require_reviewed_collection_qualification(
        arguments.target_root,
        qualification_root,
        review.target_source_revision,
    )
    selected = _equal_development_cases(source_design, target_design)
    for case in selected:
        _verify_solar_case_tree(arguments.source_root, case)
    prepared = tuple(
        _prepare_rebind_case(arguments.source_root, arguments.target_root, case)
        for case in selected
    )
    receipt = DiagnosticDevelopmentCaseRebindReceipt(
        transition_attestation_sha256=sha256_file(arguments.source_review),
        base_source_revision=review.base_source_revision,
        target_source_revision=review.target_source_revision,
        source_design_sha256=sha256_file(source_design_path),
        target_design_sha256=sha256_file(target_design_path),
        source_root=str(arguments.source_root.resolve()),
        target_root=str(arguments.target_root.resolve()),
        qualification_root=str(qualification_root),
        qualification_gates=_qualification_gate_artifacts(qualification_root),
        cases=tuple(item[0] for item in prepared),
        created_at=datetime.now(UTC).isoformat(),
    )
    with TemporaryDirectory(
        prefix=".equal-case-rebind-", dir=arguments.target_root
    ) as name:
        staging = Path(name)
        staged = _copy_to_staging(prepared, staging)
        _commit_staged_cases_with_design(
            prepared,
            staged,
            arguments.target_root,
            selected,
            arguments.output,
            receipt,
            verify_solar=True,
        )


def main() -> None:
    """Run the selected source-transition control-plane stage."""
    arguments = _parse_args()
    if arguments.stage == "author":
        _author(arguments)
    elif arguments.stage == "verify":
        _verify_attestation(arguments.attestation)
    elif arguments.stage == "rebind-equal-development-cases":
        _rebind_equal(arguments)
    else:
        _rebind(arguments)


if __name__ == "__main__":
    main()
