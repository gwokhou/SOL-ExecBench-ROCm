#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Author the AKA-derived seed problem set and its manifest.

This is the offline authoring tool for the problem set derived from AMD
AgentKernelArena (AKA). Each problem's PyTorch reference is AKA's own
correctness oracle (``module_fn``) lifted into a standalone ``def run(...)``;
axes, workloads, and dtypes are chosen per problem under the SOL-ExecBench
paper (arXiv 2603.19173) §3 methodology. Running this script regenerates the
committed problems under ``problems/AMD_AKA/`` and the manifest, recording
AKA per-task checksums when the AKA clone is present.

Usage:
    uv run python scripts/internal/aka_author_seed.py [--aka-root data/AgentKernelArena]
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, cast

import yaml

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.data.schema_versions import BenchmarkArtifactSchema
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.aka_authoring import (
    AKASeedSpec as Spec,
    load_aka_seed_specs,
)
from sol_execbench.core.dataset.aka_compatibility import (
    AKA_EXECUTION_TARGET_SPECS,
)
from sol_execbench.core.dataset.aka_contract import (
    AKA_OFFICIAL_BASELINE_ID,
    AKA_TOLERANCE_CALIBRATION_FILENAME,
    AKAArtifactRole,
    AKACapability,
    AKACorpusRole,
    AKAFusionDepth,
    AKAOfficialScoringStatus,
    AKAOperation,
    AKAPassKind,
    AKASuite,
)
from sol_execbench.core.dataset.aka_corpus import (
    AKA_LICENSE,
    AKA_PROVENANCE_CLASS,
    AKA_REPOSITORY,
    AKA_REVISION,
    FORMAL_ARCHITECTURE,
    FORMAL_ARCHITECTURE_SHA256,
    FORMAL_GFX_TARGET,
)
from sol_execbench.core.dataset.aka_task import (
    correctness_runner_path,
    functional_reference_path,
    read_task,
)
from sol_execbench.core.dataset.aka_tolerance import (
    calibration_checks,
    dtype_default_tolerance,
    load_tolerance_calibration,
    workload_contract_sha256,
)
from sol_execbench.core.dataset.schema_versions import (
    AKA_CORPUS_MANIFEST_SCHEMA_VERSION,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.platform.runtime import resolve_tool_path
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "problems" / "AMD_AKA" / "manifest.yaml"
PROBLEMS_ROOT = REPO_ROOT / "problems" / "AMD_AKA"
CALIBRATION_PATH = PROBLEMS_ROOT / AKA_TOLERANCE_CALIBRATION_FILENAME
AKA_SEED_SPECS_PATH = REPO_ROOT / "scripts/internal/aka_seed_specs.json"


SPECS = list(load_aka_seed_specs(AKA_SEED_SPECS_PATH))


def _artifact_record(
    task_root: Path,
    role: AKAArtifactRole,
    path: Path,
) -> dict[str, str]:
    return {
        "role": str(role),
        "path": path.relative_to(task_root).as_posix(),
        "sha256": sha256_file(path),
    }


def _aka_artifacts(aka_root: Path, spec: Spec) -> list[dict[str, str]]:
    task = read_task(aka_root, spec.task_path)
    runner = correctness_runner_path(task)
    if spec.suite is AKASuite.TORCH2HIP:
        semantic_reference = functional_reference_path(task)
    elif spec.suite is AKASuite.TORCH2FLYDSL:
        semantic_reference = task.root / "model.py"
    elif spec.suite is AKASuite.INSTRUCTION2TRITON:
        semantic_reference = runner
    else:
        raise ValueError(f"unsupported AKA suite for provenance: {spec.suite}")
    return [
        _artifact_record(
            task.root,
            AKAArtifactRole.CONFIG,
            task.root / "config.yaml",
        ),
        _artifact_record(
            task.root,
            AKAArtifactRole.SEMANTIC_REFERENCE,
            semantic_reference,
        ),
        _artifact_record(
            task.root,
            AKAArtifactRole.CORRECTNESS_RUNNER,
            runner,
        ),
    ]


def _workload_checks(
    spec: Spec,
    workload: dict[str, Any],
    uuid: str,
    calibrated: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if calibrated is None or spec.role is AKACorpusRole.TARGET_INCOMPATIBLE:
        checks = [dict(check) for check in workload.get("checks", [])]
    else:
        try:
            checks = [
                check.model_dump(mode="json") for check in calibrated[uuid]
            ]
        except KeyError as exc:
            raise ValueError(f"missing calibrated checks for {uuid}") from exc
    if checks:
        return checks
    return [
        {
            "type": "numeric",
            "output": output,
            **dtype_default_tolerance(str(output_spec["dtype"])).model_dump(
                mode="json",
            ),
        }
        for output, output_spec in spec.outputs.items()
    ]


def _workload_records(
    spec: Spec,
    calibrated: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    records = []
    for index, workload in enumerate(spec.workloads):
        uuid = f"aka-{spec.name}-w{index}"
        workload_inputs = cast(dict[str, Any], workload["inputs"])
        inputs = {
            name: (
                {"type": "scalar", "value": meta["scalar"]}
                if isinstance(meta, dict) and "scalar" in meta
                else dict(meta)
                if isinstance(meta, dict) and "type" in meta
                else {"type": "random"}
            )
            for name, meta in workload_inputs.items()
        }
        record = {
            "schema_version": BenchmarkArtifactSchema.WORKLOAD,
            "axes": workload["axes"],
            "inputs": inputs,
            "checks": _workload_checks(spec, workload, uuid, calibrated),
            "uuid": uuid,
        }
        Workload.model_validate(record)
        records.append(record)
    return records


def _definition_payload(spec: Spec) -> dict[str, Any]:
    payload = {
        "schema_version": BenchmarkArtifactSchema.DEFINITION,
        "name": spec.name,
        "op_type": spec.op_type,
        "description": spec.description,
        "axes": spec.axes,
        "inputs": spec.inputs,
        "outputs": spec.outputs,
        "reference": spec.reference,
    }
    if spec.custom_inputs_entrypoint is not None:
        payload["custom_inputs_entrypoint"] = spec.custom_inputs_entrypoint
    Definition.model_validate(payload)
    return payload


def _write_problem(
    spec: Spec,
    calibrated: dict[str, Any] | None,
    *,
    problems_root: Path = PROBLEMS_ROOT,
) -> dict[str, str]:
    problem_dir = problems_root / spec.suite / spec.name
    problem_dir.mkdir(parents=True, exist_ok=True)
    definition_payload = _definition_payload(spec)
    workload_records = _workload_records(spec, calibrated)
    definition_path = problem_dir / "definition.json"
    workload_path = problem_dir / "workload.jsonl"
    reference_path = problem_dir / "reference.py"
    definition_path.write_text(json.dumps(definition_payload, indent=2) + "\n")
    workload_path.write_text(
        "".join(
            json.dumps(item, sort_keys=True) + "\n" for item in workload_records
        ),
    )
    reference_path.write_text(
        f'"""Standalone PyTorch reference for {spec.name} (debug mirror)."""\n'
        + spec.reference,
    )
    return {
        "path": f"{spec.suite}/{spec.name}",
        "definition_sha256": sha256_file(definition_path),
        "workload_sha256": sha256_file(workload_path),
    }


def _format_authored_references(
    specs: list[Spec],
    records: list[dict[str, str]],
    *,
    problems_root: Path,
) -> None:
    """Ruff-format debug mirrors and bind the same source into Definitions."""
    ruff = resolve_tool_path("ruff")
    if ruff is None:
        raise RuntimeError(
            "Ruff is required to author canonical AKA references",
        )
    reference_paths = [
        problems_root / spec.suite / spec.name / "reference.py"
        for spec in specs
    ]
    completed = run_in_process_group_bounded(
        [str(ruff), "format", *map(str, reference_paths)],
        cwd=REPO_ROOT,
        timeout=120,
        max_capture_bytes=16 * 1024,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "Ruff failed").strip()
        raise RuntimeError(f"could not format AKA references: {detail}")
    for spec, record, reference_path in zip(
        specs,
        records,
        reference_paths,
        strict=True,
    ):
        header = f'"""Standalone PyTorch reference for {spec.name} (debug mirror)."""'
        mirror = reference_path.read_text(encoding="utf-8")
        if not mirror.startswith(header):
            raise ValueError(
                f"formatted AKA reference lost its header: {spec.name}",
            )
        reference = mirror[len(header) :].lstrip("\n")
        if ast.dump(ast.parse(reference)) != ast.dump(
            ast.parse(spec.reference),
        ):
            raise ValueError(
                f"Ruff changed AKA reference semantics: {spec.name}",
            )
        definition_path = reference_path.with_name("definition.json")
        payload = json.loads(definition_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"AKA definition must be an object: {spec.name}")
        payload["reference"] = reference
        Definition.model_validate(payload)
        definition_path.write_text(json.dumps(payload, indent=2) + "\n")
        record["definition_sha256"] = sha256_file(definition_path)


def _rebind_calibration_contracts(
    calibration_path: Path,
    *,
    problems_root: Path,
) -> None:
    """Rebind retained observations after a non-numeric contract change."""
    payload = load_tolerance_calibration(calibration_path)
    raw_records = payload["records"]
    if not isinstance(raw_records, list):
        raise ValueError("calibration records must be a list")
    records: dict[str, dict[str, Any]] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise ValueError("calibration records must be objects")
        record = cast(dict[str, Any], raw_record)
        workload_uuid = str(record["workload_uuid"])
        if workload_uuid in records:
            raise ValueError("calibration records must be unique")
        records[workload_uuid] = record
    observed: set[str] = set()
    for spec in SPECS:
        problem_dir = problems_root / spec.suite / spec.name
        definition = Definition.model_validate_json(
            (problem_dir / "definition.json").read_text(encoding="utf-8"),
        )
        workloads = [
            Workload.model_validate_json(line)
            for line in (problem_dir / "workload.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        for workload in workloads:
            record = records.get(workload.uuid)
            if record is None:
                raise ValueError(f"missing calibration record: {workload.uuid}")
            record["contract_sha256"] = workload_contract_sha256(
                definition,
                workload,
            )
            observed.add(workload.uuid)
    if observed != set(records):
        raise ValueError("calibration contains records outside the AKA corpus")
    atomic_write_json_value(calibration_path, payload)


def _coverage_axes(specs: list[Spec]) -> dict[str, dict[str, int]]:
    def _count(field: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in specs:
            value = str(getattr(s, field))
            out[value] = out.get(value, 0) + 1
        return dict(sorted(out.items()))

    axes = {
        "operation": _count("op_type"),
        "pass_kind": _count("pass_kind"),
        "fusion_depth": _count("fusion_depth"),
        "source_family": _count("source_family"),
        "suite": _count("suite"),
    }
    for name, values in (
        (
            "input_dtype",
            (
                sorted({str(item["dtype"]) for item in spec.inputs.values()})
                for spec in specs
            ),
        ),
        (
            "output_dtype",
            (
                sorted({str(item["dtype"]) for item in spec.outputs.values()})
                for spec in specs
            ),
        ),
        (
            "capability",
            (
                [str(item) for item in _spec_capabilities(spec)]
                for spec in specs
            ),
        ),
    ):
        counts: dict[str, int] = {}
        for items in values:
            for item in items:
                counts[item] = counts.get(item, 0) + 1
        axes[name] = dict(sorted(counts.items()))
    return axes


def _spec_capabilities(spec: Spec) -> tuple[AKACapability, ...]:
    capabilities = set(spec.capabilities)
    output_dtypes = {str(item["dtype"]) for item in spec.outputs.values()}
    if len(spec.outputs) > 1:
        capabilities.add(AKACapability.MULTI_OUTPUT)
    if len(output_dtypes) > 1:
        capabilities.add(AKACapability.MIXED_OUTPUT_DTYPE)
    if any(item.get("shape") == [] for item in spec.outputs.values()):
        capabilities.add(AKACapability.SCALAR_TENSOR_OUTPUT)
    if "uint8" in output_dtypes:
        capabilities.add(AKACapability.UINT8_OUTPUT)
    if any(item.startswith("float8") for item in output_dtypes):
        capabilities.add(AKACapability.FP8_OUTPUT)
    return tuple(sorted(capabilities))


def _manifest_entries(
    specs: list[Spec],
    aka_artifacts: dict[str, list[dict[str, str]]],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for spec in specs:
        entry: dict[str, object] = {
            "slot": spec.name,
            "task_path": spec.task_path,
            "problem_name": spec.name,
            "operation": str(spec.op_type),
            "input_dtypes": sorted(
                {str(item["dtype"]) for item in spec.inputs.values()},
            ),
            "output_dtypes": sorted(
                {str(item["dtype"]) for item in spec.outputs.values()},
            ),
            "capabilities": [
                str(capability) for capability in _spec_capabilities(spec)
            ],
            "pass_kind": str(spec.pass_kind),
            "fusion_depth": str(spec.fusion_depth),
            "source_family": str(spec.source_family),
            "suite": str(spec.suite),
            "role": str(spec.role),
            "workload_uuids": [
                f"aka-{spec.name}-w{i}" for i in range(len(spec.workloads))
            ],
            "aka_artifacts": aka_artifacts[spec.task_path],
            "golden": {},
        }
        if spec.exclusion_reason_code:
            entry["exclusion_reason_code"] = spec.exclusion_reason_code
        entries.append(entry)
    return entries


def _base_coverage_combinations() -> list[dict[str, object]]:
    return [
        {
            "operation": str(AKAOperation.MATMUL),
            "input_dtype": str(DType.FLOAT32),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 1,
        },
        {
            "operation": str(AKAOperation.MATMUL),
            "input_dtype": str(DType.BFLOAT16),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 1,
        },
        {
            "operation": str(AKAOperation.SOFTMAX),
            "input_dtype": str(DType.FLOAT32),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 1,
        },
        {
            "operation": str(AKAOperation.NORM),
            "input_dtype": str(DType.BFLOAT16),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 1,
        },
        {
            "operation": str(AKAOperation.CONV),
            "input_dtype": str(DType.FLOAT32),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 1,
        },
        {
            "operation": str(AKAOperation.ELEMENTWISE),
            "input_dtype": str(DType.FLOAT16),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 1,
        },
        {
            "operation": str(AKAOperation.ATTENTION),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 1,
        },
        {
            "operation": str(AKAOperation.NORM),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 2,
        },
        {"pass": str(AKAPassKind.BACKWARD), "min_count": 1},
        {
            "output_dtype": str(DType.FLOAT8_E4M3FN),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 1,
        },
        {"fusion_depth": str(AKAFusionDepth.FUSED), "min_count": 1},
    ]


def _formal_coverage_combinations() -> list[dict[str, object]]:
    combinations = _base_coverage_combinations()
    combinations.extend(
        [
            {
                "capability": str(capability),
                "role": str(AKACorpusRole.SCORED),
                "min_problems": problems,
                "min_workloads": workloads,
            }
            for capability, problems, workloads in (
                (AKACapability.BOUNDED_INTEGER_INPUT, 1, 5),
                (AKACapability.POSITIVE_INPUT, 1, 5),
                (AKACapability.SIMPLEX_INPUT, 1, 4),
                (AKACapability.SCALAR_TENSOR_OUTPUT, 2, 9),
                (AKACapability.MULTI_OUTPUT, 4, 23),
                (AKACapability.MIXED_OUTPUT_DTYPE, 3, 17),
                (AKACapability.CODE_DISTANCE, 2, 13),
                (AKACapability.PARTIAL_CUSTOM_INPUT, 2, 8),
                (AKACapability.STRUCTURED_OFFSETS, 1, 4),
                (AKACapability.FP8_OUTPUT, 1, 8),
                (AKACapability.UINT8_OUTPUT, 1, 8),
                (AKACapability.RAW_CODE_DISTANCE, 1, 8),
                (AKACapability.COUPLED_TOPK, 1, 4),
            )
        ],
    )
    combinations.extend(
        [
            {
                "operation": str(operation),
                "role": str(AKACorpusRole.SCORED),
                "min_problems": problems,
                "min_workloads": workloads,
            }
            for operation, problems, workloads in (
                (AKAOperation.LOSS, 2, 9),
                (AKAOperation.QUANTIZATION, 2, 13),
                (AKAOperation.ROUTING, 1, 4),
                (AKAOperation.POSITION_ENCODING, 1, 4),
            )
        ],
    )
    return combinations


def _manifest_payload(
    specs: list[Spec],
    records: list[dict[str, str]],
    entries: list[dict[str, object]],
    aka_commit: str,
    *,
    problems_root: Path,
    calibration_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": AKA_CORPUS_MANIFEST_SCHEMA_VERSION,
        "source": {
            "repository": AKA_REPOSITORY,
            "revision": AKA_REVISION,
            "license": AKA_LICENSE,
            "provenance_class": AKA_PROVENANCE_CLASS,
            "aka_commit_sha256": aka_commit,
        },
        "execution_targets": {
            gfx_target: {
                "generation": str(spec["generation"]),
                "supported_tensor_dtypes": [
                    str(dtype) for dtype in spec["supported_tensor_dtypes"]
                ],
            }
            for gfx_target, spec in AKA_EXECUTION_TARGET_SPECS.items()
        },
        "formal_analysis": {
            "architecture_profile": FORMAL_ARCHITECTURE,
            "formal_gfx_target": FORMAL_GFX_TARGET,
            "architecture_profile_sha256": FORMAL_ARCHITECTURE_SHA256,
        },
        "tolerance_calibration": {
            "path": calibration_path.relative_to(problems_root).as_posix(),
            "sha256": sha256_file(calibration_path),
        },
        "official_scoring": {
            "status": str(AKAOfficialScoringStatus.UNAVAILABLE),
            "baseline_id": AKA_OFFICIAL_BASELINE_ID,
            "reason_code": "baseline_v2_release_evidence_pending",
        },
        "formal_coverage_requirements": {
            "axes": _coverage_axes(specs),
            "combinations": _formal_coverage_combinations(),
        },
        "materialized_problems": [
            {
                "path": record["path"],
                "task_path": spec.task_path,
                "definition_sha256": record["definition_sha256"],
                "workload_sha256": record["workload_sha256"],
            }
            for spec, record in zip(specs, records, strict=True)
        ],
        "entries": entries,
    }


def _write_manifest(
    specs: list[Spec],
    records: list[dict[str, str]],
    aka_artifacts: dict[str, list[dict[str, str]]],
    aka_commit: str,
    *,
    problems_root: Path = PROBLEMS_ROOT,
    manifest_path: Path = MANIFEST_PATH,
    calibration_path: Path = CALIBRATION_PATH,
) -> None:
    entries = _manifest_entries(specs, aka_artifacts)
    payload = _manifest_payload(
        specs,
        records,
        entries,
        aka_commit,
        problems_root=problems_root,
        calibration_path=calibration_path,
    )
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def main() -> None:
    """Materialize the authored AKA corpus and its manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aka-root",
        type=Path,
        default=REPO_ROOT / "data" / "AgentKernelArena",
    )
    parser.add_argument(
        "--problems-root",
        type=Path,
        default=PROBLEMS_ROOT,
        help="destination root for authored problems and manifest",
    )
    parser.add_argument(
        "--bootstrap-calibration",
        action="store_true",
        help="write provisional problems for the runtime probe, but not the manifest",
    )
    parser.add_argument(
        "--rebind-calibration-contracts",
        action="store_true",
        help=(
            "rebind contract hashes when a schema or formatting-only change "
            "preserves every measured tolerance observation"
        ),
    )
    args = parser.parse_args()
    aka_root = args.aka_root.resolve()
    problems_root = args.problems_root.resolve()
    manifest_path = problems_root / "manifest.yaml"
    calibration_path = problems_root / AKA_TOLERANCE_CALIBRATION_FILENAME
    if not aka_root.is_dir():
        raise FileNotFoundError(
            "the pinned AKA clone is required to author provenance bindings: "
            f"{aka_root}",
        )

    if args.bootstrap_calibration:
        calibrated = None
    else:
        if not calibration_path.is_file():
            raise FileNotFoundError(
                "run scripts/internal/aka_calibrate_tolerances.py first",
            )
        calibrated = calibration_checks(calibration_path)

    records = []
    aka_artifacts: dict[str, list[dict[str, str]]] = {}
    for spec in SPECS:
        record = _write_problem(spec, calibrated, problems_root=problems_root)
        records.append(record)
        aka_artifacts[spec.task_path] = _aka_artifacts(aka_root, spec)
        print(f"authored {record['path']} ({spec.op_type}/{spec.dtype})")
    _format_authored_references(SPECS, records, problems_root=problems_root)
    if args.bootstrap_calibration:
        print("bootstrap complete; manifest intentionally left unchanged")
        return
    if args.rebind_calibration_contracts:
        _rebind_calibration_contracts(
            calibration_path,
            problems_root=problems_root,
        )
    head_file = aka_root / ".aka-head"
    aka_commit = head_file.read_text().strip() if head_file.is_file() else ""
    _write_manifest(
        SPECS,
        records,
        aka_artifacts,
        aka_commit,
        problems_root=problems_root,
        manifest_path=manifest_path,
        calibration_path=calibration_path,
    )
    print(f"wrote {manifest_path} ({len(SPECS)} problems)")


if __name__ == "__main__":
    main()
