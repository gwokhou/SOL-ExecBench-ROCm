#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Build v5 AMD SOL floors for every export-authority-eligible workload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.integrity.checksums import sha256_file
from sol_execbench.core.scoring.amd_hardware_models import load_amd_hardware_model
from sol_execbench.core.scoring.amd_sol import (
    AmdSolPerformanceDiagnostics,
    PerformanceProviderResult,
    build_amd_sol_bound_artifact,
)
from sol_execbench.core.scoring.amd_bound_graph.builder import (
    build_authority_bound_graph,
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _workloads(path: Path) -> dict[str, Workload]:
    result: dict[str, Workload] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            continue
        workload = Workload.model_validate_json(line)
        if workload.uuid in result:
            raise ValueError(f"duplicate workload UUID in {path}:{line_number}")
        result[str(workload.uuid)] = workload
    return result


def _provider_results(
    path: Path | None,
) -> dict[tuple[str, str], tuple[PerformanceProviderResult, ...]]:
    """Load real provider rows keyed by definition/workload identity."""
    if path is None:
        return {}
    grouped: dict[tuple[str, str], list[PerformanceProviderResult]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {
            "definition",
            "workload_uuid",
            "provider_result",
        }:
            raise ValueError(f"{path}:{line_number} has invalid provider-result row")
        definition, workload_uuid, raw_result = (
            row["definition"],
            row["workload_uuid"],
            row["provider_result"],
        )
        if not isinstance(definition, str) or not isinstance(workload_uuid, str):
            raise ValueError(f"{path}:{line_number} has invalid provider identity")
        if not isinstance(raw_result, dict):
            raise ValueError(f"{path}:{line_number} provider_result must be an object")
        grouped.setdefault((definition, workload_uuid), []).append(
            PerformanceProviderResult(**raw_result)
        )
    return {key: tuple(results) for key, results in grouped.items()}


def _output_path(output_dir: Path, problem_id: str, workload_uuid: str) -> Path:
    return output_dir / "bounds-v5" / problem_id / f"{workload_uuid}.amd-sol.json"


def _authority_status(payload: dict[str, Any]) -> str:
    floor = payload.get("theoretical_lower_bound")
    if not isinstance(floor, dict) or not isinstance(floor.get("status"), str):
        raise ValueError("generated AMD SOL v5 artifact has no authority floor status")
    return floor["status"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("closure_dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write rebuilt v5 bounds here; defaults to closure_dir",
    )
    parser.add_argument("--hardware-model", required=True, type=Path)
    parser.add_argument("--fusion-validation", required=True, type=Path)
    parser.add_argument(
        "--provider-results-jsonl",
        type=Path,
        help="external provider rows from run_torch_inductor_provider.py or another adapter",
    )
    parser.add_argument(
        "--hardware-model-ref", default="hardware/gfx1200-full-suite.model.json"
    )
    parser.add_argument(
        "--fusion-validation-ref", default="fusion/gfx1200-representative.json"
    )
    args = parser.parse_args()

    coverage_path = args.closure_dir / "authority-coverage.json"
    coverage = _load(coverage_path)
    output_dir = args.output_dir or args.closure_dir
    hardware = load_amd_hardware_model(args.hardware_model)
    fusion = _load(args.fusion_validation)
    fusion_sha256 = sha256_file(args.fusion_validation)
    provider_results = _provider_results(args.provider_results_jsonl)
    by_problem: dict[str, list[dict[str, Any]]] = {}
    for row in coverage["workloads"]:
        if row["blocker_codes"]:
            continue
        by_problem.setdefault(str(row["problem_id"]), []).append(row)

    index: list[dict[str, str]] = []
    blocked: list[dict[str, Any]] = []
    for problem_id, rows in sorted(by_problem.items()):
        problem_dir = args.benchmark_root / problem_id
        definition = Definition.model_validate_json(
            (problem_dir / "definition.json").read_text(encoding="utf-8")
        )
        workloads = _workloads(problem_dir / "workload.jsonl")
        for row in sorted(rows, key=lambda item: str(item["workload_uuid"])):
            workload_uuid = str(row["workload_uuid"])
            workload = workloads.get(workload_uuid)
            if workload is None:
                raise ValueError(f"missing workload {workload_uuid} in {problem_id}")
            output = _output_path(output_dir, problem_id, workload_uuid)
            output.parent.mkdir(parents=True, exist_ok=True)
            artifact = build_amd_sol_bound_artifact(
                definition,
                workload,
                hardware,
                fusion_validation=fusion,
                fusion_validation_ref=args.fusion_validation_ref,
                fusion_validation_sha256=fusion_sha256,
                evidence_path=args.fusion_validation,
                hardware_model_ref=args.hardware_model_ref,
                bound_graph=build_authority_bound_graph(definition, workload),
                performance_diagnostics=AmdSolPerformanceDiagnostics.from_provider_results(
                    provider_results.get((definition.name, workload_uuid), ())
                ),
            )
            payload = artifact.to_dict()
            output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            bound_status = _authority_status(payload)
            if bound_status != "scored":
                warnings = payload.get("warnings", [])
                blocker_codes = ["v5_bound_not_scored"]
                if any(
                    str(warning).startswith("fusion_validation_evidence_missing:")
                    for warning in warnings
                ):
                    blocker_codes.append("fusion_validation_evidence_missing")
                if any(
                    "unknown_hardware_profile" in str(warning) for warning in warnings
                ):
                    blocker_codes.append("hardware_model_profile_missing")
                if any(
                    "unproven_multinode_singleton_partition" in str(warning)
                    for warning in warnings
                ):
                    blocker_codes.append("unproven_multinode_singleton_partition")
                blocked.append(
                    {
                        "definition": definition.name,
                        "workload_uuid": workload_uuid,
                        "problem_id": problem_id,
                        "blocker_codes": blocker_codes,
                        "bound_status": bound_status,
                        "warnings": warnings,
                    }
                )
                continue
            index.append(
                {
                    "definition": definition.name,
                    "workload_uuid": workload_uuid,
                    "problem_id": problem_id,
                    "relative_path": output.relative_to(output_dir).as_posix(),
                    "sha256": sha256_file(output),
                }
            )
    index_payload = {
        "schema_version": "sol_execbench.amd_sol_bound_index.v3",
        "bound_schema_version": "sol_execbench.amd_sol_bound.v5",
        "authority_coverage_ref": str(coverage_path),
        "authority_coverage_sha256": sha256_file(coverage_path),
        "hardware_model_ref": args.hardware_model_ref,
        "hardware_model_sha256": sha256_file(args.hardware_model),
        "fusion_validation_ref": args.fusion_validation_ref,
        "fusion_validation_sha256": fusion_sha256,
        "provider_results_jsonl": (
            str(args.provider_results_jsonl)
            if args.provider_results_jsonl is not None
            else None
        ),
        "provider_results_sha256": (
            sha256_file(args.provider_results_jsonl)
            if args.provider_results_jsonl is not None
            else None
        ),
        "bounds": index,
        "blocked": blocked,
    }
    index_payload["payload_sha256"] = hashlib.sha256(
        json.dumps(index_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    index_path = output_dir / "bounds-v5-index.json"
    index_path.write_text(
        json.dumps(index_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    blocker_path = output_dir / "bound-generation-blockers.json"
    blocker_path.write_text(
        json.dumps(
            {
                "schema_version": "sol_execbench.bound_generation_blockers.v1",
                "workloads": blocked,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "bounds": len(index),
                "blocked": len(blocked),
                "index": str(index_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
