#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Build v4 AMD SOL bounds for every static-authority-eligible workload."""

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
from sol_execbench.core.scoring.amd_sol import build_amd_sol_bound_artifact
from sol_execbench.core.scoring.amd_bound_graph.builder import build_static_bound_graph


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


def _output_path(output_dir: Path, problem_id: str, workload_uuid: str) -> Path:
    return output_dir / "bounds-v4" / problem_id / f"{workload_uuid}.amd-sol.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("closure_dir", type=Path)
    parser.add_argument("--hardware-model", required=True, type=Path)
    parser.add_argument("--fusion-validation", required=True, type=Path)
    parser.add_argument(
        "--hardware-model-ref", default="hardware/gfx1200-full-suite.model.json"
    )
    parser.add_argument(
        "--fusion-validation-ref", default="fusion/gfx1200-representative.json"
    )
    args = parser.parse_args()

    coverage = _load(args.closure_dir / "static-coverage.json")
    hardware = load_amd_hardware_model(args.hardware_model)
    fusion = _load(args.fusion_validation)
    fusion_sha256 = sha256_file(args.fusion_validation)
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
            output = _output_path(args.closure_dir, problem_id, workload_uuid)
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
                bound_graph=build_static_bound_graph(definition, workload),
            )
            payload = artifact.to_dict()
            output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if payload["aggregate_bound"]["status"] != "scored":
                warnings = payload.get("warnings", [])
                blocker_codes = ["v4_bound_not_scored"]
                if any(
                    str(warning).startswith("fusion_validation_evidence_missing:")
                    for warning in warnings
                ):
                    blocker_codes.append("fusion_validation_evidence_missing")
                if any(
                    "unknown_hardware_profile" in str(warning) for warning in warnings
                ):
                    blocker_codes.append("hardware_model_profile_missing")
                blocked.append(
                    {
                        "definition": definition.name,
                        "workload_uuid": workload_uuid,
                        "problem_id": problem_id,
                        "blocker_codes": blocker_codes,
                        "bound_status": payload["aggregate_bound"]["status"],
                        "warnings": warnings,
                    }
                )
                continue
            index.append(
                {
                    "definition": definition.name,
                    "workload_uuid": workload_uuid,
                    "problem_id": problem_id,
                    "relative_path": output.relative_to(args.closure_dir).as_posix(),
                    "sha256": sha256_file(output),
                }
            )
    index_payload = {"bounds": index, "blocked": blocked}
    index_payload["payload_sha256"] = hashlib.sha256(
        json.dumps(index_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    index_path = args.closure_dir / "bounds-v4-index.json"
    index_path.write_text(
        json.dumps(index_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    blocker_path = args.closure_dir / "bound-generation-blockers.json"
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
