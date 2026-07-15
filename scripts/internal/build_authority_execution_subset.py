#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Turn full-suite authority coverage into an execution subset and blocker ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sol_execbench.core.data.json_utils import stable_model_json
from sol_execbench.core.dataset.ready_subset import (
    ReadySubset,
    ReadySubsetClaimBoundary,
    ReadySubsetDenominator,
    ReadySubsetExclusionReason,
    ReadySubsetProblemRef,
    ReadySubsetWorkloadRef,
)
from sol_execbench.core.scoring.full_suite import (
    validate_full_suite_coverage,
)
from sol_execbench.core.timestamps import utc_timestamp


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def build_inputs(
    *,
    dataset_root: Path,
    manifest: dict[str, Any],
    coverage: dict[str, Any],
    requirements: dict[str, Any],
    bounds_by_key: dict[tuple[str, str], dict[str, str]] | None = None,
    hardware_model: Path | None = None,
    additional_blockers_by_key: dict[tuple[str, str], tuple[str, ...]] | None = None,
) -> tuple[ReadySubset, dict[str, Any]]:
    """Build the executable export-authority subset and complete blocker ledger."""
    validate_full_suite_coverage(coverage, requirements, manifest)
    grouped: defaultdict[str, list[ReadySubsetWorkloadRef]] = defaultdict(list)
    meta: dict[str, tuple[str, str]] = {}
    exclusions: list[ReadySubsetExclusionReason] = []
    authority_rows: list[dict[str, Any]] = []
    for row in coverage["workloads"]:
        problem_id = str(row["problem_id"])
        blockers = tuple(
            dict.fromkeys(
                (
                    *[str(item) for item in row["blocker_codes"]],
                    *(additional_blockers_by_key or {}).get(
                        (str(row["definition"]), str(row["workload_uuid"])),
                        (),
                    ),
                )
            )
        )
        common = {
            "category": str(row["category"]),
            "problem_id": problem_id,
            "problem_path": problem_id,
            "workload_uuid": str(row["workload_uuid"]),
            "row_index": int(row["row_index"]),
        }
        if blockers:
            exclusions.append(
                ReadySubsetExclusionReason(
                    **common,
                    readiness_class="static_bound_blocked",
                    readiness_status="blocked",
                    reason_codes=list(blockers),
                    blocker_types=["amd_sol_static_coverage"],
                    message="; ".join(blockers),
                )
            )
            authority_rows.append(
                {
                    "definition": row["definition"],
                    "workload_uuid": row["workload_uuid"],
                    "official_blockers": list(blockers),
                }
            )
            continue
        grouped[problem_id].append(
            ReadySubsetWorkloadRef(
                uuid=str(row["workload_uuid"]),
                row_index=int(row["row_index"]),
                readiness_class="static_authority_eligible",
                readiness_status="ready",
                closure_inputs={
                    **common,
                    "readiness_checksum": None,
                },
            )
        )
        meta[problem_id] = (str(row["category"]), problem_id)
        key = (str(row["definition"]), str(row["workload_uuid"]))
        bound = (bounds_by_key or {}).get(key)
        if bound is None or hardware_model is None:
            # A selected workload cannot yet be official until its v5 bound is
            # materialized. Keep that blocker explicit so a partial trace
            # cannot be promoted by the release-baseline builder.
            authority_rows.append(
                {
                    "definition": row["definition"],
                    "workload_uuid": row["workload_uuid"],
                    "official_blockers": ["missing_v4_bound_evidence"],
                }
            )
        else:
            authority_rows.append(
                {
                    "definition": row["definition"],
                    "workload_uuid": row["workload_uuid"],
                    "official_blockers": [],
                    "bound_ref": bound["relative_path"],
                    "bound_sha256": bound["sha256"],
                    "hardware_model_ref": hardware_model.name,
                    "hardware_model_sha256": hashlib.sha256(
                        hardware_model.read_bytes()
                    ).hexdigest(),
                }
            )
    included = sum(len(rows) for rows in grouped.values())
    subset = ReadySubset(
        created_at=utc_timestamp(),
        dataset_root=dataset_root.as_posix(),
        selected_categories=tuple(
            sorted({str(row["category"]) for row in coverage["workloads"]})
        ),
        denominator=ReadySubsetDenominator(
            total_workloads=len(coverage["workloads"]),
            included_workloads=included,
            excluded_workloads=len(exclusions),
        ),
        included_workloads=included,
        excluded_workloads=len(exclusions),
        problems=[
            ReadySubsetProblemRef(
                category=meta[problem_id][0],
                problem_id=problem_id,
                problem_path=meta[problem_id][1],
                workloads=sorted(
                    rows, key=lambda item: (item.row_index, item.uuid or "")
                ),
            )
            for problem_id, rows in sorted(grouped.items())
        ],
        exclusions=sorted(
            exclusions,
            key=lambda item: (
                item.problem_id,
                item.row_index,
                item.workload_uuid or "",
            ),
        ),
        claim_boundary=ReadySubsetClaimBoundary(ready_to_attempt_rocm_execution=True),
    ).with_checksum()
    authority = {
        "schema_version": "sol_execbench.full_suite_authority_inputs.v1",
        "suite_manifest_sha256": manifest["payload_sha256"],
        "coverage_sha256": coverage["payload_sha256"],
        "workloads": sorted(
            authority_rows,
            key=lambda item: (str(item["definition"]), str(item["workload_uuid"])),
        ),
    }
    authority["payload_sha256"] = hashlib.sha256(
        json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return subset, authority


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("closure_dir", type=Path)
    parser.add_argument("--bounds-index", type=Path)
    parser.add_argument("--hardware-model", type=Path)
    parser.add_argument("--additional-blockers", type=Path, action="append", default=[])
    args = parser.parse_args()
    closure_dir = args.closure_dir
    manifest = _load(closure_dir / "canonical-suite.json")
    coverage = _load(closure_dir / "authority-coverage.json")
    requirements = _load(closure_dir / "hardware-profile-requirements.json")
    if (args.bounds_index is None) != (args.hardware_model is None):
        raise ValueError(
            "--bounds-index and --hardware-model must be supplied together"
        )
    bounds_by_key = None
    additional_blockers_by_key: dict[tuple[str, str], tuple[str, ...]] = {}
    for blocker_path in args.additional_blockers:
        payload = _load(blocker_path)
        rows = payload.get("workloads")
        if not isinstance(rows, list):
            raise ValueError("additional blockers must contain a workloads list")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("additional blocker workload must be an object")
            key = (str(row["definition"]), str(row["workload_uuid"]))
            codes = row.get("blocker_codes")
            if not isinstance(codes, list) or not all(
                isinstance(code, str) for code in codes
            ):
                raise ValueError(
                    "additional blocker workload has invalid blocker_codes"
                )
            additional_blockers_by_key[key] = tuple(
                dict.fromkeys((*additional_blockers_by_key.get(key, ()), *codes))
            )
    if args.bounds_index is not None:
        index = _load(args.bounds_index)
        raw_bounds = index.get("bounds")
        if not isinstance(raw_bounds, list):
            raise ValueError("bounds index must contain a bounds list")
        bounds_by_key = {
            (str(item["definition"]), str(item["workload_uuid"])): {
                "relative_path": str(item["relative_path"]),
                "sha256": str(item["sha256"]),
            }
            for item in raw_bounds
            if isinstance(item, dict)
        }
    subset, authority = build_inputs(
        dataset_root=args.dataset_root,
        manifest=manifest,
        coverage=coverage,
        requirements=requirements,
        bounds_by_key=bounds_by_key,
        hardware_model=args.hardware_model,
        additional_blockers_by_key=additional_blockers_by_key,
    )
    _write(closure_dir / "authority-ready-subset.json", stable_model_json(subset))
    _write(
        closure_dir / "authority-inputs.json",
        json.dumps(authority, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "included_workloads": subset.included_workloads,
                "excluded_workloads": subset.excluded_workloads,
                "problems": len(subset.problems),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
