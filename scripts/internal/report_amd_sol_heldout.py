#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Gate AMD SOL publication on held-out external measurement contradictions.

The JSONL input is deliberately provider-neutral.  Each row contains a bound
identity plus the complete ``PerformanceProviderResult`` payload emitted by an
external compiler/search provider (for example TorchInductor, Origami or
rocMLIR).  Measurements remain diagnostics: this script can only reject a
floor that a faster held-out implementation disproves; it never upgrades one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sol_execbench.core.scoring.amd_sol import (
    PerformanceProviderResult,
    amd_sol_bound_from_dict,
)


def _sha256_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_bound_artifacts(path: Path) -> dict[tuple[str, str], object]:
    artifacts: dict[tuple[str, str], object] = {}
    for artifact_path in sorted(path.rglob("*.amd-sol.json")):
        artifact = amd_sol_bound_from_dict(
            json.loads(artifact_path.read_text(encoding="utf-8"))
        )
        key = (artifact.definition, str(artifact.workload_uuid))
        if key in artifacts:
            raise ValueError(f"duplicate bound artifact for {key}")
        artifacts[key] = artifact
    if not artifacts:
        raise ValueError(f"no AMD SOL artifacts found below {path}")
    return artifacts


def _measurement_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        required = {"definition", "workload_uuid", "provider_result"}
        if set(row) != required:
            raise ValueError(
                f"{path}:{line_number} must contain exactly {sorted(required)}"
            )
        if not isinstance(row["definition"], str) or not isinstance(
            row["workload_uuid"], str
        ):
            raise ValueError(f"{path}:{line_number} has invalid workload identity")
        if not isinstance(row["provider_result"], dict):
            raise ValueError(f"{path}:{line_number} provider_result must be an object")
        result = PerformanceProviderResult(**row["provider_result"])
        if result.measured_latency_ms is None:
            raise ValueError(
                f"{path}:{line_number} must provide measured_latency_ms for held-out gating"
            )
        rows.append(
            {
                "definition": row["definition"],
                "workload_uuid": row["workload_uuid"],
                "provider_result": result,
            }
        )
    if not rows:
        raise ValueError(f"{path} contains no held-out measurements")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounds-dir", required=True, type=Path)
    parser.add_argument("--held-out-jsonl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fail-on-contradiction", action="store_true")
    args = parser.parse_args()

    artifacts = _load_bound_artifacts(args.bounds_dir)
    contradictions: list[dict[str, object]] = []
    unmatched: list[dict[str, object]] = []
    matched = 0
    for row in _measurement_rows(args.held_out_jsonl):
        key = (row["definition"], row["workload_uuid"])
        artifact = artifacts.get(key)
        if artifact is None:
            result = row["provider_result"]
            assert isinstance(result, PerformanceProviderResult)
            unmatched.append(
                {
                    "definition": key[0],
                    "workload_uuid": key[1],
                    "provider_name": result.provider_name,
                    "provider_revision": result.provider_revision,
                    "raw_evidence_ref": result.raw_evidence_ref,
                    "raw_evidence_sha256": result.raw_evidence_sha256,
                    "reason": "missing_v5_bound_artifact",
                }
            )
            continue
        result = row["provider_result"]
        assert isinstance(result, PerformanceProviderResult)
        if result.target_architecture != artifact.hardware_model.architecture:
            raise ValueError(f"held-out provider architecture does not match {key}")
        matched += 1
        assert result.measured_latency_ms is not None
        if artifact.t_sol_floor_ms > result.measured_latency_ms:
            contradictions.append(
                {
                    "definition": artifact.definition,
                    "workload_uuid": str(artifact.workload_uuid),
                    "t_sol_floor_ms": artifact.t_sol_floor_ms,
                    "fastest_known_ms": result.measured_latency_ms,
                    "ratio": artifact.t_sol_floor_ms / result.measured_latency_ms,
                    "provider_name": result.provider_name,
                    "provider_revision": result.provider_revision,
                    "raw_evidence_ref": result.raw_evidence_ref,
                    "raw_evidence_sha256": result.raw_evidence_sha256,
                }
            )
    payload: dict[str, object] = {
        "schema_version": "sol_execbench.amd_sol_heldout_contradictions.v1",
        "bounds_dir": str(args.bounds_dir),
        "held_out_jsonl": str(args.held_out_jsonl),
        "matched_measurements": matched,
        "unmatched_measurement_count": len(unmatched),
        "contradiction_count": len(contradictions),
        "publication_eligible": not contradictions and not unmatched,
        "contradictions": sorted(
            contradictions,
            key=lambda item: (-float(item["ratio"]), str(item["workload_uuid"])),
        ),
        "unmatched_measurements": sorted(
            unmatched,
            key=lambda item: (str(item["definition"]), str(item["workload_uuid"])),
        ),
    }
    payload["checksum_sha256"] = _sha256_payload(payload)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"contradictions": len(contradictions), "output": str(args.output)})
    )
    return 1 if (contradictions or unmatched) and args.fail_on_contradiction else 0


if __name__ == "__main__":
    raise SystemExit(main())
