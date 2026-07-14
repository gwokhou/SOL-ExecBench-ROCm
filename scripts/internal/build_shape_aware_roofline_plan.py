#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shard shape-aware roofline collection by exact authority profile.

The plan contains no synthetic performance values.  It is a checksummed work
queue: a collector must run every listed workload/profile shard and retain raw
shape/layout/launch/occupancy evidence before it can create a validated
``sol_execbench.shape_aware_roofline.v1`` artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sol_execbench.core.integrity.checksums import sha256_file
from sol_execbench.core.scoring.hardware_profile_requirements import (
    hardware_profile_requirements_from_dict,
)


PLAN_SCHEMA_VERSION = "sol_execbench.shape_aware_roofline_plan.v1"
_DIMENSIONS = ["shape", "layout", "launch", "occupancy"]


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def build_plan(
    coverage: dict[str, Any],
    requirements: dict[str, Any],
    *,
    coverage_sha256: str,
    requirements_sha256: str,
) -> dict[str, Any]:
    """Build a stable, profile-sharded collection plan from authority coverage."""
    if coverage.get("schema_version") != "sol_execbench.full_suite_coverage.v3":
        raise ValueError("shape-aware plan requires full-suite coverage v3")
    if coverage.get("payload_sha256") != _digest(
        {key: value for key, value in coverage.items() if key != "payload_sha256"}
    ):
        raise ValueError("authority coverage checksum mismatch")
    parsed_requirements = hardware_profile_requirements_from_dict(requirements)
    if requirements.get("payload_sha256") != requirements_sha256:
        raise ValueError("requirements file checksum mismatch")
    if coverage.get("architecture") != parsed_requirements.architecture:
        raise ValueError("coverage and requirements architectures do not match")
    rows = coverage.get("workloads")
    if not isinstance(rows, list):
        raise ValueError("coverage workloads must be a list")

    shards: dict[str, list[dict[str, str]]] = {
        key: [] for key in parsed_requirements.required_profile_keys
    }
    eligible = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("coverage workload must be an object")
        blockers = row.get("blocker_codes")
        profile_keys = row.get("authority_profile_keys")
        if not isinstance(blockers, list) or not isinstance(profile_keys, list):
            raise ValueError("coverage workload has no authority profile manifest")
        if blockers:
            if profile_keys:
                raise ValueError("blocked workload cannot have authority profiles")
            continue
        eligible += 1
        identity = {
            "definition": _string(row, "definition"),
            "workload_uuid": _string(row, "workload_uuid"),
            "problem_id": _string(row, "problem_id"),
        }
        if not profile_keys:
            raise ValueError("authority workload has no required profile")
        for key in profile_keys:
            if not isinstance(key, str) or key not in shards:
                raise ValueError(
                    f"authority workload references profile outside requirements: {key!r}"
                )
            shards[key].append(identity)
    missing = [key for key, rows in shards.items() if not rows]
    if missing:
        raise ValueError(
            "required profiles have no authority workload shard: " + ", ".join(missing)
        )
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "architecture": parsed_requirements.architecture,
        "authority_coverage_sha256": coverage_sha256,
        "requirements_sha256": requirements_sha256,
        "required_dimensions": _DIMENSIONS,
        "authority_workload_count": eligible,
        "profile_shards": [
            {
                "profile_key": key,
                "workloads": sorted(
                    values,
                    key=lambda item: (
                        item["problem_id"],
                        item["definition"],
                        item["workload_uuid"],
                    ),
                ),
            }
            for key, values in sorted(shards.items())
        ],
    }
    payload["payload_sha256"] = _digest(payload)
    return payload


def _string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"coverage workload field {key!r} must be non-empty")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-coverage", required=True, type=Path)
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    requirements = _load(args.requirements)
    requirements_payload_sha256 = requirements.get("payload_sha256")
    if not isinstance(requirements_payload_sha256, str):
        raise ValueError("requirements artifact has no payload checksum")
    plan = build_plan(
        _load(args.authority_coverage),
        requirements,
        coverage_sha256=sha256_file(args.authority_coverage),
        requirements_sha256=requirements_payload_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "authority_workloads": plan["authority_workload_count"],
                "profile_shards": len(plan["profile_shards"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
