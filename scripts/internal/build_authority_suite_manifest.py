#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Build a canonical suite manifest containing only unblocked authority rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical_suite", type=Path)
    parser.add_argument("authority_input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--filtered-authority-output", type=Path)
    args = parser.parse_args()

    canonical = _load(args.canonical_suite)
    authority = _load(args.authority_input)
    authority_rows = authority.get("workloads")
    canonical_rows = canonical.get("workloads")
    if not isinstance(authority_rows, list) or not isinstance(canonical_rows, list):
        raise ValueError("suite and authority inputs must contain workload lists")
    eligible = {
        (str(row["definition"]), str(row["workload_uuid"]))
        for row in authority_rows
        if isinstance(row, dict) and row.get("official_blockers") == []
    }
    rows = [
        row
        for row in canonical_rows
        if isinstance(row, dict)
        and (str(row.get("definition")), str(row.get("workload_uuid"))) in eligible
    ]
    identities = {(str(row["definition"]), str(row["workload_uuid"])) for row in rows}
    if identities != eligible or len(rows) != len(identities):
        raise ValueError("canonical suite does not exactly cover authority identities")
    payload: dict[str, Any] = {
        "schema_version": "sol_execbench.authority_suite_manifest.v1",
        "source_suite_manifest_sha256": canonical.get("payload_sha256"),
        "workloads": rows,
    }
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.filtered_authority_output is not None:
        filtered_authority = {
            key: value
            for key, value in authority.items()
            if key not in {"workloads", "payload_sha256"}
        }
        filtered_authority["workloads"] = [
            row
            for row in authority_rows
            if isinstance(row, dict)
            and (str(row.get("definition")), str(row.get("workload_uuid"))) in eligible
        ]
        filtered_authority["payload_sha256"] = hashlib.sha256(
            json.dumps(
                filtered_authority, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        args.filtered_authority_output.parent.mkdir(parents=True, exist_ok=True)
        args.filtered_authority_output.write_text(
            json.dumps(filtered_authority, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"output": str(args.output), "workloads": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
