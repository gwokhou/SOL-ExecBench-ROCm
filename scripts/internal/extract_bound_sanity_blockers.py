#!/usr/bin/env python3
"""Extract invalid-SOL workload blockers from official-score evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.evidence.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "sol_execbench.official_score_evidence.v1":
        raise ValueError(
            "bound sanity blockers must be extracted from "
            "sol_execbench.official_score_evidence.v1"
        )
    rows = []
    for item in payload.get("scores", []):
        codes = set(item.get("blocker_reason_codes", []))
        invalid = sorted(
            codes & {"baseline_not_slower_than_sol_bound", "candidate_below_sol_bound"}
        )
        if invalid:
            rows.append(
                {
                    "definition": item["definition"],
                    "workload_uuid": item["workload_uuid"],
                    "blocker_codes": ["sol_bound_sanity_failed", *invalid],
                }
            )
    result = {
        "schema_version": "sol_execbench.bound_sanity_blockers.v1",
        "workloads": rows,
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"blockers": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
